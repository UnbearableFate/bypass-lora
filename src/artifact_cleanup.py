"""Inventory and clean up large experiment artifacts of the research project.

`plan` (the default) is a dry run: it reads an `rl export` snapshot of the records and the artifact areas
on disk, classifies every path as keep, hold or candidate, and writes plan.json and summary.md into a new
output directory. It changes nothing on disk.

`apply` deletes the candidates of one reviewed plan, and only with --confirm <sha256 of plan.json>. It
first re-plans the same scope from fresh records and the current disk, and refuses (deleting nothing)
when any candidate changed, disappeared, or is no longer a candidate. Every deletion goes to a JSON Lines
log next to the plan.

Like every Python workload of this project it runs in a PBS job, never on the login node:

    rl export --out <dir>                    # on the login node: one consistent read of the records
    python src/artifact_cleanup.py plan --root <workspace> --records <dir>/snapshot.json \
        --scope H-0002 --out <new output dir>
    python src/artifact_cleanup.py apply --plan <output dir>/plan.json \
        --records <fresh export>/snapshot.json --confirm <plan sha256>

Decisions:
  keep       never deleted by this tool: records, ledger, data, worktrees, small metadata and result files,
             predictions, generated data, files no rule covers.
  hold       of a deletable type, but still needed: an unfinished attempt, an open experiment or task, a
             lock, a contract, a reopen condition or a start checkpoint names it. Each hold names the
             event after which it may go.
  candidate  of a deletable type and needed by nothing open: checkpoints, optimizer states, intermediate
             checkpoints, regression dumps, feature caches, pytest temp output, large training traces and
             logs, bytecode caches inside run directories, unused code snapshots and environments.

Paths that live texts name relative to the project (research/artifacts/T-0020/confirmation-lock.json) are
looked up in the main checkout and in every researcher worktree (rl puts them in <project>-worktrees/), since
a lock is often committed only on the task branch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import socket
import stat
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

PLAN_FORMAT = "artifact-cleanup-plan/1"
KEEP, HOLD, CANDIDATE = "keep", "hold", "candidate"
CLOSED_TASK_STATES = frozenset({"accepted"})  # the coordinator's rule: a stopped task still blocks a scope
FINISHED_ATTEMPT_STATES = frozenset({"ended"})
LIVE_DISPOSITIONS = frozenset({"active", "candidate"})  # a hypothesis that may still get work
LARGE_BYTES = 1 << 20  # training traces and logs from this size on are deletable
RECENT_SECONDS = 24 * 3600  # an unreferenced snapshot younger than this may belong to an attempt being registered
LOCK_MAX_BYTES = 4 << 20  # locks and the research documents an open task names are read up to this size
LOCK_MAX_DEPTH = 5
DOC_SUFFIXES = frozenset({".md", ".json", ".txt"})
HASH_CHUNK = 8 << 20

# what a file or directory is, and why a deletable type may go once nothing open needs it
CLASSES = {
    "checkpoint": "trained weights of a finished attempt; its manifest, metrics and predictions stay",
    "optimizer": "optimizer/scheduler/RNG resume state of a finished attempt",
    "intermediate-checkpoint": "intermediate step checkpoint, superseded by the final checkpoint",
    "regression-dump": "old/new output tensors of a regression check; the verdict file stays",
    "feature-cache": "cached feature tensors; the analysis report that used them stays",
    "test-temp": "pytest temporary output copied into the run directory; the test verdicts stay",
    "trace": "large training trace or log; manifests and metrics keep the summary",
    "cache": "Python bytecode or pytest cache; regenerated on demand",
    "snapshot": "read-only code snapshot of finished attempts; the commit is in the git repository",
    "environment": "Python environment of a uv.lock that no open work uses; uv sync --frozen rebuilds it",
}
DELETABLE = frozenset(CLASSES)
KEEP_RULES = {
    "records": "research records, notes, reviews and sealed documents; rl and the reviews refer to them",
    "ledger": "miyabi-mcp ledger, plans, submissions and PBS logs: job accounting owned by the executor",
    "data": "input data (data/v5-sealed is the sealed confirmation data)",
    "worktree": "working tree of a researcher or agent session",
    "tooling": "the project's rl tooling environment (.venv)",
    "metadata": "small metadata or result file (manifest, metrics, config, report, small log)",
    "predictions": "per-example predictions: results and evidence are computed from them",
    "data-copy": "generated data in a run directory: later runs name the producing attempt as data source",
    "unclassified": "no cleanup rule covers this file type: review by hand",
    "link": "symbolic link: never followed or deleted",
    "unregistered": "directory not in the records (an attempt being registered, or a stray directory)",
    "recent-snapshot": "snapshot staged in the last 24 h that no attempt references yet",
    "snapshot-no-commit": "its commit is not in the git repository: this snapshot is the only copy",
    "other": "not an attempt, snapshot or environment directory",
}
RULE_IDS = {name: f"C{i}-{name}" for i, name in enumerate(CLASSES, 1)}
RULE_IDS.update({name: f"K{i}-{name}" for i, name in enumerate(KEEP_RULES, 1)})

SNAPSHOT_DIR = re.compile(r"[0-9a-f]{40}")
ENV_DIR = re.compile(r"([0-9a-f]{12})-[\w.-]+")
_ID = re.compile(r"(?<![A-Za-z])(ATT|RUN)-?(\d{4})(?:-(\d+))?")
_RANGE = re.compile(r"(?<![A-Za-z])(?:ATT|RUN)-?(\d{4})(?:-\d+)?\s*(?:\.\.|–|—|~|～|至|to)\s*(?:(?:ATT|RUN)-?)?(\d{4})")
_REL_PATH = re.compile(r"(?<![\w./-])((?:runs|snapshots|envs|data|research)/[\w./@+=-]+)", re.ASCII)
_INTERMEDIATE_DIR = re.compile(r"(?:step|checkpoint|ckpt|epoch|iter)[-_]?\d+", re.IGNORECASE)
_OPTIMIZER = re.compile(r"(?:optimizer|optim|scheduler|scaler|rng_state)[\w.-]*\.(?:pt|pth|bin)", re.IGNORECASE)
_FEATURES = re.compile(r"(?:features?|activations?|hidden[-_]?states?|embeddings?)[\w.-]*"
                       r"\.(?:pt|pth|npy|npz|safetensors)", re.IGNORECASE)
_CHECKPOINT_FILE = re.compile(r"checkpoint[\w.-]*\.(?:pt|pth|bin)", re.IGNORECASE)
CHECKPOINT_NAMES = frozenset({"adapter.pt", "adapter.bin", "adapter_model.bin", "adapter_model.safetensors", "model.pt",
                              "model.bin", "model.safetensors", "pytorch_model.bin", "weights.pt", "state_dict.pt"})
TENSOR_SUFFIXES = frozenset({".pt", ".pth", ".bin", ".npy", ".npz", ".safetensors", ".ckpt"})
LOG_SUFFIXES = frozenset({".log", ".out", ".err"})
DATA_STEMS = frozenset({"train", "dev", "calibration", "test", "validation", "valid", "confirmation"})
CACHE_DIRS = frozenset({"__pycache__", ".pytest_cache"})


class CleanupError(Exception):
    """An input problem or a refusal; the message says what to change."""


# --------------------------------------------------------------------------------------------- records


def _decode(value):
    """rl export stores nested objects as JSON text; rl show returns them decoded."""
    if isinstance(value, str) and value[:1] in "{[":
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


@dataclass
class Records:
    path: str
    sha256: str
    meta: dict
    nodes: dict  # label -> {id: properties}
    edges: list  # (source, relation, target), relation in lower case

    def label(self, name: str) -> dict:
        return self.nodes.get(name, {})

    def targets(self, source: str, relation: str) -> list:
        return [t for s, r, t in self.edges if s == source and r == relation]

    def sources(self, target: str, relation: str) -> list:
        return [s for s, r, t in self.edges if t == target and r == relation]


def load_records(path) -> Records:
    raw = Path(path).read_bytes()
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise CleanupError(f"{path}: not JSON ({exc})") from None
    if not isinstance(data, dict) or not isinstance(data.get("nodes"), dict) or not isinstance(data.get("edges"), list):
        raise CleanupError(f"{path}: not an rl export snapshot (snapshot.json with nodes and edges)")
    nodes = {}
    for label, rows in data["nodes"].items():
        table = nodes.setdefault(label, {})
        for props in rows:
            table[props["id"]] = {k: _decode(v) for k, v in props.items()}
    edges = sorted({(e["source"], str(e["edge"]).lower(), e["target"]) for e in data["edges"]})
    return Records(str(path), hashlib.sha256(raw).hexdigest(), data.get("meta") or {}, nodes, edges)


def _text(value) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)


# --------------------------------------------------------------------------------------------- disk facts


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(HASH_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def stat_fingerprint(st: os.stat_result) -> str:
    return f"{st.st_size}:{st.st_mtime_ns}:{st.st_ctime_ns}:{st.st_ino}"


def _lstat(path: Path):
    """lstat, or None when the file vanished meanwhile (the project is live while it is planned)."""
    try:
        return os.lstat(path)
    except FileNotFoundError:
        return None


def _readlink(path: Path) -> str:
    try:
        return os.readlink(path)
    except OSError:
        return "?"


def tree_facts(path: Path, content: bool) -> dict:
    """Bytes, files and a fingerprint of a directory tree; links are recorded, never followed."""
    total, files, newest, lines = 0, 0, 0, []
    for dirpath, dirnames, filenames in os.walk(path):
        dirnames.sort()
        base = Path(dirpath)
        for name in dirnames:
            if (base / name).is_symlink():
                lines.append(f"L {(base / name).relative_to(path)} -> {_readlink(base / name)}")
        for name in sorted(filenames):
            p = base / name
            st = _lstat(p)
            if st is None:
                continue
            rel = p.relative_to(path)
            newest = max(newest, st.st_mtime_ns)
            if stat.S_ISLNK(st.st_mode):
                lines.append(f"L {rel} -> {_readlink(p)}")
                continue
            total += st.st_size
            files += 1
            lines.append(f"F {rel} {st.st_size} {sha256_file(p) if content else st.st_mtime_ns}")
    digest = hashlib.sha256("\n".join(lines).encode("utf-8", "surrogateescape")).hexdigest()
    return {"bytes": total, "files": files, "newest_mtime_ns": newest,
            "fingerprint": {("sha256" if content else "listing"): digest}}


class Git:
    """Read-only facts from the project's git repository (commit presence, the uv.lock id of a checkout's HEAD)."""

    def __init__(self, root: Path):
        self.root = root

    def _run(self, args, stdin="", at: Path | None = None):
        try:
            done = subprocess.run(["git", "-C", str(at or self.root), *args], input=stdin, capture_output=True,
                                  text=True, timeout=120, check=False)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return done.stdout if done.returncode == 0 else None

    def commits_present(self, shas) -> dict:
        shas = sorted(shas)
        if not shas:
            return {}
        out = self._run(["cat-file", "--batch-check"], "".join(f"{s}^{{commit}}\n" for s in shas))
        if out is None:
            return {s: None for s in shas}  # unknown: never a candidate
        lines = out.splitlines()
        return {s: (len(line.split()) == 3 and line.split()[1] == "commit") for s, line in zip(shas, lines)}

    def head_lock_id(self, at: Path | None = None):
        """The environment id of a checkout (default: the main one): the first 12 digits of HEAD:uv.lock."""
        out = self._run(["rev-parse", "HEAD:uv.lock"], at=at)
        return out.strip()[:12] if out else None


# --------------------------------------------------------------------------------------------- holds


@dataclass(frozen=True)
class Hold:
    kind: str  # unfinished-attempt, open-experiment, open-task, contract, reopen-condition, lock, start-checkpoint,
    #            relation, environment, scope-open
    source: str  # the record or file that holds it
    reason: str
    release: str  # the event after which it may go

    def as_dict(self) -> dict:
        return {"kind": self.kind, "source": self.source, "reason": self.reason, "release": self.release}


def _is_lock(path: Path) -> bool:
    """A lock file by name (lr-lock.json, confirmation-lock.json): it pins what a later step will read."""
    return "lock" in path.name.lower() and path.suffix == ".json"


def _ids_in_text(text: str) -> tuple[set, set]:
    """Attempt ids (ATT-0001-1) and run numbers (0001) named in text, ranges such as ATT-0080-1..ATT-0089-1 expanded."""
    attempts, runs = set(), set()
    for kind, number, index in _ID.findall(text):
        if kind == "ATT" and index:
            attempts.add(f"ATT-{number}-{index}")
        else:
            runs.add(number)  # RUN-0001, or ATT0179 without an index: every attempt of that run
    for first, last in _RANGE.findall(text):
        lo, hi = int(first), int(last)
        if 0 < hi - lo <= 200:
            runs.update(f"{n:04d}" for n in range(lo, hi + 1))
    return attempts, runs


def default_worktrees(root: Path) -> Path | None:
    """rl's researcher worktrees: <project>/../<project>-worktrees/, when it exists."""
    root = Path(os.path.abspath(root))
    path = root.parent / f"{root.name}-worktrees"
    return path if path.is_dir() else None


class Model:
    """What the records and the referenced files say is still needed."""

    def __init__(self, records: Records, root: Path, git: Git | None = None, worktrees: Path | None = None):
        self.records = records
        self.root = Path(os.path.abspath(root))
        self.git = git or Git(self.root)
        self.worktrees = Path(os.path.abspath(worktrees)) if worktrees else None
        # where a project-relative path may live: the main checkout, then each researcher worktree (a git checkout)
        self.checkouts = [self.root]
        if self.worktrees and self.worktrees.is_dir():
            self.checkouts += [d for d in sorted(self.worktrees.iterdir())
                               if d.is_dir() and not d.is_symlink() and (d / ".git").exists()]
        self._abs_path = re.compile("|".join(re.escape(str(b)) + r"/[\w./@+=-]+"
                                             for b in [self.root] + ([self.worktrees] if self.worktrees else [])),
                                    re.ASCII)
        self.tasks = records.label("Task")
        self.experiments = records.label("Experiment")
        self.hypotheses = records.label("Hypothesis")
        self.decisions = records.label("Decision")
        self.runs = records.label("Run")
        self.results = records.label("Result")
        self.attempts = records.label("Attempt")
        self.exp_tasks = defaultdict(list)
        for t in self.tasks:
            for e in records.targets(t, "targets"):
                self.exp_tasks[e].append(t)
        self.exp_hyps = {e: sorted(set(records.targets(e, "tests"))) for e in self.experiments}
        self.run_exp = {r: (p.get("experiment_id") or next(iter(records.targets(r, "of_experiment")), None))
                        for r, p in self.runs.items()}
        self.att_run = {a: (p.get("run_id") or next(iter(records.targets(a, "of_run")), None))
                        for a, p in self.attempts.items()}
        self.run_atts = defaultdict(list)
        self.exp_atts = defaultdict(list)
        for a, r in sorted(self.att_run.items()):
            self.run_atts[r].append(a)
            self.exp_atts[self.run_exp.get(r)].append(a)
        self.cited = defaultdict(set)  # absolute path -> {"RES-x (role)"}
        for res_id, res in self.results.items():
            manifest = res.get("manifest") if isinstance(res.get("manifest"), dict) else {}
            for item in list(manifest.get("artifacts") or []) + list(res.get("verification") or []):
                if isinstance(item, dict) and isinstance(item.get("path"), str):
                    self.cited[self._abs(item["path"])].add(f"{res_id} ({item.get('role', '?')})")
        self.snapshot_names = sorted(p.name for p in self._children("snapshots") if SNAPSHOT_DIR.fullmatch(p.name))
        self.env_names = sorted(p.name for p in self._children("envs") if ENV_DIR.fullmatch(p.name))
        self.attempt_holds = defaultdict(list)  # a hold always covers the whole run directory of the attempt
        self.snapshot_holds = defaultdict(list)
        self.env_holds = defaultdict(list)
        self.locks = defaultdict(list)  # lock path -> the holds that made it live
        self.sources = []  # what was checked, for the plan header
        self._collect()

    # -- small helpers

    def _children(self, area: str) -> list:
        base = self.root / area
        return sorted(base.iterdir()) if base.is_dir() else []

    def _abs(self, path: str) -> str:
        p = Path(path)
        return os.path.normpath(p if p.is_absolute() else self.root / p)

    def _rel(self, path: Path) -> str:
        return os.path.relpath(path, self.root) if self.root in path.parents else str(path)

    def paths_named(self, text: str) -> list:
        """Files and directories a text names: absolute paths under the project or the worktrees, and paths
        relative to the project (runs/, snapshots/, envs/, data/, research/) in every checkout."""
        found = {Path(os.path.normpath(m.group(0).rstrip("."))) for m in self._abs_path.finditer(text)}
        for raw in _REL_PATH.findall(text):
            found.update(Path(os.path.normpath(base / raw.rstrip("."))) for base in self.checkouts)
        return sorted(found)

    def _research_doc(self, path: Path) -> bool:
        """A small research document (note, brief, receipt, lock) in some checkout; read when a live text names it."""
        if path.suffix not in DOC_SUFFIXES or not any(base / "research" in path.parents for base in self.checkouts):
            return False
        try:
            return path.is_file() and path.stat().st_size <= LOCK_MAX_BYTES
        except OSError:
            return False

    def attempt_finished(self, a: str) -> bool:
        return self.attempts[a].get("execution_state") in FINISHED_ATTEMPT_STATES

    def attempt_state(self, a: str) -> str:
        p = self.attempts[a]
        return f"{p.get('execution_state')}/{p.get('outcome')}/result:{p.get('result_status')}"

    def attempt_snapshot(self, a: str):
        resource = self.attempts[a].get("resource")
        if isinstance(resource, dict) and resource.get("snapshot"):
            return Path(resource["snapshot"]).name
        spec = self.runs.get(self.att_run.get(a), {}).get("spec")
        code = spec.get("code_ref") if isinstance(spec, dict) else None
        return code.get("commit") if isinstance(code, dict) else None

    def spec(self, a: str) -> dict:
        spec = self.runs.get(self.att_run.get(a), {}).get("spec")
        return spec if isinstance(spec, dict) else {}

    def experiment_open(self, e: str) -> list:
        """Why experiment e is still open (empty: closed)."""
        why = [f"{t} {self.tasks[t].get('state')}" for t in sorted(self.exp_tasks.get(e, []))
               if self.tasks[t].get("state") not in CLOSED_TASK_STATES]
        why += [f"{a} {self.attempts[a].get('execution_state')}" for a in self.exp_atts.get(e, [])
                if not self.attempt_finished(a)]
        if not self.exp_tasks.get(e):
            why.append(f"{e} has no task yet")
        return why

    @staticmethod
    def _release(e: str, why: list) -> str:
        return f"{e} closed: " + ", ".join(w.replace(" has no task yet", " gets a task") for w in why) + \
            " -> accepted/ended"

    def _read(self, path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    # -- references named in text

    def scan(self, text: str, hold: Hold, depth: int = 0):
        """Hold what a live text names: attempts (by id, run id or a path inside runs/<ATT>/), snapshots (a commit
        prefix of 7 or more hex digits, or a path), environments (the 12-digit uv.lock id) and, recursively, the
        content of every lock file (*lock*.json) it names."""
        attempts, runs = _ids_in_text(text)
        for number in runs:
            attempts.update(self.run_atts.get(f"RUN-{number}", []))
        for a in sorted(attempts):
            if a in self.attempts:
                self.attempt_holds[a].append(hold)
        lowered = text.lower()
        for sha in self.snapshot_names:
            if sha[:7] in lowered:
                self.snapshot_holds[sha].append(hold)
        for name in self.env_names:
            if name[:12] in lowered:
                self.env_holds[name].append(hold)
        for path in self.paths_named(text):
            if _is_lock(path):
                self._scan_lock(path, hold, depth)

    def _scan_lock(self, path: Path, hold: Hold, depth: int):
        if depth >= LOCK_MAX_DEPTH or any(h.release == hold.release for h in self.locks.get(str(path), [])):
            return
        try:
            if not path.is_file() or path.stat().st_size > LOCK_MAX_BYTES:
                return
        except OSError:
            return
        self.locks[str(path)].append(hold)
        rel = self._rel(path)
        self.scan(self._read(path), Hold("lock", rel, f"pinned by lock {rel} ({hold.reason})", hold.release),
                  depth + 1)

    # -- the live sources

    def _collect(self):
        for a in sorted(self.attempts):  # unfinished attempts and what their runs read
            if not self.attempt_finished(a):
                hold = Hold("unfinished-attempt", a, f"{a} is {self.attempt_state(a)}", f"{a} ended")
                self.attempt_holds[a].append(hold)
                self.scan(_text(self.spec(a)), hold)
                self.sources.append(f"{a} {self.attempt_state(a)}")
        for e in sorted(self.experiments):  # open experiments: their attempts, contract, run specs, results
            why = self.experiment_open(e)
            if not why:
                continue
            release = self._release(e, why)
            hold = Hold("open-experiment", e, f"{e} is open ({', '.join(why)})", release)
            self.sources.append(f"{e} open: {', '.join(why)}")
            for a in self.exp_atts.get(e, []):
                self.attempt_holds[a].append(hold)
                self.scan(_text(self.spec(a)), hold)
            for res in self.results.values():
                if self.att_run.get(res.get("attempt_id")) and self.run_exp.get(self.att_run[res["attempt_id"]]) == e:
                    self.scan(_text(res.get("manifest")) + _text(res.get("verification")), hold)
            path = self.experiments[e].get("contract_path")
            if path:
                self.scan(self._read(self.root / path), Hold("contract", e, f"named in the contract of open {e}",
                                                             release))
            for relation in ("depends_on", "code_from"):
                for target in self.records.targets(e, relation):
                    for a in self.exp_atts.get(target, []):
                        self.attempt_holds[a].append(
                            Hold("relation", e, f"open {e} {relation} {target}", release))
        for e in sorted(self.experiments):  # contracts of experiments that test an active hypothesis
            active = [h for h in self.exp_hyps.get(e, [])
                      if self.hypotheses.get(h, {}).get("disposition") in LIVE_DISPOSITIONS]
            path = self.experiments[e].get("contract_path")
            if active and path and not self.experiment_open(e):
                self.scan(self._read(self.root / path),
                          Hold("contract", e, f"named in the contract of {e}, which tests live {', '.join(active)}",
                               f"{', '.join(active)} paused or stopped"))
                self.sources.append(f"contract {path} (tests live {', '.join(active)})")
        for t in sorted(self.tasks):  # open tasks: record text, briefs, the research documents the record names,
            task = self.tasks[t]      # and the locks under research/artifacts/<T>/, in every checkout
            if task.get("state") in CLOSED_TASK_STATES:
                continue
            hold = Hold("open-task", t, f"named by open task {t} ({task.get('state')})", f"{t} accepted")
            text = " ".join(_text(task.get(k) or "") for k in ("title", "next_action", "note", "output_refs",
                                                                "wait_attempts"))
            docs = set()
            for base in self.checkouts:
                docs.update((base / "research" / "tasks").glob(f"{t}-*.md"))
                docs.update((base / "research" / "tasks").glob(f"{t}.md"))
                docs.update((base / "research" / "artifacts" / t).glob("*lock*.json"))
            docs.update(p for p in self.paths_named(text) if self._research_doc(p))
            docs = sorted(d for d in docs if self._research_doc(d))
            self.sources.append(f"{t} {task.get('state')}: record, " + ", ".join(self._rel(d) for d in docs))
            # a lock goes in by its path, so that scan() reads it as a lock (recursively, hold kind "lock")
            self.scan(text + "\n" + "\n".join(str(d) if _is_lock(d) else f"{d}\n{self._read(d)}" for d in docs), hold)
        for d in sorted(self.decisions):  # reopen conditions
            condition = self.decisions[d].get("reopen_condition")
            if condition:
                targets = ", ".join(self.records.targets(d, "decides"))
                self.scan(_text(condition), Hold("reopen-condition", d, f"named in the reopen condition of {d} "
                                                 f"({targets})", f"the reopen condition of {d} no longer applies"))
                self.sources.append(f"{d} reopen condition ({targets})")
        done, changed = set(), True  # start checkpoints of held attempts, transitively, for each release event
        while changed:
            changed = False
            for a in sorted(self.attempt_holds):
                start = self.spec(a).get("resume_from") if a in self.attempts else None
                if not start:
                    continue
                for release in sorted({h.release for h in self.attempt_holds[a]}):
                    if (a, release) not in done:
                        done.add((a, release))
                        changed = True
                        self.scan(_text(start), Hold("start-checkpoint", a,
                                                     f"start checkpoint (resume_from) of held {a}", release))

    # -- the env ids in use

    def envs_in_use(self) -> dict:
        """Environment directory -> holds, from the runs of held attempts and the uv.lock of every checkout's HEAD."""
        holds = defaultdict(list, {k: list(v) for k, v in self.env_holds.items()})
        for checkout in self.checkouts:
            head = self.git.head_lock_id(None if checkout == self.root else checkout)
            where = "the main checkout" if checkout == self.root else f"worktree {checkout.name}"
            for name in self.env_names:
                if head and name.startswith(head):
                    holds[name].append(Hold("environment", f"{checkout.name}:HEAD:uv.lock",
                                            f"uv.lock of HEAD in {where} is {head}",
                                            f"HEAD of {where} moves to another uv.lock"))
        for a in sorted(self.attempt_holds):
            ref = _text(self.spec(a).get("environment_ref") or "").lower()
            for name in self.env_names:
                if name[:12] in ref:
                    holds[name] += [Hold("environment", a, f"environment of held {a}", release)
                                    for release in sorted({h.release for h in self.attempt_holds[a]})]
        return holds


# --------------------------------------------------------------------------------------------- classification


def classify_file(rel_parts: tuple, path: Path, size: int, large_bytes: int) -> str:
    """The class of one file inside an attempt directory (rel_parts: path below runs/<ATT>/)."""
    name = rel_parts[-1]
    lower = name.lower()
    suffix = Path(lower).suffix
    dirs = [d.lower() for d in rel_parts[:-1]]
    if suffix in TENSOR_SUFFIXES and any(_INTERMEDIATE_DIR.fullmatch(d) for d in dirs):
        return "intermediate-checkpoint"
    if lower in CHECKPOINT_NAMES or suffix in (".safetensors", ".ckpt") or _CHECKPOINT_FILE.fullmatch(lower):
        return "checkpoint"
    if _OPTIMIZER.fullmatch(lower):
        return "optimizer"
    if suffix in TENSOR_SUFFIXES and "regression" in dirs:
        return "regression-dump"
    if _FEATURES.fullmatch(lower):
        return "feature-cache"
    if "predictions" in lower:
        return "predictions"
    if suffix == ".jsonl":
        if _is_step_trace(path):
            return "trace" if size >= large_bytes else "metadata"
        if Path(lower).stem in DATA_STEMS or "data" in dirs:
            return "data-copy"
        return "metadata"
    if suffix in LOG_SUFFIXES:
        return "trace" if size >= large_bytes else "metadata"
    if suffix in TENSOR_SUFFIXES:
        return "unclassified"
    return "metadata"


def _is_step_trace(path: Path) -> bool:
    """A JSON Lines training trace: its first record is an object with a "step" field."""
    try:
        with open(path, "rb") as source:
            first = source.readline(1 << 16)
        record = json.loads(first)
    except (OSError, ValueError):
        return False
    return isinstance(record, dict) and "step" in record


@dataclass
class Plan:
    scope: dict
    entries: list = field(default_factory=list)


class Planner:
    def __init__(self, model: Model, large_bytes: int = LARGE_BYTES, now: float | None = None,
                 hash_candidates: bool = True):
        self.model = model
        self.root = model.root
        self.worktrees = model.worktrees
        self.large_bytes = large_bytes
        self.now = time.time() if now is None else now
        self.hash_candidates = hash_candidates
        self.env_holds = model.envs_in_use()

    # -- scope

    def scope(self, spec: str) -> dict:
        m, rec = self.model, self.model.records
        if spec == "all":
            return {"id": "all", "experiments": sorted(m.experiments), "hypotheses": sorted(m.hypotheses),
                    "attempts": sorted(m.attempts), "tasks": [], "gate": {"open": False, "why": []}}
        ids = [s.strip() for s in spec.split(",") if s.strip()]
        experiments, hypotheses = set(), set()
        for i in ids:
            if i in m.hypotheses:
                hypotheses.add(i)
                experiments.update(rec.sources(i, "tests"))
            elif i in m.experiments:
                experiments.add(i)
                hypotheses.update(m.exp_hyps.get(i, []))
            else:
                raise CleanupError(f"scope {i}: no such hypothesis or experiment in the records")
        experiments = sorted(e for e in experiments if e in m.experiments)
        tasks = sorted({t for e in experiments for t in m.exp_tasks.get(e, [])})
        attempts = sorted(a for e in experiments for a in m.exp_atts.get(e, []))
        why = [f"{t} is {m.tasks[t].get('state')}" for t in tasks if m.tasks[t].get("state") not in CLOSED_TASK_STATES]
        why += [f"{a} is {m.attempt_state(a)}" for a in attempts if not m.attempt_finished(a)]
        why += [f"{e} has no task" for e in experiments if not m.exp_tasks.get(e)]
        warnings = [f"{h} is still {m.hypotheses[h].get('disposition')}/{m.hypotheses[h].get('evidence_status')}: "
                    f"later work on it may need these files" for h in sorted(hypotheses)
                    if m.hypotheses.get(h, {}).get("disposition") in LIVE_DISPOSITIONS]
        return {"id": ",".join(ids), "experiments": experiments, "hypotheses": sorted(hypotheses),
                "tasks": [{"id": t, "state": m.tasks[t].get("state"), "title": m.tasks[t].get("title")} for t in tasks],
                "attempts": attempts, "gate": {"open": bool(why), "why": why}, "warnings": warnings}

    # -- entries

    def _entry(self, path: Path, area: str, kind: str, cls: str, nbytes: int, files: int, **extra) -> dict:
        return {"path": str(path), "area": area, "kind": kind, "class": cls, "bytes": nbytes, "files": files, **extra}

    def _decide(self, entry: dict, holds: list, checked: list):
        cls = entry["class"]
        entry["checked"] = checked
        if cls not in DELETABLE:
            entry.update(decision=KEEP, rule=RULE_IDS[cls], reason=KEEP_RULES[cls], holds=[])
            return
        unique = sorted({h for h in holds}, key=lambda h: (h.kind, h.source, h.reason))
        entry.update(decision=HOLD if unique else CANDIDATE, rule=RULE_IDS[cls], reason=CLASSES[cls],
                     holds=[h.as_dict() for h in unique])

    @staticmethod
    def _add_content_hash(entry: dict):
        """Candidates also carry the sha256 of their content (a directory: of its file list and contents)."""
        path = Path(entry["path"])
        if entry["kind"] == "file":
            entry["fingerprint"]["sha256"] = sha256_file(path)
        else:
            entry["fingerprint"]["sha256"] = tree_facts(path, content=True)["fingerprint"]["sha256"]

    def attempt_entries(self, a_dir: Path) -> list:
        m = self.model
        a = a_dir.name
        if a not in m.attempts:
            facts = tree_facts(a_dir, content=False)
            entry = self._entry(a_dir, "runs", "dir", "unregistered", facts["bytes"], facts["files"],
                                fingerprint=facts["fingerprint"])
            self._decide(entry, [], [f"{a} not in the records"])
            return [entry]
        run = m.att_run.get(a)
        exp = m.run_exp.get(run)
        tasks = [f"{t} {m.tasks[t].get('state')}" for t in sorted(m.exp_tasks.get(exp, []))]
        base_checked = [f"{a} {m.attempt_state(a)}", str(run), str(exp)] + tasks
        attempt_holds = list(m.attempt_holds.get(a, []))
        if exp not in m.experiments:
            attempt_holds.append(Hold("open-experiment", str(exp), f"experiment of {a} not in the records",
                                      "the records name its experiment"))
        entries = []
        for dirpath, dirnames, filenames in os.walk(a_dir):
            base = Path(dirpath)
            descend = []
            for name in sorted(dirnames):
                child = base / name
                if child.is_symlink():
                    st = _lstat(child)
                    if st is not None:
                        entries.append(self._file_entry(child, "link", a, exp, attempt_holds, base_checked, st))
                elif name in CACHE_DIRS or (base.name == "tmp" and name.startswith("test_")):
                    entries.append(self._dir_entry(child, "runs", "cache" if name in CACHE_DIRS else "test-temp",
                                                   attempt_holds, base_checked, attempt=a, experiment=exp))
                else:
                    descend.append(name)
            dirnames[:] = descend
            for name in sorted(filenames):
                p = base / name
                st = _lstat(p)
                if st is None:
                    continue
                rel = p.relative_to(a_dir).parts
                cls = "link" if stat.S_ISLNK(st.st_mode) else classify_file(rel, p, st.st_size, self.large_bytes)
                entries.append(self._file_entry(p, cls, a, exp, attempt_holds, base_checked, st))
        return entries

    def _file_entry(self, p: Path, cls: str, a: str, exp, attempt_holds: list, checked: list,
                    st: os.stat_result) -> dict:
        entry = self._entry(p, "runs", "file", cls, st.st_size, 1, attempt=a, experiment=exp,
                            fingerprint={"stat": stat_fingerprint(st)})
        cites = sorted(self.model.cited.get(str(p), ()))
        self._decide(entry, attempt_holds, checked + [f"cited by {c}" for c in cites])
        return entry

    def _dir_entry(self, path: Path, area: str, cls: str, holds: list, checked: list, **extra) -> dict:
        facts = tree_facts(path, content=False)
        entry = self._entry(path, area, "dir", cls, facts["bytes"], facts["files"], fingerprint=facts["fingerprint"],
                            **extra)
        self._decide(entry, holds, checked)
        return entry

    def snapshot_entries(self, only_attempts: set | None) -> list:
        m = self.model
        users = defaultdict(list)
        for a in sorted(m.attempts):
            sha = m.attempt_snapshot(a)
            if sha:
                users[sha].append(a)
        dirs = m._children("snapshots")
        if only_attempts is not None:
            dirs = [d for d in dirs if any(a in only_attempts for a in users.get(d.name, []))]
        present = m.git.commits_present([d.name for d in dirs if SNAPSHOT_DIR.fullmatch(d.name)])
        entries = []
        for d in dirs:
            st = _lstat(d)
            if st is None:
                continue
            if not SNAPSHOT_DIR.fullmatch(d.name) or not stat.S_ISDIR(st.st_mode):
                entries.append(self._kept(d, "snapshots", "other"))
                continue
            holds = list(m.snapshot_holds.get(d.name, []))
            for a in users.get(d.name, []):
                for h in m.attempt_holds.get(a, []):
                    holds.append(Hold(h.kind, h.source, f"code of held {a}: {h.reason}", h.release))
            checked = [f"used by {a} {m.attempt_state(a)}" for a in users.get(d.name, [])]
            cls = "snapshot"
            if not holds and not users.get(d.name):
                age = self.now - st.st_mtime
                if age < RECENT_SECONDS:
                    cls = "recent-snapshot"
                checked.append(f"no attempt uses it; staged {age / 3600:.1f} h ago")
            if cls == "snapshot" and not holds and present.get(d.name) is not True:
                cls = "snapshot-no-commit"
                checked.append("commit missing from the repository" if present.get(d.name) is False
                               else "repository not readable")
            entries.append(self._dir_entry(d, "snapshots", cls, holds, checked))
        return entries

    def env_entries(self) -> list:
        entries = []
        for d in self.model._children("envs"):
            st = _lstat(d)
            if st is None:
                continue
            if ENV_DIR.fullmatch(d.name) and stat.S_ISDIR(st.st_mode):
                entries.append(self._dir_entry(d, "envs", "environment", self.env_holds.get(d.name, []),
                                               [f"uv.lock id {d.name[:12]}"]))
            else:
                entries.append(self._kept(d, "envs", "other"))
        return entries

    def kept_area_entries(self) -> list:
        """research/, data/, .miyabi-mcp/, the worktrees and .venv: one keep entry per top-level directory."""
        entries = []
        for area, cls in (("research", "records"), ("data", "data"), (".miyabi-mcp", "ledger")):
            for d in self.model._children(area):
                entries.append(self._kept(d, area, cls))
        if self.worktrees and self.worktrees.is_dir():
            for d in sorted(self.worktrees.iterdir()):
                entries.append(self._kept(d, "worktrees", "worktree"))
        venv = self.root / ".venv"
        if venv.is_dir():
            entries.append(self._kept(venv, "tooling", "tooling"))
        return entries

    def _kept(self, d: Path, area: str, cls: str):
        """A keep entry for a whole file or tree (its size only; nothing in it is ever a candidate)."""
        st = _lstat(d)
        if st is None:
            return None
        if stat.S_ISDIR(st.st_mode):
            total, files, newest = 0, 0, 0
            for dirpath, _, filenames in os.walk(d):
                for name in filenames:
                    fst = _lstat(Path(dirpath) / name)
                    if fst is None:
                        continue
                    total += 0 if stat.S_ISLNK(fst.st_mode) else fst.st_size
                    files += 1
                    newest = max(newest, fst.st_mtime_ns)
            entry = self._entry(d, area, "dir", cls, total, files, fingerprint={"listing": f"{total}:{files}:{newest}"})
        else:
            entry = self._entry(d, area, "file", cls, st.st_size, 1, fingerprint={"stat": stat_fingerprint(st)})
        self._decide(entry, [], [])
        return entry

    def plan(self, scope_spec: str) -> Plan:
        scope = self.scope(scope_spec)
        plan = Plan(scope)
        attempts = set(scope["attempts"])
        for d in self.model._children("runs"):
            if scope["id"] == "all" or d.name in attempts:
                st = _lstat(d)
                if st is not None and stat.S_ISDIR(st.st_mode):
                    plan.entries.extend(self.attempt_entries(d))
                else:
                    plan.entries.append(self._kept(d, "runs", "other"))
        plan.entries.extend(self.snapshot_entries(None if scope["id"] == "all" else attempts))
        if scope["id"] == "all":
            plan.entries.extend(self.env_entries())
            plan.entries.extend(self.kept_area_entries())
        plan.entries = [e for e in plan.entries if e is not None]  # None: vanished while it was planned
        if scope["gate"]["open"]:
            hold = Hold("scope-open", scope["id"], "scope has open work: " + "; ".join(scope["gate"]["why"]),
                        "every task of the scope accepted and every attempt ended")
            for entry in plan.entries:
                if entry["decision"] == CANDIDATE:
                    entry.update(decision=HOLD, holds=entry["holds"] + [hold.as_dict()])
        if self.hash_candidates:
            for entry in plan.entries:
                if entry["decision"] == CANDIDATE:
                    self._add_content_hash(entry)
        plan.entries.sort(key=lambda e: e["path"])
        return plan


# --------------------------------------------------------------------------------------------- outputs


def _gb(n: int) -> str:
    return f"{n / 1e9:.3f} GB"


def totals(entries: list) -> dict:
    out = {d: {"bytes": 0, "entries": 0} for d in (KEEP, HOLD, CANDIDATE)}
    for e in entries:
        out[e["decision"]]["bytes"] += e["bytes"]
        out[e["decision"]]["entries"] += 1
    return out


def summary_text(doc: dict, plan_path: Path, digest: str) -> str:
    entries, scope = doc["entries"], doc["scope"]
    t = doc["totals"]
    lines = [f"# Artifact cleanup plan: {scope['id']}", "",
             f"- plan: `{plan_path}`", f"- plan sha256: `{digest}`",
             f"- records: `{doc['records']['path']}` (data_version {doc['records'].get('data_version')}, "
             f"sha256 {doc['records']['sha256'][:16]})",
             f"- created: {doc['created_at']} on {doc['host']}; root `{doc['root']}`",
             f"- totals: candidate {_gb(t[CANDIDATE]['bytes'])} ({t[CANDIDATE]['entries']}), "
             f"hold {_gb(t[HOLD]['bytes'])} ({t[HOLD]['entries']}), keep {_gb(t[KEEP]['bytes'])} "
             f"({t[KEEP]['entries']})", ""]
    if scope["id"] != "all":
        gate = scope["gate"]
        lines += [f"## Scope {scope['id']}", "",
                  f"- hypotheses: {', '.join(scope['hypotheses']) or '-'}; experiments: "
                  f"{', '.join(scope['experiments']) or '-'}; attempts: {len(scope['attempts'])}",
                  "- tasks: " + (", ".join(f"{x['id']} {x['state']}" for x in scope["tasks"]) or "none"),
                  ("- gate: REFUSED, no candidates while work is open: " + "; ".join(gate["why"])) if gate["open"]
                  else "- gate: closed (every task accepted, every attempt ended)"]
        lines += [f"- warning: {w}" for w in scope.get("warnings", [])]
        lines.append("")
    lines += ["## By area and class", "", "| area | class | candidate | hold | keep |", "|---|---|---|---|---|"]
    table = defaultdict(lambda: {KEEP: [0, 0], HOLD: [0, 0], CANDIDATE: [0, 0]})
    for e in entries:
        cell = table[(e["area"], e["class"])][e["decision"]]
        cell[0] += e["bytes"]
        cell[1] += 1
    for (area, cls), cells in sorted(table.items()):
        lines.append(f"| {area} | {cls} | " + " | ".join(
            f"{_gb(cells[d][0])} ({cells[d][1]})" if cells[d][1] else "-" for d in (CANDIDATE, HOLD, KEEP)) + " |")
    releases = defaultdict(lambda: [0, 0, set()])
    for e in entries:
        if e["decision"] == HOLD:
            for release in sorted({h["release"] for h in e["holds"]}):
                releases[release][0] += e["bytes"]
                releases[release][1] += 1
                releases[release][2].update(h["source"] for h in e["holds"] if h["release"] == release)
    if releases:
        lines += ["", "## Holds by release event (an entry may wait for several)", ""]
        for release, (nbytes, n, sources) in sorted(releases.items(), key=lambda kv: -kv[1][0]):
            shown = sorted(sources)
            more = f" +{len(shown) - 8}" if len(shown) > 8 else ""
            lines.append(f"- {_gb(nbytes)} ({n}): {release} [held by {', '.join(shown[:8])}{more}]")
    candidates = sorted((e for e in entries if e["decision"] == CANDIDATE), key=lambda e: -e["bytes"])
    cited = [e for e in candidates if any(c.startswith("cited by ") for c in e["checked"])]
    if cited:
        lines += ["", "## Candidates that a result cites (rl doctor skips a deleted file; the result keeps its digest)",
                  "", f"{len(cited)} entries, {_gb(sum(e['bytes'] for e in cited))}:", ""]
        lines += [f"- {_gb(e['bytes'])} {e['class']} `{e['path']}` "
                  f"({', '.join(c for c in e['checked'] if c.startswith('cited by '))})" for e in cited[:15]]
        if len(cited) > 15:
            lines.append(f"- ... {len(cited) - 15} more in plan.json")
    if candidates:
        lines += ["", "## Largest candidates", ""]
        for e in candidates[:25]:
            lines.append(f"- {_gb(e['bytes'])} {e['class']} `{e['path']}` ({'; '.join(e['checked'][:4])})")
    large = sorted((e for e in entries if e["decision"] == KEEP and e["bytes"] >= 10 << 20 and e["area"] == "runs"),
                   key=lambda e: -e["bytes"])
    if large:
        lines += ["", "## Large files kept by rule (10 MiB or more, runs/)", ""]
        lines += [f"- {_gb(e['bytes'])} {e['class']} `{e['path']}`" for e in large[:20]]
    lines += ["", "## Apply (coordinator, in a PBS job, after review)", "",
              "```", f"python src/artifact_cleanup.py apply --plan {plan_path} \\",
              "    --records <fresh rl export>/snapshot.json --confirm " + digest, "```", ""]
    return "\n".join(lines)


def write_plan(planner: Planner, scope: str, records: Records, out_dir: Path) -> tuple[Path, str]:
    out_dir = Path(os.path.abspath(out_dir))
    if (out_dir / "plan.json").exists():
        raise CleanupError(f"{out_dir}/plan.json exists: a plan is never overwritten; choose a new --out")
    plan = planner.plan(scope)
    doc = {"format": PLAN_FORMAT, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "host": socket.gethostname(), "pbs_job_id": os.environ.get("PBS_JOBID"),
           "tool": str(Path(__file__).resolve()), "root": str(planner.root),
           "worktrees": str(planner.worktrees) if planner.worktrees else None,
           "params": {"large_bytes": planner.large_bytes, "recent_seconds": RECENT_SECONDS,
                      "closed_task_states": sorted(CLOSED_TASK_STATES)},
           "records": {"path": records.path, "sha256": records.sha256,
                       "data_version": records.meta.get("data_version"),
                       "updated_at": records.meta.get("updated_at")},
           "scope": plan.scope, "checked_against": {"live_sources": planner.model.sources,
                                                    "locks": sorted(planner.model.locks)},
           "rules": {RULE_IDS[k]: v for k, v in {**CLASSES, **KEEP_RULES}.items()},
           "totals": totals(plan.entries), "entries": plan.entries}
    out_dir.mkdir(parents=True, exist_ok=True)
    data = json.dumps(doc, ensure_ascii=False, indent=1).encode("utf-8")
    plan_path = out_dir / "plan.json"
    with open(plan_path, "xb") as sink:
        sink.write(data)
    digest = hashlib.sha256(data).hexdigest()
    (out_dir / "summary.md").write_text(summary_text(doc, plan_path, digest), encoding="utf-8")
    return plan_path, digest


# --------------------------------------------------------------------------------------------- apply


def _deletable_location(path: Path, root: Path) -> bool:
    """Only paths below runs/, snapshots/ or envs/ may ever be deleted, also after resolving the links of the
    directories above them."""
    if not path.is_absolute() or ".." in path.parts:
        return False
    for base in (root / "runs", root / "snapshots", root / "envs"):
        if base in path.parents:
            real_base, real_parent = Path(os.path.realpath(base)), Path(os.path.realpath(path.parent))
            return real_parent == real_base or real_base in real_parent.parents
    return False


def _remove_tree(path: Path):
    for dirpath, _, _ in os.walk(path):  # read-only snapshot trees: make each directory writable first
        mode = os.lstat(dirpath).st_mode
        os.chmod(dirpath, stat.S_IMODE(mode) | stat.S_IRWXU)
    shutil.rmtree(path)


def apply_plan(plan_path: Path, records_path: Path, confirm: str, log_path: Path | None = None,
               git: Git | None = None) -> dict:
    data = Path(plan_path).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if confirm != digest:
        raise CleanupError(f"--confirm does not match the plan: {plan_path} has sha256 {digest}; review the plan "
                           "and pass that value")
    doc = json.loads(data)
    if doc.get("format") != PLAN_FORMAT:
        raise CleanupError(f"{plan_path} is not a {PLAN_FORMAT} plan")
    root = Path(doc["root"])
    worktrees = Path(doc["worktrees"]) if doc.get("worktrees") else None
    log_path = Path(log_path) if log_path else Path(plan_path).with_name("deletions.jsonl")
    planned = [e for e in doc["entries"] if e["decision"] == CANDIDATE]
    records = load_records(records_path)

    def log(event: dict):
        with open(log_path, "a", encoding="utf-8") as sink:
            sink.write(json.dumps({"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **event},
                                  ensure_ascii=False) + "\n")

    log({"event": "apply-start", "plan": str(plan_path), "plan_sha256": digest, "records": str(records_path),
         "records_sha256": records.sha256, "data_version": records.meta.get("data_version"),
         "host": socket.gethostname(), "pbs_job_id": os.environ.get("PBS_JOBID"), "candidates": len(planned)})
    problems = []
    old_version, new_version = doc["records"].get("data_version"), records.meta.get("data_version")
    if isinstance(old_version, int) and isinstance(new_version, int) and new_version < old_version:
        problems.append(f"records data_version {new_version} is older than the plan's {old_version}")
    if doc["scope"]["gate"]["open"]:
        problems.append("the plan's scope was open when it was made")
    model = Model(records, root, git, worktrees)
    planner = Planner(model, large_bytes=doc["params"]["large_bytes"])
    fresh = {e["path"]: e for e in planner.plan(doc["scope"]["id"]).entries}
    for e in planned:
        path = Path(e["path"])
        again = fresh.get(e["path"])
        if not _deletable_location(path, root):
            problems.append(f"{path}: outside the deletable areas")
        elif not os.path.lexists(path):
            problems.append(f"{path}: missing")
        elif again is None:
            problems.append(f"{path}: not in a fresh plan of scope {doc['scope']['id']}")
        elif again["decision"] != CANDIDATE:
            why = "; ".join(h["reason"] for h in again.get("holds", [])) or again.get("reason", "")
            problems.append(f"{path}: now {again['decision']} ({why})")
        elif (again["kind"], again["class"], again["bytes"]) != (e["kind"], e["class"], e["bytes"]):
            problems.append(f"{path}: changed ({e['kind']} {e['class']} {e['bytes']} B -> "
                            f"{again['kind']} {again['class']} {again['bytes']} B)")
        elif again["fingerprint"] != e["fingerprint"]:
            problems.append(f"{path}: content or metadata changed since the plan")
    if problems:
        log({"event": "refused", "problems": problems})
        raise CleanupError("apply refused, nothing deleted:\n  " + "\n  ".join(problems))
    deleted, freed = 0, 0
    for e in planned:
        path = Path(e["path"])
        st = _lstat(path)
        if st is None or (e["kind"] == "file" and stat_fingerprint(st) != e["fingerprint"]["stat"]):
            log({"event": "stopped", "path": str(path), "problem": "changed or vanished during apply"})
            raise CleanupError(f"{path} changed or vanished during apply; stopped after {deleted} deletions "
                               f"(see {log_path})")
        try:
            if e["kind"] == "dir":
                _remove_tree(path)
            else:
                os.unlink(path)
        except OSError as exc:
            log({"event": "stopped", "path": str(path), "problem": str(exc)})
            raise CleanupError(f"deleting {path} failed ({exc}); stopped after {deleted} deletions "
                               f"(see {log_path})") from None
        deleted += 1
        freed += e["bytes"]
        log({"event": "deleted", "path": str(path), "kind": e["kind"], "class": e["class"], "bytes": e["bytes"],
             "files": e["files"], "fingerprint": e["fingerprint"], "rule": e["rule"], "attempt": e.get("attempt"),
             "experiment": e.get("experiment"), "checked": e["checked"]})
    log({"event": "apply-end", "deleted": deleted, "bytes": freed})
    return {"deleted": deleted, "bytes": freed, "log": str(log_path)}


# --------------------------------------------------------------------------------------------- CLI


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command")
    p = sub.add_parser("plan", help="dry run: classify and write plan.json and summary.md (default)")
    p.add_argument("--root", required=True, help="the research workspace, e.g. /work/<group>/<user>/<project>")
    p.add_argument("--records", required=True, help="snapshot.json of `rl export --out <dir>`")
    p.add_argument("--out", required=True, help="new output directory for plan.json and summary.md")
    p.add_argument("--scope", default="all", help="all (default), or H-x / E-x ids separated by commas")
    p.add_argument("--worktrees", help="directory of the researcher worktrees, where locks on task branches are "
                                       "looked up (default: <root>/../<root name>-worktrees when it exists)")
    p.add_argument("--large-bytes", type=int, default=LARGE_BYTES,
                   help=f"training traces and logs from this size on are deletable (default {LARGE_BYTES})")
    a = sub.add_parser("apply", help="delete the candidates of a reviewed plan")
    a.add_argument("--plan", required=True)
    a.add_argument("--records", required=True, help="a fresh `rl export` snapshot.json, made after the review")
    a.add_argument("--confirm", required=True, help="the sha256 of plan.json printed in summary.md")
    a.add_argument("--log", help="deletion log (default: deletions.jsonl next to the plan)")
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] not in (["plan"], ["apply"], ["-h"], ["--help"]):
        argv = ["plan", *argv]  # the dry run is the default
    args = parser.parse_args(argv)
    try:
        if args.command == "apply":
            done = apply_plan(Path(args.plan), Path(args.records), args.confirm, args.log)
            print(f"deleted {done['deleted']} entries, {_gb(done['bytes'])}; log {done['log']}")
            return 0
        records = load_records(args.records)
        root = Path(os.path.abspath(args.root))
        if not (root / "runs").is_dir():
            raise CleanupError(f"--root {root}: no runs/ directory; is this the research workspace?")
        worktrees = Path(args.worktrees) if args.worktrees else default_worktrees(root)
        planner = Planner(Model(records, root, worktrees=worktrees), large_bytes=args.large_bytes)
        plan_path, digest = write_plan(planner, args.scope, records, Path(args.out))
        print(f"plan {plan_path}\nsha256 {digest}\nsummary {plan_path.with_name('summary.md')}")
        return 0
    except (CleanupError, OSError) as exc:
        print(f"artifact_cleanup: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
