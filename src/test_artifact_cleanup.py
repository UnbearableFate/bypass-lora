"""Tests of the artifact cleanup tool on synthetic projects in temporary directories (never on real files)."""
import hashlib
import json
import os
import shutil
import stat
import subprocess
import time
from pathlib import Path

import pytest

import artifact_cleanup as ac

LARGE = 1000  # --large-bytes of these tests: traces and logs from 1000 bytes on are deletable
SHA_USED = hashlib.sha1(b"used by ATT-0001-1 only").hexdigest()
SHA_HELD = hashlib.sha1(b"used by the queued ATT-0004-1").hexdigest()
SHA_RECENT = hashlib.sha1(b"staged just now, no attempt yet").hexdigest()
SHA_LOST = hashlib.sha1(b"commit not in the repository").hexdigest()
SHA_IDLE = hashlib.sha1(b"old and unused").hexdigest()
HEAD_LOCK, OLD_LOCK = "3c916b51337d", "c073f466bd0e"
OLD = time.time() - 3 * 86400

# attempt: (experiment, execution state, run spec additions, snapshot)
ATTEMPTS = {
    "ATT-0001-1": ("E-0001", "ended", {}, SHA_USED),
    "ATT-0002-1": ("E-0001", "ended", {}, None),
    "ATT-0003-1": ("E-0002", "ended", {}, None),
    "ATT-0004-1": ("E-0002", "queued", {"environment_ref": f"uv.lock {HEAD_LOCK};cuda"}, SHA_HELD),
    "ATT-0006-1": ("E-0003", "ended", {"resume_from": "ATT-0010-1"}, None),
    "ATT-0007-1": ("E-0003", "ended", {}, None),
    "ATT-0008-1": ("E-0004", "ended", {}, None),
    "ATT-0009-1": ("E-0001", "ended", {}, None),
    "ATT-0010-1": ("E-0003", "ended", {"resume_from": "runs/ATT-0011-1/adapter"}, None),
    "ATT-0011-1": ("E-0003", "ended", {}, None),
    "ATT-0012-1": ("E-0005", "ended", {}, None),
}
EXPERIMENTS = {  # experiment: (hypothesis it tests, tasks)
    "E-0001": ("H-0002", ["T-0001"]),
    "E-0002": ("H-0001", ["T-0002"]),
    "E-0003": ("H-0002", ["T-0003"]),
    "E-0004": ("H-0002", ["T-0004"]),
    "E-0005": ("H-0001", ["T-0005"]),
    "E-0006": ("H-0001", []),
}
H2_CANDIDATES = ["runs/ATT-0001-1/adapter/adapter.pt", "runs/ATT-0001-1/optimizer.pt",
                 "runs/ATT-0001-1/regression/old.pt", "runs/ATT-0001-1/step-100/adapter.pt",
                 "runs/ATT-0001-1/steps.jsonl", "runs/ATT-0001-1/tmp/test_reload0", "runs/ATT-0001-1/train.log",
                 f"snapshots/{SHA_USED}"]


class FakeGit:
    def __init__(self, present=(SHA_USED, SHA_HELD, SHA_RECENT, SHA_IDLE), head_lock=HEAD_LOCK):
        self.present, self.head_lock = set(present), head_lock

    def commits_present(self, shas):
        return {s: s in self.present for s in shas}

    def head_lock_id(self, at=None):
        return self.head_lock if at is None else None


def write(path: Path, data="x", size=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if size is not None:
        data = (data * size)[:size]
    path.write_text(data)
    return path


def export(path: Path, root: Path, data_version=1, tasks=None):
    """An `rl export` snapshot.json: nodes per label (nested values as JSON text), edges with upper-case names."""
    tasks = tasks or {"T-0001": "accepted", "T-0002": "active", "T-0003": "accepted", "T-0004": "accepted",
                      "T-0005": "accepted"}
    nodes = {
        "Hypothesis": [{"id": "H-0001", "disposition": "active", "evidence_status": "untested"},
                       {"id": "H-0002", "disposition": "paused", "evidence_status": "challenged"}],
        "Experiment": [{"id": e, "contract_path": f"research/experiments/{e}.md"} for e in EXPERIMENTS],
        "Task": [{"id": t, "state": s, "title": f"task {t}",
                  "next_action": "read research/notes/T-0002-wait.md first" if t == "T-0002" else None}
                 for t, s in tasks.items()],
        "Decision": [{"id": "D-0001", "reopen_condition": "如果需要在 D1 checkpoint ATT-0006-1 上重新检验"}],
        "Run": [{"id": "RUN-" + a[4:8], "experiment_id": e,
                 "spec": {"config_ref": "src/train.py", "environment_ref": f"uv.lock {HEAD_LOCK}", **extra}}
                for a, (e, _, extra, _) in ATTEMPTS.items()],
        "Attempt": [{"id": a, "run_id": "RUN-" + a[4:8], "execution_state": state,
                     "outcome": "success" if state == "ended" else "unknown",
                     "result_status": "complete" if state == "ended" else "absent",
                     "resource": {"snapshot": str(root / "snapshots" / sha)} if sha else {"kind": "remote"}}
                    for a, (_, state, _, sha) in ATTEMPTS.items()],
        "Result": [{"id": "RES-0001", "attempt_id": "ATT-0001-1", "manifest": {"artifacts": [
            {"path": str(root / "runs/ATT-0001-1/adapter/adapter.pt"), "role": "checkpoint"}]}, "verification": []}],
    }
    for rows in nodes.values():
        for row in rows:
            for k, v in row.items():
                if isinstance(v, (dict, list)):
                    row[k] = json.dumps(v)
    edges = [(t, "TARGETS", e) for e, (_, ts) in EXPERIMENTS.items() for t in ts if t in tasks]
    edges += [(t, "TARGETS", "E-0002") for t in tasks if t not in {"T-0001", "T-0002", "T-0003", "T-0004", "T-0005"}]
    edges += [(e, "TESTS", h) for e, (h, _) in EXPERIMENTS.items()]
    edges += [("E-0002", "CODE_FROM", "E-0004"), ("D-0001", "DECIDES", "H-0002")]
    edges += [("RUN-" + a[4:8], "OF_EXPERIMENT", e) for a, (e, *_) in ATTEMPTS.items()]
    edges += [(a, "OF_RUN", "RUN-" + a[4:8]) for a in ATTEMPTS]
    doc = {"format": "smooth-research-snapshot/1", "meta": {"data_version": data_version}, "nodes": nodes,
           "edges": [{"source": s, "source_label": "", "edge": r, "target": t, "target_label": "", "props": {}}
                     for s, r, t in edges]}
    path.write_text(json.dumps(doc, ensure_ascii=False))
    return path


class Project:
    def __init__(self, tmp: Path):
        self.tmp = tmp
        self.root = tmp / "proj"
        self.worktrees = tmp / "proj-worktrees"
        self.git = FakeGit()
        r = self.root
        for a in ATTEMPTS:
            write(r / "runs" / a / "adapter" / "adapter.pt", size=2048)
            write(r / "runs" / a / "manifest.json", '{"status": "ok"}')
        att1 = r / "runs" / "ATT-0001-1"
        write(att1 / "predictions.jsonl", '{"id": 1}\n')
        write(att1 / "steps.jsonl", '{"step": 1, "loss": 2.0}\n', size=1500)
        write(att1 / "train.log", "loss ", size=1200)
        write(att1 / "eval.log", "ok\n")
        write(att1 / "optimizer.pt", size=600)
        write(att1 / "step-100" / "adapter.pt", size=700)
        write(att1 / "regression" / "old.pt", size=900)
        write(att1 / "data" / "train.jsonl", '{"prompt": "q"}\n')
        write(att1 / "tmp" / "test_reload0" / "adapter" / "adapter.pt", size=300)
        os.symlink("manifest.json", att1 / "manifest-link.json")
        write(r / "runs" / "ATT-0099-1" / "adapter.pt", size=100)  # not in the records
        for sha in (SHA_USED, SHA_HELD, SHA_RECENT, SHA_LOST, SHA_IDLE):
            write(r / "snapshots" / sha / "src" / "train.py", "print('x')\n")
            if sha != SHA_RECENT:
                os.utime(r / "snapshots" / sha, (OLD, OLD))
        os.chmod(r / "snapshots" / SHA_USED / "src", 0o555)  # snapshots are read-only
        os.chmod(r / "snapshots" / SHA_USED, 0o555)
        for lock in (HEAD_LOCK, OLD_LOCK):
            write(r / "envs" / f"{lock}-cuda" / "bin" / "python", size=50)
        # uv links an environment's files from its cache: deleting the environment frees only its own data
        cached = write(tmp / "uv-cache" / "torch.so", size=5000)
        os.link(cached, r / "envs" / f"{OLD_LOCK}-cuda" / "bin" / "torch.so")
        own = write(r / "envs" / f"{OLD_LOCK}-cuda" / "lib" / "a.so", size=300)
        os.link(own, r / "envs" / f"{OLD_LOCK}-cuda" / "lib" / "b.so")  # both names inside the environment
        write(r / "research" / "tasks" / "T-0002-brief.md", "# T-0002\nUse the E-0002 data.\n")
        write(r / "research" / "notes" / "T-0002-wait.md", "Compare against the adapter of ATT-0002-1.\n")
        write(r / "research" / "experiments" / "E-0005.md", "start checkpoint: runs/ATT-0009-1/adapter/adapter.pt\n")
        write(r / "research" / "experiments" / "E-0002.md", "E-0002 contract without attempt ids\n")
        write(r / "research" / "artifacts" / "T-0001" / "lr-lock.json", '{"source": "ATT-0012-1"}')
        write(r / "data" / "v5-sealed" / "valid.jsonl", '{"id": 1}\n')
        write(r / ".miyabi-mcp" / "ledger.jsonl", '{"job": 1}\n')
        checkout = self.worktrees / "researcher-1"  # the lock lives only on the task branch
        write(checkout / ".git", "gitdir: elsewhere\n")
        write(checkout / "research" / "artifacts" / "T-0002" / "confirmation-lock.json",
              json.dumps({"checkpoints": ["runs/ATT-0007-1/adapter/adapter.pt"],
                          "learning_rate_lock": "research/artifacts/T-0001/lr-lock.json"}))
        write(checkout / "research" / "artifacts" / "T-0001" / "lr-lock.json", '{"source": "ATT-0012-1"}')
        self.records = export(tmp / "snapshot.json", r)

    def model(self, records=None, git=None):
        return ac.Model(ac.load_records(records or self.records), self.root, git or self.git, self.worktrees)

    def plan(self, scope="all", records=None, git=None):
        return ac.Planner(self.model(records, git), large_bytes=LARGE).plan(scope)

    def write_plan(self, scope="H-0002", out="plan"):
        records = ac.load_records(self.records)
        planner = ac.Planner(ac.Model(records, self.root, self.git, self.worktrees), large_bytes=LARGE)
        return ac.write_plan(planner, scope, records, self.tmp / out)


@pytest.fixture
def project(tmp_path):
    return Project(tmp_path)


def by_path(plan, root):
    return {os.path.relpath(e["path"], root): e for e in plan.entries}


def kinds(entry):
    return {h["kind"] for h in entry["holds"]}


def tree_state(*tops):
    state = {}
    for top in tops:
        for dirpath, dirnames, filenames in os.walk(top):
            for name in dirnames + filenames:
                p = Path(dirpath) / name
                st = os.lstat(p)
                item = (stat.S_IFMT(st.st_mode), stat.S_IMODE(st.st_mode), st.st_size, st.st_mtime_ns)
                if stat.S_ISREG(st.st_mode):
                    item += (hashlib.sha256(p.read_bytes()).hexdigest(),)
                elif stat.S_ISLNK(st.st_mode):
                    item += (os.readlink(p),)
                state[str(p)] = item
    return state


# ------------------------------------------------------------------------------------------- classification


@pytest.mark.parametrize("rel, line, repeat, expected", [
    ("adapter/adapter.pt", "x", 10, "checkpoint"),
    ("model.safetensors", "x", 10, "checkpoint"),
    ("checkpoint_final.pt", "x", 10, "checkpoint"),
    ("step-100/adapter.pt", "x", 10, "intermediate-checkpoint"),
    ("checkpoints/epoch_2/weights.bin", "x", 10, "intermediate-checkpoint"),
    ("optimizer.pt", "x", 10, "optimizer"),
    ("scheduler_state.pth", "x", 10, "optimizer"),
    ("regression/new.pt", "x", 10, "regression-dump"),
    ("features.train.pt", "x", 10, "feature-cache"),
    ("e16/calibration.normal.predictions.jsonl", '{"id": 1}\n', 200, "predictions"),
    ("steps.jsonl", '{"step": 1}\n', 200, "trace"),
    ("steps.jsonl", '{"step": 1}\n', 1, "metadata"),
    ("data/train.jsonl", '{"prompt": "q"}\n', 200, "data-copy"),
    ("e8/calibration.jsonl", '{"prompt": "q"}\n', 200, "data-copy"),
    ("diagnostics.jsonl", '{"episode_id": 1}\n', 200, "metadata"),
    ("train.log", "x", 2000, "trace"),
    ("pytest.log", "ok", 1, "metadata"),
    ("probe.npy", "x", 10, "unclassified"),
    ("manifest.json", "{}", 1, "metadata"),
])
def test_classify_file(tmp_path, rel, line, repeat, expected):
    path = write(tmp_path / rel, line * repeat)
    assert ac.classify_file(Path(rel).parts, path, path.stat().st_size, LARGE) == expected


def test_ids_in_text_expands_ranges_and_bare_run_ids():
    attempts, runs = ac._ids_in_text("lock pins ATT-0129-1..ATT-0137-1; also RUN-0005, ATT0179 and runs/ATT-0042-2/x")
    assert attempts == {"ATT-0129-1", "ATT-0137-1", "ATT-0042-2"}
    assert {f"{n:04d}" for n in range(129, 138)} <= runs and {"0005", "0179"} <= runs
    _, runs = ac._ids_in_text("ATT-0001-1 to 2000 steps")  # not a range of attempts
    assert runs == set()


# ------------------------------------------------------------------------------------------- holds


def test_whole_project_plan_holds_what_open_work_needs(project):
    e = by_path(project.plan(), project.root)
    adapter = lambda a: e[f"runs/{a}/adapter/adapter.pt"]  # noqa: E731
    assert adapter("ATT-0001-1")["decision"] == "candidate"
    assert adapter("ATT-0001-1")["rule"] == "C1-checkpoint"
    assert "cited by RES-0001 (checkpoint)" in adapter("ATT-0001-1")["checked"]
    assert kinds(adapter("ATT-0004-1")) >= {"unfinished-attempt", "open-experiment"}
    assert kinds(adapter("ATT-0003-1")) == {"open-experiment"}
    assert kinds(adapter("ATT-0008-1")) == {"relation"}  # open E-0002 takes its code from E-0004
    assert kinds(adapter("ATT-0002-1")) == {"open-task"}  # named in a note that the open task names
    assert {h["release"] for h in adapter("ATT-0002-1")["holds"]} == {"T-0002 accepted"}
    assert kinds(adapter("ATT-0007-1")) == {"lock"}  # the lock exists only in the task's worktree
    assert kinds(adapter("ATT-0012-1")) == {"lock"}  # a lock named by that lock
    # the same lock in the main checkout and a worktree is read once, from the main checkout
    assert {h["source"] for h in adapter("ATT-0012-1")["holds"]} == {"research/artifacts/T-0001/lr-lock.json"}
    assert kinds(adapter("ATT-0006-1")) == {"reopen-condition"}
    for a in ("ATT-0010-1", "ATT-0011-1"):  # resume_from chain of the held ATT-0006-1
        assert kinds(adapter(a)) == {"start-checkpoint"}
        assert {h["release"] for h in adapter(a)["holds"]} == {"the reopen condition of D-0001 no longer applies"}
    assert kinds(adapter("ATT-0009-1")) == {"contract"}  # named in the contract of E-0005, which tests live H-0001
    assert {h["release"] for h in adapter("ATT-0009-1")["holds"]} == {"H-0001 paused or stopped"}


def test_whole_project_plan_classifies_every_area(project):
    e = by_path(project.plan(), project.root)
    decision = {k: (v["decision"], v["class"]) for k, v in e.items()}
    assert decision["runs/ATT-0001-1/manifest.json"] == ("keep", "metadata")
    assert decision["runs/ATT-0001-1/predictions.jsonl"] == ("keep", "predictions")
    assert decision["runs/ATT-0001-1/eval.log"] == ("keep", "metadata")
    assert decision["runs/ATT-0001-1/data/train.jsonl"] == ("keep", "data-copy")
    assert decision["runs/ATT-0001-1/manifest-link.json"] == ("keep", "link")
    assert decision["runs/ATT-0001-1/steps.jsonl"] == ("candidate", "trace")
    assert decision["runs/ATT-0001-1/tmp/test_reload0"] == ("candidate", "test-temp")
    assert e["runs/ATT-0001-1/tmp/test_reload0"]["kind"] == "dir"
    assert "runs/ATT-0001-1/tmp/test_reload0/adapter/adapter.pt" not in e
    assert decision["runs/ATT-0099-1"] == ("keep", "unregistered")
    assert decision[f"snapshots/{SHA_USED}"] == ("candidate", "snapshot")
    assert decision[f"snapshots/{SHA_HELD}"] == ("hold", "snapshot")
    assert decision[f"snapshots/{SHA_RECENT}"] == ("keep", "recent-snapshot")
    assert decision[f"snapshots/{SHA_LOST}"] == ("keep", "snapshot-no-commit")
    assert decision[f"snapshots/{SHA_IDLE}"] == ("candidate", "snapshot")
    assert decision[f"envs/{HEAD_LOCK}-cuda"] == ("hold", "environment")
    assert decision[f"envs/{OLD_LOCK}-cuda"] == ("candidate", "environment")
    old_env = e[f"envs/{OLD_LOCK}-cuda"]
    assert (old_env["bytes"], old_env["freed"]) == (50 + 5000 + 300 + 300, 50 + 300)  # the cache keeps torch.so
    assert decision["research/tasks"] == ("keep", "records")
    assert decision["data/v5-sealed"] == ("keep", "data")
    assert decision[".miyabi-mcp/ledger.jsonl"] == ("keep", "ledger")
    assert decision["../proj-worktrees/researcher-1"] == ("keep", "worktree")
    assert all(x["fingerprint"].get("sha256") for x in e.values() if x["decision"] == "candidate")


def test_unknown_commit_presence_never_makes_a_candidate(project):
    git = FakeGit()
    git.commits_present = lambda shas: {s: None for s in shas}  # the repository could not be read
    e = by_path(project.plan(git=git), project.root)
    assert e[f"snapshots/{SHA_IDLE}"]["class"] == "snapshot-no-commit"
    assert e[f"snapshots/{SHA_IDLE}"]["decision"] == "keep"


# ------------------------------------------------------------------------------------------- scope gate


def test_scope_of_closed_hypothesis_has_candidates_only_from_its_experiments(project):
    plan = project.plan("H-0002")
    scope = plan.scope
    assert scope["experiments"] == ["E-0001", "E-0003", "E-0004"]
    assert [(t["id"], t["state"]) for t in scope["tasks"]] == [("T-0001", "accepted"), ("T-0003", "accepted"),
                                                               ("T-0004", "accepted")]
    assert scope["gate"] == {"open": False, "why": []}
    candidates = sorted(os.path.relpath(x["path"], project.root) for x in plan.entries if x["decision"] == "candidate")
    assert candidates == H2_CANDIDATES
    held = {os.path.relpath(x["path"], project.root) for x in plan.entries if x["decision"] == "hold"}
    assert "runs/ATT-0007-1/adapter/adapter.pt" in held and "runs/ATT-0009-1/adapter/adapter.pt" in held


@pytest.mark.parametrize("scope, why", [
    ("E-0002", ["T-0002 is active", "ATT-0004-1 is queued/unknown/result:absent"]),
    ("E-0006", ["E-0006 has no task"]),
    ("H-0001", ["T-0002 is active", "ATT-0004-1 is queued/unknown/result:absent", "E-0006 has no task"]),
])
def test_scope_with_open_work_is_refused_and_lists_its_tasks(project, scope, why):
    plan = project.plan(scope)
    assert plan.scope["gate"] == {"open": True, "why": why}
    assert not [x for x in plan.entries if x["decision"] == "candidate"]
    if scope == "E-0002":
        assert plan.scope["tasks"] == [{"id": "T-0002", "state": "active", "title": "task T-0002"}]
        assert {x["decision"] for x in plan.entries if x["class"] in ac.DELETABLE} == {"hold"}


def test_a_stopped_task_still_blocks_its_scope(project):
    records = export(project.tmp / "stopped.json", project.root,
                     tasks={"T-0001": "stopped", "T-0002": "active", "T-0003": "accepted", "T-0004": "accepted",
                            "T-0005": "accepted"})
    plan = project.plan("E-0001", records=records)
    assert plan.scope["gate"]["why"] == ["T-0001 is stopped"]
    assert not [x for x in plan.entries if x["decision"] == "candidate"]
    assert by_path(plan, project.root)["runs/ATT-0001-1/adapter/adapter.pt"]["decision"] == "hold"


def test_unknown_scope_is_an_error(project):
    with pytest.raises(ac.CleanupError, match="no such hypothesis or experiment"):
        project.plan("E-0099")


# ------------------------------------------------------------------------------------------- dry run


def test_dry_run_changes_nothing_on_disk(project, monkeypatch, capsys):
    monkeypatch.setattr(ac, "Git", lambda root: project.git)
    before = tree_state(project.root, project.worktrees)
    for i, scope in enumerate(("all", "H-0002", "E-0002")):
        args = ["--root", str(project.root), "--records", str(project.records), "--out", str(project.tmp / f"out{i}"),
                "--scope", scope, "--large-bytes", str(LARGE)]
        assert ac.main(args if i == 0 else ["plan", *args]) == 0  # plan is also the default command
    assert tree_state(project.root, project.worktrees) == before
    summary = (project.tmp / "out2" / "summary.md").read_text()
    assert "REFUSED" in summary and "T-0002 active" in summary
    out = capsys.readouterr().out
    assert "plan.json" in out and "sha256" in out


def test_plan_is_never_overwritten(project):
    project.write_plan(out="same")
    with pytest.raises(ac.CleanupError, match="never overwritten"):
        project.write_plan(out="same")


def test_plan_file_records_scope_rules_and_digest(project):
    plan_path, digest = project.write_plan()
    doc = json.loads(plan_path.read_bytes())
    assert hashlib.sha256(plan_path.read_bytes()).hexdigest() == digest
    assert doc["format"] == ac.PLAN_FORMAT and doc["records"]["data_version"] == 1
    assert doc["scope"]["id"] == "H-0002" and doc["params"]["large_bytes"] == LARGE
    assert set(doc["rules"]) == set(ac.RULE_IDS.values())
    assert any("confirmation-lock.json" in path for path in doc["checked_against"]["locks"])
    assert doc["totals"]["candidate"]["entries"] == len(H2_CANDIDATES)
    assert digest in (plan_path.parent / "summary.md").read_text()


# ------------------------------------------------------------------------------------------- apply


def candidates(plan_path):
    return [e["path"] for e in json.loads(plan_path.read_bytes())["entries"] if e["decision"] == "candidate"]


def events(plan_path):
    log = plan_path.parent / "deletions.jsonl"
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def assert_untouched(paths):
    assert all(os.path.lexists(p) for p in paths)


def test_apply_needs_the_plan_digest(project):
    plan_path, digest = project.write_plan()
    for confirm in ("", "0" * 64, digest[:-1]):
        with pytest.raises(ac.CleanupError, match="--confirm does not match"):
            ac.apply_plan(plan_path, project.records, confirm, git=project.git)
    assert_untouched(candidates(plan_path))


def test_apply_refuses_a_changed_candidate(project):
    plan_path, digest = project.write_plan()
    target = project.root / "runs/ATT-0001-1/adapter/adapter.pt"
    target.write_text("y" * 2048)  # same size, new content
    with pytest.raises(ac.CleanupError, match="changed since the plan"):
        ac.apply_plan(plan_path, project.records, digest, git=project.git)
    assert_untouched(candidates(plan_path))
    assert [x["event"] for x in events(plan_path)] == ["apply-start", "refused"]


def test_apply_refuses_a_missing_candidate(project):
    plan_path, digest = project.write_plan()
    (project.root / "runs/ATT-0001-1/train.log").unlink()
    with pytest.raises(ac.CleanupError, match="train.log: missing"):
        ac.apply_plan(plan_path, project.records, digest, git=project.git)
    assert_untouched(set(candidates(plan_path)) - {str(project.root / "runs/ATT-0001-1/train.log")})


def test_apply_refuses_a_candidate_that_open_work_now_names(project):
    plan_path, digest = project.write_plan()
    tasks = {"T-0001": "accepted", "T-0002": "active", "T-0003": "accepted", "T-0004": "accepted",
             "T-0005": "accepted", "T-0006": "active"}
    fresh = export(project.tmp / "fresh.json", project.root, data_version=2, tasks=tasks)
    write(project.root / "research" / "tasks" / "T-0006-brief.md", "Start from runs/ATT-0001-1/adapter/adapter.pt\n")
    with pytest.raises(ac.CleanupError, match="now hold"):
        ac.apply_plan(plan_path, fresh, digest, git=project.git)
    assert_untouched(candidates(plan_path))


def test_apply_refuses_older_records(project):
    plan_path, digest = project.write_plan()
    old = export(project.tmp / "old.json", project.root, data_version=0)
    with pytest.raises(ac.CleanupError, match="older than the plan"):
        ac.apply_plan(plan_path, old, digest, git=project.git)
    assert_untouched(candidates(plan_path))


def test_apply_refuses_a_plan_of_an_open_scope(project):
    plan_path, digest = project.write_plan(scope="E-0002")
    with pytest.raises(ac.CleanupError, match="scope was open"):
        ac.apply_plan(plan_path, project.records, digest, git=project.git)


@pytest.mark.parametrize("tamper, message", [
    ("research", "outside the deletable areas"),
    ("keep", "now keep"),
    ("dotdot", "outside the deletable areas"),
])
def test_apply_refuses_an_edited_plan(project, tamper, message):
    plan_path, _ = project.write_plan()
    doc = json.loads(plan_path.read_bytes())
    if tamper == "research":
        doc["entries"].append({**doc["entries"][0], "path": str(project.root / "research/tasks/T-0002-brief.md"),
                               "decision": "candidate"})
    elif tamper == "dotdot":
        doc["entries"].append({**doc["entries"][0], "decision": "candidate",
                               "path": str(project.root / "runs/../data/v5-sealed/valid.jsonl")})
    else:
        manifest = next(e for e in doc["entries"] if e["path"].endswith("ATT-0001-1/manifest.json"))
        manifest["decision"] = "candidate"
    data = json.dumps(doc).encode()
    plan_path.chmod(0o644)
    plan_path.write_bytes(data)  # the person who edits the plan also passes its new digest
    with pytest.raises(ac.CleanupError, match=message):
        ac.apply_plan(plan_path, project.records, hashlib.sha256(data).hexdigest(), git=project.git)
    assert_untouched(candidates(plan_path))
    assert (project.root / "research/tasks/T-0002-brief.md").exists()
    assert (project.root / "data/v5-sealed/valid.jsonl").exists()


def test_apply_deletes_exactly_the_candidates_and_logs_them(project):
    plan_path, digest = project.write_plan()
    doc = json.loads(plan_path.read_bytes())
    gone = candidates(plan_path)
    kept = [e["path"] for e in doc["entries"] if e["decision"] != "candidate"]
    before = tree_state(project.root / "research", project.root / "data", project.root / ".miyabi-mcp",
                        project.worktrees)
    done = ac.apply_plan(plan_path, project.records, digest, git=project.git)
    assert done["deleted"] == len(gone) == len(H2_CANDIDATES)
    assert done["bytes"] == done["freed"] == doc["totals"]["candidate"]["bytes"]
    assert not [p for p in gone if os.path.lexists(p)]
    assert_untouched(kept)
    assert not (project.root / "snapshots" / SHA_USED).exists()  # a read-only snapshot tree
    assert (project.root / "runs/ATT-0001-1/manifest.json").exists()
    assert (project.root / "runs/ATT-0002-1/adapter/adapter.pt").exists()
    assert tree_state(project.root / "research", project.root / "data", project.root / ".miyabi-mcp",
                      project.worktrees) == before
    log = events(plan_path)
    assert [x["event"] for x in log] == ["apply-start"] + ["deleted"] * len(gone) + ["apply-end"]
    assert sorted(x["path"] for x in log if x["event"] == "deleted") == sorted(gone)
    assert log[-1]["bytes"] == done["bytes"]


def test_cli_returns_2_on_a_refusal(project, capsys):
    plan_path, _ = project.write_plan()
    assert ac.main(["apply", "--plan", str(plan_path), "--records", str(project.records), "--confirm", "0"]) == 2
    assert "--confirm does not match" in capsys.readouterr().err
    assert_untouched(candidates(plan_path))


# ------------------------------------------------------------------------------------------- boundaries


def test_deletable_location(tmp_path):
    root = tmp_path / "proj"
    (root / "runs" / "ATT-0001-1").mkdir(parents=True)
    (tmp_path / "elsewhere").mkdir()
    os.symlink(tmp_path / "elsewhere", root / "runs" / "ATT-0001-1" / "out")
    assert ac._deletable_location(root / "runs/ATT-0001-1/adapter.pt", root)
    assert ac._deletable_location(root / "snapshots" / SHA_USED, root)
    assert not ac._deletable_location(root / "runs", root)
    assert not ac._deletable_location(root / "research/state", root)
    assert not ac._deletable_location(root / "runs/../research/state", root)
    assert not ac._deletable_location(Path("runs/ATT-0001-1/adapter.pt"), root)
    assert not ac._deletable_location(root / "runs/ATT-0001-1/out/file", root)  # below a link that leaves runs/


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_git_facts_from_a_real_repository(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout

    git("init", "-q")
    (repo / "uv.lock").write_text("lock\n")
    git("add", "uv.lock")
    git("-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-q", "-m", "lock")
    head, blob = git("rev-parse", "HEAD").strip(), git("rev-parse", "HEAD:uv.lock").strip()
    facts = ac.Git(repo)
    assert facts.head_lock_id() == blob[:12]
    assert facts.commits_present([head, "0" * 40]) == {head: True, "0" * 40: False}
    assert ac.Git(tmp_path / "missing").commits_present([head]) == {head: None}
