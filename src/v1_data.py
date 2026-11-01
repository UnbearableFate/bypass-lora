"""Deterministic V1 PIQA/BoolQ data preparation and completion encoding.

The module only reads already-present official JSONL/list files; it never fetches
data. Encoded examples use labels ``-100`` for prompt and padding positions.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


_SEED_PREFIX = (20261029, "concat_loop_v2", "V1")
_TASKS = ("piqa", "boolq")


def derive_seed(*parts: Any) -> int:
    """Hash arguments as canonical JSON and return the first 64 bits (big-endian)."""
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            rows.append(value)
    return rows


def read_piqa_train(jsonl_path: str | Path, labels_path: str | Path) -> list[dict[str, Any]]:
    """Read official PIQA train JSONL and its separate train-labels.lst file."""
    rows = _read_jsonl(jsonl_path)
    labels = [line.strip() for line in Path(labels_path).read_text(encoding="utf-8").splitlines()
              if line.strip()]
    if len(rows) != len(labels):
        raise ValueError(f"PIQA row/label count mismatch: {len(rows)} rows, {len(labels)} labels")
    result = []
    for i, (row, label_text) in enumerate(zip(rows, labels)):
        try:
            label = int(label_text)
        except ValueError as exc:
            raise ValueError(f"PIQA label {i} is not an integer: {label_text!r}") from exc
        if label not in (0, 1):
            raise ValueError(f"PIQA label {i} must be 0 or 1, got {label}")
        if not all(isinstance(row.get(key), str) for key in ("goal", "sol1", "sol2")):
            raise ValueError(f"PIQA row {i} is missing string goal/sol1/sol2 fields")
        result.append({"task": "piqa", "id": i, "goal": row["goal"],
                       "solutions": [row["sol1"], row["sol2"]], "label": label})
    return result


def read_boolq_train(jsonl_path: str | Path) -> list[dict[str, Any]]:
    """Read official BoolQ train JSONL or parquet, preserving boolean answers.

    Parquet support is intentionally local and imports pyarrow only when needed.
    """
    path = Path(jsonl_path)
    if path.suffix.lower() == ".parquet":
        try:
            import pyarrow.parquet as parquet
        except ImportError as exc:
            raise RuntimeError("reading BoolQ parquet requires pyarrow") from exc
        rows = parquet.read_table(path).to_pylist()
    else:
        rows = _read_jsonl(path)
    result = []
    for i, row in enumerate(rows):
        if not isinstance(row.get("passage"), str) or not isinstance(row.get("question"), str):
            raise ValueError(f"BoolQ row {i} is missing string passage/question fields")
        answer = row.get("answer")
        if not isinstance(answer, bool):
            raise ValueError(f"BoolQ row {i} answer must be a JSON boolean")
        result.append({"task": "boolq", "id": row.get("id", i),
                       "passage": row["passage"], "question": row["question"],
                       "label": answer})
    return result


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_v1_splits(
    piqa: Sequence[Mapping[str, Any]],
    boolq: Sequence[Mapping[str, Any]],
    *,
    source_files: Mapping[str, str | Path] | None = None,
    source_metadata: Mapping[str, Any] | None = None,
    dev_size: int = 1000,
    calibration_size: int = 2048,
) -> dict[str, Any]:
    """Split each official training set independently; return slices and audit metadata.

    ``source_files`` is an optional name→path mapping (normally the three source
    files) used to include input SHA256 values in metadata.
    """
    if dev_size < 0 or calibration_size < 0:
        raise ValueError("split sizes must be non-negative")
    slices: dict[str, dict[str, list[dict[str, Any]]]] = {}
    split_ids: dict[str, dict[str, list[Any]]] = {}
    for task, data in (("piqa", piqa), ("boolq", boolq)):
        if len(data) < dev_size + calibration_size:
            raise ValueError(f"{task} has {len(data)} rows; needs at least {dev_size + calibration_size}")
        order = list(range(len(data)))
        purpose = f"{task}-split"
        random.Random(derive_seed(*_SEED_PREFIX, purpose)).shuffle(order)
        ids = {"dev": order[:dev_size],
               "calibration": order[dev_size:dev_size + calibration_size],
               "train": order[dev_size + calibration_size:]}
        slices[task] = {name: [dict(data[i]) for i in indices]
                        for name, indices in ids.items()}
        split_ids[task] = {name: [data[i].get("id", i) for i in indices]
                           for name, indices in ids.items()}
    encoded_ids = json.dumps(split_ids, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":")).encode("utf-8")
    return {"splits": slices,
            "metadata": {"split_seed_rule": "derive_seed(20261029, concat_loop_v2, V1, {task}-split)",
                         "dev_size": dev_size, "calibration_size": calibration_size,
                         "split_ids": split_ids,
                         "split_mapping_sha256": hashlib.sha256(encoded_ids).hexdigest(),
                         "source_metadata": dict(source_metadata or {}),
                         "source_files": {name: {"path": str(path), "sha256": _sha256_file(path)}
                                          for name, path in sorted((source_files or {}).items())}}}


def build_training_stream(splits: Mapping[str, Any], epoch: int = 0) -> list[dict[str, Any]]:
    """Shuffle each task's train rows then alternate tasks while both have rows.

    Every example appears once per epoch. The epoch RNG seed is the fixed V1
    training seed plus the zero-based epoch index.
    """
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    train_data = splits.get("splits", splits)
    task_rows = {task: list(train_data[task]["train"]) for task in _TASKS}
    rng = random.Random(derive_seed(*_SEED_PREFIX, "train", 0) + epoch)
    for values in task_rows.values():
        rng.shuffle(values)
    stream: list[dict[str, Any]] = []
    positions = {task: 0 for task in _TASKS}
    while any(positions[t] < len(task_rows[t]) for t in _TASKS):
        for task in _TASKS:
            pos = positions[task]
            if pos < len(task_rows[task]):
                stream.append(dict(task_rows[task][pos]))
                positions[task] += 1
    return stream


def _encode(tokenizer: Any, text: str) -> list[int]:
    """Tokenizer adapter that deliberately adds no EOS/special token."""
    if hasattr(tokenizer, "encode"):
        ids = tokenizer.encode(text, add_special_tokens=False)
    else:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(token) for token in ids]


def _prompt_tokens(tokenizer: Any, example: Mapping[str, Any], budget: int) -> list[int]:
    if example["task"] == "piqa":
        prompt = _encode(tokenizer, f"Goal: {example['goal']}\nSolution:")
        return prompt[-budget:] if budget else []
    if example["task"] == "boolq":
        passage = _encode(tokenizer, f"Passage: {example['passage']}\n")
        question = _encode(tokenizer, f"Question: {example['question']}\nAnswer:")
        if len(question) > budget:
            raise ValueError("BoolQ question/template exceeds prompt token budget")
        # Right-truncate passage; preserve the complete question and answer cue.
        return passage[:budget - len(question)] + question
    raise ValueError(f"unknown task: {example['task']!r}")


def encode_candidates(tokenizer: Any, example: Mapping[str, Any], max_length: int = 512) -> list[dict[str, Any]]:
    """Encode every candidate against one shared truncated prompt prefix.

    PIQA candidates are ``' ' + solution``; BoolQ candidates are ``' yes'`` and
    ``' no'``. The longest candidate reserves space, so candidate input lengths
    never force different prompt truncation. Candidate tokens are never cut.
    """
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    if example["task"] == "piqa":
        completions = [" " + str(x) for x in example["solutions"]]
    elif example["task"] == "boolq":
        completions = [" yes", " no"]
    else:
        raise ValueError(f"unknown task: {example['task']!r}")
    completion_ids = [_encode(tokenizer, text) for text in completions]
    reserve = max(map(len, completion_ids), default=0)
    if reserve > max_length:
        raise ValueError("a complete candidate exceeds max_length")
    prompt = _prompt_tokens(tokenizer, example, max_length - reserve)
    result = []
    for text, suffix in zip(completions, completion_ids):
        input_ids = prompt + suffix
        result.append({"task": example["task"], "id": example.get("id"),
                       "candidate": text, "input_ids": input_ids,
                       "labels": [-100] * len(prompt) + suffix.copy(),
                       "prompt_length": len(prompt), "completion_length": len(suffix)})
    return result


def encode_training_example(tokenizer: Any, example: Mapping[str, Any], max_length: int = 512) -> dict[str, Any]:
    """Encode the gold PIQA solution or BoolQ yes/no completion for CE training."""
    candidates = encode_candidates(tokenizer, example, max_length=max_length)
    label = example["label"]
    if example["task"] == "piqa":
        selected = candidates[int(label)]
    else:
        selected = candidates[0 if label is True else 1]
    return dict(selected)


def collate_encoded(examples: Sequence[Mapping[str, Any]], pad_token_id: int) -> dict[str, list[list[int]]]:
    """Dynamically right-pad; attention masks reflect sequence positions, not token IDs."""
    width = max((len(example["input_ids"]) for example in examples), default=0)
    input_ids, labels, attention_mask = [], [], []
    for example in examples:
        n = len(example["input_ids"])
        pad = width - n
        input_ids.append(list(example["input_ids"]) + [pad_token_id] * pad)
        labels.append(list(example["labels"]) + [-100] * pad)
        attention_mask.append([1] * n + [0] * pad)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
