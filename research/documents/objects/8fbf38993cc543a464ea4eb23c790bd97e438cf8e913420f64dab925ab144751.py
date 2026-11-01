import json

from v1_data import (
    build_training_stream,
    collate_encoded,
    derive_seed,
    encode_candidates,
    encode_training_example,
    make_v1_splits,
    read_boolq_train,
    read_piqa_train,
)


class CharacterTokenizer:
    """Simple deterministic tokenizer whose character IDs expose truncation."""

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(char) for char in text]


def test_readers_keep_official_label_shapes(tmp_path):
    piqa_file = tmp_path / "train.jsonl"
    piqa_file.write_text(json.dumps({"goal": "g", "sol1": "a", "sol2": "b"}) + "\n")
    labels = tmp_path / "train-labels.lst"
    labels.write_text("1\n")
    boolq_file = tmp_path / "boolq.jsonl"
    boolq_file.write_text(json.dumps({"id": "q1", "passage": "p", "question": "q", "answer": True}) + "\n")

    assert read_piqa_train(piqa_file, labels)[0]["label"] == 1
    assert read_boolq_train(boolq_file)[0]["label"] is True


def test_splits_are_deterministic_disjoint_and_auditable():
    piqa = [{"task": "piqa", "id": i} for i in range(30)]
    boolq = [{"task": "boolq", "id": i} for i in range(30)]
    one = make_v1_splits(piqa, boolq, dev_size=4, calibration_size=5)
    two = make_v1_splits(piqa, boolq, dev_size=4, calibration_size=5)
    assert one["metadata"]["split_mapping_sha256"] == two["metadata"]["split_mapping_sha256"]
    for task in ("piqa", "boolq"):
        groups = one["splits"][task]
        ids = [{row["id"] for row in groups[name]} for name in ("train", "dev", "calibration")]
        assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
        assert set.union(*ids) == set(range(30))
    assert derive_seed("a", 2) == derive_seed("a", 2)
    assert derive_seed("a", 2) != derive_seed(2, "a")


def test_interleaving_balances_task_order_and_changes_by_epoch():
    splits = {"piqa": {"train": [{"task": "piqa", "id": i} for i in range(5)]},
              "boolq": {"train": [{"task": "boolq", "id": i} for i in range(2)]}}
    stream = build_training_stream(splits, epoch=0)
    assert [row["task"] for row in stream[:4]] == ["piqa", "boolq", "piqa", "boolq"]
    assert len(stream) == 7
    assert sorted((row["task"], row["id"]) for row in stream) == sorted(
        [("piqa", i) for i in range(5)] + [("boolq", i) for i in range(2)])
    assert stream != build_training_stream(splits, epoch=1)


def test_candidate_encoding_uses_shared_prompt_and_completion_only_loss():
    tokenizer = CharacterTokenizer()
    example = {"task": "piqa", "id": 7, "goal": "abcdefghijklmnop",
               "solutions": ["short", "longer"], "label": 1}
    candidates = encode_candidates(tokenizer, example, max_length=18)
    assert candidates[0]["input_ids"][:candidates[0]["prompt_length"]] == candidates[1]["input_ids"][:candidates[1]["prompt_length"]]
    assert candidates[0]["prompt_length"] == candidates[1]["prompt_length"]
    assert len(candidates[0]["input_ids"]) <= 18
    assert candidates[1]["input_ids"][-len(" longer"):] == [ord(c) for c in " longer"]
    selected = encode_training_example(tokenizer, example, max_length=18)
    assert selected["candidate"] == " longer"
    assert selected["labels"][:selected["prompt_length"]] == [-100] * selected["prompt_length"]
    assert selected["labels"][selected["prompt_length"]:] == [ord(c) for c in " longer"]


def test_boolq_truncation_preserves_question_and_pad_eos_mask():
    tokenizer = CharacterTokenizer()
    example = {"task": "boolq", "id": "q", "passage": "x" * 100,
               "question": "kept?", "label": False}
    encoded = encode_training_example(tokenizer, example, max_length=40)
    prompt = "".join(chr(x) for x in encoded["input_ids"][:encoded["prompt_length"]])
    assert "Question: kept?\nAnswer:" in prompt
    assert "x" in prompt and len(encoded["input_ids"]) <= 40
    other = {"task": "boolq", "input_ids": [4, 9], "labels": [-100, 9]}
    padded = collate_encoded([other, {"task": "boolq", "input_ids": [4], "labels": [-100]}], pad_token_id=4)
    assert padded["input_ids"] == [[4, 9], [4, 4]]
    assert padded["attention_mask"] == [[1, 1], [1, 0]]
    assert padded["labels"] == [[-100, 9], [-100, -100]]
