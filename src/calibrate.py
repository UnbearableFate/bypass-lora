"""Base-only numerical calibration. Run before adapter checks, on their hardware."""
import json
import os
import platform
import sys
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B-Base"
REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"


def load_base():
    torch.set_num_threads(4)
    torch.manual_seed(173)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="eager",
    ).cuda().eval()
    model.requires_grad_(False)
    return model


def fixture(device="cuda"):
    # Fixed IDs require no task data and exercise differing query/key positions.
    return torch.tensor([[50, 99, 234, 71, 800, 43, 91, 120],
                         [53, 82, 910, 44, 306, 72, 11, 207]], device=device)


def cached_logits(model, ids, mask, chunk):
    past = None
    result = []
    pos = (mask.cumsum(-1) - 1).clamp_min(0)
    for start in range(0, ids.shape[1], chunk):
        end = min(start + chunk, ids.shape[1])
        out = model(input_ids=ids[:, start:end], attention_mask=mask[:, :end],
                    position_ids=pos[:, start:end], past_key_values=past, use_cache=True)
        past = out.past_key_values
        result.append(out.logits.float())
    return torch.cat(result, dim=1)


def main(out):
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    model = load_base()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    ids = fixture()
    measurements = {}
    with torch.no_grad():
        for layout in ("none", "left", "right"):
            mask = torch.ones_like(ids)
            if layout == "left":
                mask[0, :2] = 0
            elif layout == "right":
                mask[0, -2:] = 0
            pos = (mask.cumsum(-1) - 1).clamp_min(0)
            reference = model(ids, attention_mask=mask, position_ids=pos, use_cache=False).logits.float()
            for repeat in range(3):
                for chunk in (8, 3, 1):
                    observed = cached_logits(model, ids, mask, chunk)
                    diff = (reference - observed).abs()[mask.bool()]
                    measurements[f"{layout}.repeat{repeat}.chunk{chunk}.max_abs"] = diff.max().item()
    # Pre-fixed rule: twice largest measured Base discrepancy + FP32 rounding floor.
    tolerance = 2 * max(measurements.values()) + 2e-5
    config = {"bf16_atol": tolerance, "bf16_rtol": 0.0,
              "fp32_atol": 2e-5, "fp32_rtol": 2e-5,
              "rule": "2 * max Base discrepancy + 2e-5; three repeats; full/chunk3/token; no/left/right padding; valid positions",
              "model_revision": REVISION, "tokenizer_revision": REVISION,
              "hardware": torch.cuda.get_device_name(), "torch": torch.__version__,
              "cuda": torch.version.cuda, "transformers": transformers.__version__,
              "python": platform.python_version(), "tokenizer_class": type(tokenizer).__name__,
              "measurements": measurements}
    (out / "tolerances.json").write_text(json.dumps(config, indent=2) + "\n")
    (out / "manifest.json").write_text(json.dumps({
        "status": "complete", "spec_digest": os.environ["SR_SPEC_DIGEST"],
        "metrics": {"calibration.pass": True, "bf16_atol": tolerance,
                    "elapsed_s": time.monotonic() - start, **measurements},
        "artifacts": [{"path": str(out / "tolerances.json"), "role": "tolerance_configuration"}],
        "validator": {"name": "base_only_calibration", "passed": True},
    }, indent=2) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
