"""E-0026 pre-training registration: implementation parameter counts and paired init.

Run only in PBS. Counts every adapter by instantiating it on the pinned Base,
selects the all-7 LoRA rank nearest to W-02 from the implementation counts,
and checks that W-01 and W-02 start from elementwise-identical parameters.
"""
import argparse
import os
from pathlib import Path

import torch

from calibrate import REVISION, load_base
from rb_train import LAYERS, LORA_RANKS, PARAMETERS, make_model
from v1_lora import LoRAConfig, LoRAModel
from v5_train import write_json


def counts(model):
    stored = sum(p.numel() for p in model.adapter_parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'stored': stored, 'trainable': trainable}


def nearest_rank(target, per_rank):
    """Nearest integer rank by absolute count difference; ties go to fewer parameters."""
    return min(per_rank, key=lambda rank: (abs(per_rank[rank] - target), per_rank[rank]))


def run(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    result = {'contract': 'E-0026@1', 'model_revision': REVISION,
              'source_commit': os.environ.get('SR_CODE_COMMIT', 'unknown'), 'arms': {}}
    states = {}
    for arm in ('W-01', 'W-02', 'W-03'):
        model = make_model(arm)  # asserts the registry count against the implementation
        result['arms'][arm] = {**counts(model), 'registered': PARAMETERS[arm],
                               'config': {k: v for k, v in vars(model.config).items()}}
        if arm != 'W-03':
            states[arm] = {name: value.detach().cpu().clone()
                           for name, value in model.named_parameters() if value.requires_grad}
        del model
        torch.cuda.empty_cache()
    per_rank = {}
    for rank in range(args.min_rank, args.max_rank + 1):
        model = LoRAModel(load_base(), LoRAConfig(layers=LAYERS, rank=rank, alpha=rank))
        per_rank[rank] = counts(model)['trainable']
        del model
        torch.cuda.empty_cache()
    target = result['arms']['W-02']['trainable']
    chosen = nearest_rank(target, per_rank)
    result['lora_rank_counts'] = {str(k): v for k, v in per_rank.items()}
    result['lora_rank_nearest_W-02'] = chosen
    result['lora_rank_registered'] = LORA_RANKS['W-03']
    keys = sorted(states['W-01'])
    result['W-01_W-02_init'] = {
        'tensors': len(keys),
        'same_names': keys == sorted(states['W-02']),
        'elementwise_identical': keys == sorted(states['W-02']) and all(
            torch.equal(states['W-01'][k], states['W-02'][k]) for k in keys)}
    checks = {
        'counts_match_registry': all(v['stored'] == v['trainable'] == v['registered']
                                     for v in result['arms'].values()),
        'W-01_equals_W-02': result['arms']['W-01']['trainable'] == result['arms']['W-02']['trainable'],
        'W-03_rank_is_nearest': chosen == LORA_RANKS['W-03'],
        'W-03_count_from_rank_scan': per_rank.get(LORA_RANKS['W-03']) == PARAMETERS['W-03'],
        'W-01_W-02_init_identical': result['W-01_W-02_init']['elementwise_identical'],
    }
    result['checks'] = checks
    result['passed'] = all(checks.values())
    write_json(out / 'width-registration.json', result)
    if not result['passed']:
        raise SystemExit(f'E-0026 registration check failed: {checks}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--min-rank', type=int, default=28)
    parser.add_argument('--max-rank', type=int, default=34)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
