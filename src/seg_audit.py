"""Read-only E-0025 artifact audit using the reviewed validation functions.

Run in PBS. This loads no model, changes no scientific path or gate, and does
not read confirmation data. Formal/intervention/analysis source stays 7d8b346.
"""
import argparse
import os
from pathlib import Path
import time

from rb1_artifacts import read, reference
from seg_analysis import summary
from seg_artifacts import validate, validate_rows
from v5_train import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    passed, metrics, started = False, {}, time.monotonic()
    try:
        inventory = read(args.inventory)
        source = inventory['code_commit']
        if source != '7d8b3464b6145504d155268f4c667fa3ecc53b00':
            raise ValueError('audit source must equal the final RV-0079 release')
        results, paired, exclusions, answer_counts = {}, None, None, None
        for run in inventory['runs']:
            arm, root = run['arm'], Path(run['output'])
            manifest, refs, _ = validate(root, arm, source)
            rows = read(root/'predictions.json')
            if paired is None:
                paired = [(r['task'], r['id'], r['gold']) for r in rows]
                answer_counts = [r['gold_loss_tokens'] for r in rows]
            validate_rows(rows, paired)
            if [r['gold_loss_tokens'] for r in rows] != answer_counts:
                raise ValueError('candidate token counts changed across segmented arms')
            current = read(root/'excluded-train.json')
            if exclusions is None: exclusions = current
            if current != exclusions or {(r['task'], r['id']) for r in current} != {('piqa', 5528)}:
                raise ValueError('segmented training exclusions differ from RB1')
            results[arm] = {'refs': refs, 'metrics': manifest['metrics'], 'summary': summary(rows)}
        if set(results) != {f'SEG-{i:02}' for i in range(5)}:
            raise ValueError('audit requires the complete initial grid')
        checked = read(inventory['deterministic_check']['manifest'])
        if (checked.get('source_commit') != source or checked.get('status') != 'complete' or
            checked.get('validator', {}).get('passed') is not True or
            checked['metrics'].get('no_channel_identity') is not True):
            raise ValueError('exact-source no-channel deterministic receipt failed')
        write_json(out/'audit.json', {'training_source': source, 'coverage_decision': 'D-0049',
            'arms': results, 'excluded': exclusions, 'deterministic_check': reference(inventory['deterministic_check']['manifest']),
            'checks': {'initial_identity': True, 'reload_identity': True, 'no_channel_identity': True,
                       'full_dev_and_curves': True, 'checkpoint_binding': True,
                       'sample_order_budget_LR_accounting': True, 'iv_off_pending': True}})
        metrics.update({'audited_arms': 5, 'trained_arms': 4, 'dev_per_task': 1000,
                        'formal_steps_per_training_arm': 13115, 'passed': True,
                        'initial.reload.no_channel_checks': True, 'iv_off_pending': True,
                        'confirmation_evaluated': False})
        passed = True
    finally:
        metrics['elapsed_s'] = time.monotonic()-started
        write_json(out/'manifest.json', {'status': 'complete' if passed else 'partial', 'contract': 'E-0025@1',
            'spec_digest': os.environ['SR_SPEC_DIGEST'], 'source_commit': os.environ['SR_CODE_COMMIT'],
            'metrics': metrics, 'artifacts': [{'path': str(out/'audit.json'), 'role': 'reviewed_artifact_audit'}] if (out/'audit.json').exists() else [],
            'validator': {'name': 'segmented_final_artifact_audit', 'passed': passed}})


if __name__ == '__main__':
    main()
