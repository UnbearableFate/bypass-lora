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


def seam_summary(rows):
    """Descriptive encoding counts; does not imply segment 1 was executed."""
    result = {}
    for task in ('piqa', 'boolq'):
        subset = [r for r in rows if r['task'] == task]
        if not subset:
            raise ValueError('both tasks need seam records')
        if any(len(r['candidate_tokens']) != 2 or min(r['candidate_tokens']) <= 0 or
               r['segment1_tokens'] <= 0 or r['segment2_prompt_tokens'] <= 0 or
               r['segment1_tokens']+r['segment2_prompt_tokens']+max(r['candidate_tokens']) > 512
               for r in subset):
            raise ValueError('invalid segmented token-budget/candidate seam metadata')
        result[task] = {'rows': len(subset), 'unique_ids': len({r['id'] for r in subset}),
            'identical_prompts': sum(r['identical'] for r in subset),
            'changed_prompts': sum(not r['identical'] for r in subset),
            'v1_changed_tokens': sum(r['v1_changed_tokens'] for r in subset),
            'seg_changed_tokens': sum(r['seg_changed_tokens'] for r in subset),
            'v1_prompt_tokens': sum(r['v1_prompt_tokens'] for r in subset),
            'seg_prompt_tokens': sum(r['seg_prompt_tokens'] for r in subset),
            'encoded_segment1_tokens': sum(r['segment1_tokens'] for r in subset),
            'segment2_prompt_tokens': sum(r['segment2_prompt_tokens'] for r in subset),
            'first_segment_cropped_rows': sum(r['full_segment1_tokens'] > r['segment1_tokens'] for r in subset),
            'first_segment_cropped_tokens': sum(r['full_segment1_tokens']-r['segment1_tokens'] for r in subset)}
    return result


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
        training_seams, dev_seams, seam_refs = None, None, {}
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
            current_train = read(root/'seams-train.json')
            current_dev = read(root/'evaluation-audit.json')['seams']
            if training_seams is None:
                training_seams, dev_seams = current_train, current_dev
            if current_train != training_seams or current_dev != dev_seams:
                raise ValueError('encoded seam records differ across paired arms')
            if len(current_train) != 26230 or len(current_dev) != 2000:
                raise ValueError('seam accounting requires every presentation and dev row')
            seam_refs[arm] = {'training': reference(root/'seams-train.json'),
                              'dev': reference(root/'evaluation-audit.json')}
        if set(results) not in ({f'SEG-{i:02}' for i in range(5)}, {f'SEG-{i:02}' for i in range(8)}):
            raise ValueError('audit requires the complete initial or expanded grid')
        checked = read(inventory['deterministic_check']['manifest'])
        if (checked.get('source_commit') != source or checked.get('status') != 'complete' or
            checked.get('validator', {}).get('passed') is not True or
            checked['metrics'].get('no_channel_identity') is not True):
            raise ValueError('exact-source no-channel deterministic receipt failed')
        write_json(out/'audit.json', {'training_source': source, 'coverage_decision': 'D-0049',
            'arms': results, 'excluded': exclusions, 'deterministic_check': reference(inventory['deterministic_check']['manifest']),
            'seams': {'training_presentations': seam_summary(training_seams), 'dev': seam_summary(dev_seams),
                      'all_arms_encoding_identical': True, 'per_row_refs': seam_refs,
                      'scope': 'Encoding counts only; actual executed tokens are in per-arm metrics. Repeated training IDs counted per presentation.'},
            'checks': {'initial_identity': True, 'reload_identity': True, 'no_channel_identity': True,
                       'full_dev_and_curves': True, 'checkpoint_binding': True,
                       'sample_order_budget_LR_accounting': True, 'iv_off_pending': True}})
        metrics.update({'audited_arms': len(results), 'trained_arms': len(results)-1, 'dev_per_task': 1000,
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
