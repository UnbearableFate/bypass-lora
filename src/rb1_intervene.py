"""RB1 frozen-history dev evaluation using the unchanged V1 scoring function.

The proxy only supplies side-history options to BranchModel.forward. Candidate
encoding, answer masking, PIQA normalization, BoolQ scoring, ties and NLL remain
v1_run.evaluate/token_losses. Donors are normal same-checkpoint forwards, paired
by candidate index, with a fixed within-task derangement of examples.
"""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer

from calibrate import MODEL, REVISION
from rb_history import donor_indices
from rb_read_diagnostics import ReadSummary
from rb_train import contract, make_model
from v1_data import encode_candidates
from v1_run import batch_tensors, evaluate, prepare
from v5_train import file_sha256, write_json


MODES = ('iv_off', 'iv_window1', 'iv_window8', 'iv_strict', 'iv_swap', 'iv_permute')
INTERVENTION_ARMS = ('RB1-01', 'RB1-02', 'RB1-03', 'RB1-04', 'RB1-08', 'RB1-09', 'W-02')
# E-0026 registers only these three frozen interventions for the r128 memory arm.
WIDTH_MODES = ('iv_off', 'iv_swap', 'iv_permute')


def intervention_modes(arm):
    return ('iv_swap',) if arm == 'RB1-01' else WIDTH_MODES if arm == 'W-02' else MODES


class HistoryView:
    def __init__(self, model, rows, tokenizer, kind=None, trace_reads=True):
        self.model, self.rows, self.tok, self.kind = model, rows, tokenizer, kind
        self.trace_reads = trace_reads and kind != 'iv_off'
        self.donors = donor_indices(rows, 'R1') if kind == 'iv_swap' else None
        self.calls = 0
        self.summary = ReadSummary()

    def eval(self):
        self.model.eval()
        return self

    @torch.no_grad()
    def __call__(self, input_ids, attention_mask, use_cache=False):
        if use_cache or input_ids.shape[0] != 1:
            raise ValueError('RB1 interventions require uncached batch-one V1 scoring')
        row_index, candidate_index = divmod(self.calls, 2)
        if row_index >= len(self.rows):
            raise ValueError('more candidate forwards than registered dev rows')
        row = self.rows[row_index]
        encoded = encode_candidates(self.tok, row)
        if len(encoded) != 2 or input_ids[0].tolist() != encoded[candidate_index]['input_ids']:
            raise ValueError('V1 candidate order/encoding changed during intervention evaluation')
        options = None if self.kind is None else {'kind': self.kind}
        if self.kind == 'iv_swap' and self.model.config.branch != 'kernel_tokenwise':
            donor_row = self.rows[self.donors[row_index]]
            donor_encoded = encode_candidates(self.tok, donor_row)[candidate_index]
            donor_batch = batch_tensors([donor_encoded], self.tok.pad_token_id)
            normal_donor = self.model(input_ids=donor_batch['input_ids'],
                attention_mask=donor_batch['attention_mask'], use_cache=False, capture_history=True)
            options['donor'] = {layer: value['history'] for layer, value in normal_donor.diagnostics.items()}
            del normal_donor
        elif self.kind == 'iv_permute':
            options.update(phase='R1', example_ids=[f'{row["task"]}:{row["id"]}'])
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False,
                            intervention=options, trace_reads=self.trace_reads)
        gold = int(row['label']) if row['task'] == 'piqa' else (0 if row['label'] else 1)
        if self.trace_reads and candidate_index == gold:
            labels = torch.tensor([encoded[candidate_index]['labels']], device=input_ids.device)
            self.summary.add(row['task'], output.diagnostics, attention_mask, labels)
        self.calls += 1
        return output


@contextmanager
def original_backbone(model):
    """Temporarily remove wrappers for an independent same-allocation Base check."""
    adapted = model.adapters()
    for index, layer in adapted.items():
        model.model.model.layers[index] = layer.original
    try:
        yield model.model
    finally:
        for index, layer in adapted.items():
            model.model.model.layers[index] = layer


@torch.no_grad()
def evaluate_history(model, splits, tokenizer, kind=None, limit=None, trace_reads=True):
    rows = [row for task in ('piqa', 'boolq')
            for row in splits['splits'][task]['dev'][:limit]]
    view = HistoryView(model, rows, tokenizer, kind, trace_reads)
    metrics, predictions = evaluate(view, splits, tokenizer, limit, split_names=('dev',))
    if view.calls != 2*len(rows):
        raise ValueError('not all registered candidate forwards completed')
    return metrics, predictions, view.summary.finish()


def outcome_arrays(predictions):
    return {task: [int(row['prediction'] == row['gold']) for row in predictions if row['task'] == task]
            for task in ('piqa', 'boolq')}


def run(args):
    from rb1_artifacts import validate_formal_run
    from rb1_rules import paired_bootstrap
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    passed, values, artifacts = False, {}, []
    start = time.monotonic()
    try:
        if args.arm not in INTERVENTION_ARMS:
            raise ValueError('arm is outside the registered frozen-intervention grid')
        _, inventory, adapter = validate_formal_run(args.training_run, args.arm, args.training_source)
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        splits = prepare(args.data_dir)
        model = make_model(args.arm, phase='R1', seed=0, checkpoint=adapter).eval()
        with original_backbone(model) as backbone:
            base_metrics, base_predictions = evaluate(backbone, splits, tokenizer, split_names=('dev',))
        normal_metrics, normal_predictions, normal_reads = evaluate_history(model, splits, tokenizer)
        conditions = {'normal': {'metrics': normal_metrics, 'predictions': normal_predictions,
                                  'read_diagnostics': normal_reads},
                      'Base_same_allocation': {'metrics': base_metrics, 'predictions': base_predictions}}
        modes = intervention_modes(args.arm)
        for kind in modes:
            tick = time.monotonic()
            metrics, predictions, reads = evaluate_history(model, splits, tokenizer, kind)
            conditions[kind] = {'metrics': metrics, 'predictions': predictions,
                                'read_diagnostics': reads, 'elapsed_s': time.monotonic()-tick}
            # These comparisons share the allocation, model weights and V1 scorer.
            if kind == 'iv_off' and predictions != base_predictions:
                raise ValueError('iv_off differs from independent same-allocation Base scoring')
            if args.arm == 'RB1-01' and predictions != normal_predictions:
                raise ValueError('T0 iv_swap is not a no-op')
            write_json(out/'conditions.json', conditions)
        bootstrap = paired_bootstrap({kind: outcome_arrays(value['predictions'])
                                      for kind, value in conditions.items()})
        write_json(out/'conditions.json', conditions)
        write_json(out/'bootstrap.json', {'repeats': 10000, 'seed': 20261101,
                                         'scope': 'descriptive paired task-stratified dev intervals',
                                         'comparisons': bootstrap})
        provenance = {'formal_run': inventory,
                      'training_manifest': {'path': str((Path(args.training_run)/'manifest.json').resolve()),
                       'sha256': file_sha256(Path(args.training_run)/'manifest.json')},
                      'checkpoint_inventory': {'path': str((Path(args.training_run)/'checkpoint.json').resolve()),
                       'sha256': file_sha256(Path(args.training_run)/'checkpoint.json')},
                      'source_commit': os.environ['SR_CODE_COMMIT'], 'model_revision': REVISION,
                      'tokenizer_revision': REVISION,
                      'environment_lock_sha256': file_sha256(Path(__file__).resolve().parents[1]/'uv.lock'),
                      'diagnostic_scope': 'gold-candidate tokens; actual first/last LatentReader queries; normalized positive-key probabilities',
                      'donor_rule': 'R1 within-task derangement; normal donor same candidate index; donor past clamped to recipient position',
                      'hardware': torch.cuda.get_device_name(), 'seed': 0}
        write_json(out/'provenance.json', provenance)
        values.update(arm=args.arm, conditions=list(conditions), dev_rows=len(normal_predictions),
                      iv_off_identical_to_Base=args.arm != 'RB1-01',
                      T0_swap_noop=args.arm == 'RB1-01')
        artifacts = [{'path': str(out/name), 'role': name} for name in
                     ('conditions.json', 'bootstrap.json', 'provenance.json')]
        passed = True
    finally:
        values['elapsed_s'] = time.monotonic()-start
        write_json(out/'manifest.json', {'status': 'complete' if passed else 'partial',
            'contract': contract(args.arm, 'train'), 'spec_digest': os.environ['SR_SPEC_DIGEST'],
            'arm': args.arm, 'phase': 'R1', 'seed': 0, 'stage': 'intervention',
            'metrics': values, 'artifacts': artifacts,
            'validator': {'name': 'rb1_frozen_history', 'passed': passed}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=INTERVENTION_ARMS, required=True)
    parser.add_argument('--training-run', type=Path, required=True)
    parser.add_argument('--training-source', help='reviewed training source commit (E-0026 W-02 only)')
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
