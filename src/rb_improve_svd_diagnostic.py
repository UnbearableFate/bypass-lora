"""Compare SVD precision and drivers on the same observed I-G matrices in PBS."""
import argparse
import os
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer

from calibrate import MODEL, REVISION
from rb_improve_init import (GradientCollector, basis_orthogonality_error,
                             calibration_examples, scaled_subspace)
from rb_train import load_data, make_model
from v1_run import batch_tensors, example_mean_objective, token_losses
from v5_train import write_json


def run(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    tick = time.monotonic()
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    tok.pad_token_id = tok.eos_token_id
    splits, excluded = load_data(args.data_dir, args.split_manifest, tok)
    examples, samples = calibration_examples(splits, tok, excluded)
    model = make_model('G-T').eval()
    collector = GradientCollector(model)
    try:
        for example in examples:
            model.zero_grad(set_to_none=True)
            batch = batch_tensors([example], tok.pad_token_id)
            collector.valid = batch['attention_mask'].bool()
            losses, _ = token_losses(model, batch)
            example_mean_objective(losses, 2).backward()
    finally:
        collector.close()
    records = {}
    with torch.no_grad():
        for index, stats in collector.stats.items():
            G = stats['G']
            cases = {}
            for dtype in (torch.float32, torch.float64):
                matrix = G.to(dtype)
                for driver in ('gesvdj', 'gesvd'):
                    start = time.monotonic()
                    left, singular, right = torch.linalg.svd(matrix, full_matrices=False, driver=driver)
                    basis = left[:, :32].double()
                    cases[f'{dtype}/{driver}'] = {
                        'orthogonality_max_error_FP64': basis_orthogonality_error(basis),
                        'stored_FP32_orthogonality_max_error': basis_orthogonality_error(basis.float()),
                        'leading_singular_residual_relative': float((G.double() @ right[:32].double().T -
                            basis * singular[:32].double()).norm()/G.double().norm()),
                        'first_singular_value': singular[0].item(),
                        'rank32_singular_value': singular[31].item(),
                        'last_singular_value': singular[-1].item(),
                        'seconds': time.monotonic()-start}
            value, production = scaled_subspace(G, stats['covariance'], stats['Z2'],
                model.adapters()[index].side.reader, 32)
            records[str(index)] = {'gradient_shape': list(G.shape), 'gradient_dtype': str(G.dtype),
                'solver_cases': cases, 'production': production,
                'passed': production['unscaled_orthogonality_max_error'] <= 1e-4}
    passed = all(row['passed'] for row in records.values())
    write_json(out/'svd-diagnostic.json', {'layers': records, 'samples': samples,
        'float32_matmul_precision': torch.get_float32_matmul_precision(),
        'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        'hardware': torch.cuda.get_device_name(), 'job_id': os.environ['PBS_JOBID'],
        'gradient_accumulation_unchanged': 'FP32, as specified by E-0029@1', 'passed': passed})
    write_json(out/'manifest.json', {'status': 'complete' if passed else 'partial',
        'contract': 'E-0029@1', 'spec_digest': os.environ['SR_SPEC_DIGEST'],
        'metrics': {'passed': passed, 'elapsed_s': time.monotonic()-tick, 'dev_read': False},
        'artifacts': [{'path': str(out/'svd-diagnostic.json'), 'role': 'solver_diagnostic'}],
        'validator': {'name': 'e0029_same_gradient_SVD_precision_comparison', 'passed': passed}})
    if not passed:
        raise ValueError('accurate SVD still fails the stored-basis orthogonality check')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--split-manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())
