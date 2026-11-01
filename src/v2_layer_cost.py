"""Bounded E-0012 full-shape cost measurement, inside a PBS MIG allocation."""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from torch import nn
import torch.nn.functional as F

from calibrate import load_base
from v2_layer_diagnostics import (BilinearAddress, checkpoints_from_map, classifier_score,
                                  geometry, write_json)
from v2_train import adapted


def timed(action):
    torch.cuda.synchronize()
    tick = time.monotonic()
    action()
    torch.cuda.synchronize()
    return time.monotonic() - tick


def linear_case(width, support):
    n = 14336 * (16 if support else 1)
    nv = 2048 * (16 if support else 1)
    batch_size = 8192 if support else 1024
    x = torch.randn(n, width, device='cuda')
    y = torch.randint(92, (n,), device='cuda')
    validation_x = torch.randn(nv, width, device='cuda')
    validation_y = torch.randint(92, (nv,), device='cuda')
    model = nn.Linear(width, 92, device='cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=.03)

    def epoch():
        order = torch.randperm(n, device='cuda')
        for index in order.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x[index]), y[index]) + .5e-4 * model.weight.square().sum()
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite benchmark loss')
            loss.backward()
            optimizer.step()

    timed(epoch)  # warm optimizer buffers and kernels
    training = timed(epoch)
    validation = timed(lambda: classifier_score(model, validation_x, validation_y))
    return {'fit_examples': n, 'validation_examples': nv, 'width': width,
            'classes': 92, 'batch_size': batch_size,
            'epoch_s': training, 'validation_s': validation}


def address_case(rank):
    # Maximum query width bounds both feature families, with full rank-1024 keys.
    q = torch.randn(16384, 2048, device='cuda')
    k = torch.randn(16384, 16, 1024, device='cuda')
    y = torch.randint(16, (16384,), device='cuda')
    model = BilinearAddress(2048, 1024, rank, 'cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)

    def epoch():
        order = torch.randperm(14336, device='cuda')
        for index in order.split(512):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model((q[index], k[index])), y[index])
            loss += .5e-4 * (model.U.square().sum() + model.V.square().sum())
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite benchmark loss')
            loss.backward()
            optimizer.step()

    def validation():
        from v2_layer_diagnostics import address_score
        address_score(model, q[14336:], k[14336:], y[14336:])

    timed(epoch)
    training, checking = timed(epoch), timed(validation)
    return {'rank': rank, 'query_width': 2048, 'key_width': 1024, 'facts': 16,
            'fit_episodes': 14336, 'validation_episodes': 2048, 'batch_size': 512,
            'epoch_s': training, 'validation_s': checking}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='/work/xg24i002/x10041/bypass-lora')
    parser.add_argument('--smoke-output', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out, project, smoke_out = Path(args.output).resolve(), Path(args.project), Path(args.smoke_output)
    out.mkdir(parents=True, exist_ok=True)
    result, success = {}, False
    tick = time.monotonic()
    try:
        torch.set_num_threads(4)
        torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        result['environment'] = {'torch': torch.__version__, 'cuda': torch.version.cuda,
                                 'python': sys.version, 'device': torch.cuda.get_device_name()}
        result['linear'] = {}
        for name, width, support in (('support1024', 1024, True), ('question2048', 2048, False),
                                     ('support128', 128, True), ('question128', 128, False)):
            result['linear'][name] = linear_case(width, support)
            gc.collect(); torch.cuda.empty_cache()
        result['address'] = {str(rank): address_case(rank) for rank in (128, 256, 512, 1024)}
        gc.collect(); torch.cuda.empty_cache()
        checkpoints = checkpoints_from_map(project / 'research/artifacts/T-0011/training-run-map.json')
        base = load_base()
        cache = torch.load(smoke_out / 'features.dev.pt', map_location='cpu', weights_only=True)
        result['geometry_smoke'] = {}
        for checkpoint in checkpoints:
            with adapted(base, checkpoint=checkpoint['checkpoint']) as model:
                model.eval()
                result['geometry_smoke'][checkpoint['arm']] = geometry(base, model.adapters()[14].side, cache,
                    checkpoint['tap'], checkpoint['backend'])
        del base, cache
        gc.collect(); torch.cuda.empty_cache()
        l3_output = out / 'l3-full-shape'
        argv = [sys.executable, str(Path(__file__).with_name('v2_probe.py')), '--checkpoint', checkpoints[0]['checkpoint'],
                '--train', str(project / 'runs/ATT-0039-1/data/train.jsonl'),
                '--heldout', str(project / 'runs/ATT-0039-1/data/calibration.jsonl'),
                '--limit', '4096', '--batch-size', '16', '--output', str(l3_output)]
        l3_start = time.monotonic()
        with (out / 'l3.stdout.log').open('w') as log:
            subprocess.run(argv, check=True, stdout=log, stderr=subprocess.STDOUT)
        result['full_shape_L3_external_s'] = time.monotonic() - l3_start
        l3 = json.loads((l3_output / 'manifest.json').read_text())['metrics']
        if l3['train_examples'] != 4096 or l3['test_examples'] != 2048:
            raise ValueError('L3 cost run did not execute the full registered shape')
        result['full_shape_L3_internal_s'] = l3['elapsed_s']
        smoke = json.loads((smoke_out / 'report.json').read_text())
        extraction = sum(smoke['timings']['extract_' + s + '_s'] * n / smoke['split_counts'][s]
                         for s, n in (('train', 16384), ('calibration', 2048), ('dev', 1000)))
        linear = result['linear']
        epoch_counts = {'support1024': (4 * 64 + 6 * 32) * 3,
                        'question2048': (2 * 64 + 8 * 32) * 3,
                        'support128': 36 * 32 * 3, 'question128': 48 * 32 * 3}
        probe_counts = {'support1024': 10, 'question2048': 10, 'support128': 36, 'question128': 48}
        linear_s = sum(linear[k]['epoch_s'] * v for k, v in epoch_counts.items())
        linear_s += sum(linear[k]['validation_s'] * v * 12 for k, v in probe_counts.items())
        address_s = sum(4 * (32 * 3 * row['epoch_s'] + 12 * row['validation_s']) for row in result['address'].values())
        l3_s = 10 * result['full_shape_L3_external_s']
        # Add 20% for normalization, copies, hashes, saves, geometry and imports.
        estimated = (extraction + linear_s + address_s + l3_s) * 1.2
        result['full_estimate'] = {'extraction_s': extraction, 'linear_s': linear_s,
            'address_s': address_s, 'L3_s': l3_s, 'overhead_multiplier': 1.2,
            'expected_s': math.ceil(estimated), 'walltime_s_at_1_5': math.ceil(1.5 * estimated),
            'expected_node_h': estimated / 3600 * .25,
            'basis': 'full-size synthetic epochs and validation; all 16 address probes bounded by query_width2048; actual full-shape L3 on V2-01; measured smoke extraction; actual checkpoint geometry smoke; no scientific interpretation before full L0'}
        success = True
    finally:
        result['elapsed_s'] = time.monotonic() - tick
        write_json(out / 'cost.json', result)
        write_json(out / 'manifest.json', {'status': 'complete' if success else 'partial',
            'spec_digest': os.environ['SR_SPEC_DIGEST'],
            'metrics': {'elapsed_s': result['elapsed_s'], **result.get('full_estimate', {})},
            'artifacts': [{'path': str(out / 'cost.json'), 'role': 'full_shape_cost_and_geometry_validation'}],
            'validator': {'name': 'E0012_full_shape_cost', 'passed': success}})
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
