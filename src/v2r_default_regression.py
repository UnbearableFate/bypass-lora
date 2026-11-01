"""E-0014 default-off regression: V2 computations under two code trees must agree.

`dump --src DIR` imports the V2 modules from DIR (only APIs that existed before
E-0014) and saves losses, logits, gradients, four optimizer steps of the formal
train() and evaluation logits for every control. `compare` checks two dumps for
exact equality. Trees that have the E-0015 isolated-encoding controls (v2r_diagnostics)
also dump them for the soft_slot arm (E-0016 D4/D5 refactor of that builder).
Run both in one PBS allocation; never on a login node.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile

ARMS = (('direct_hidden', 'soft_slot', 'F_out'), ('direct_hidden', 'kernel_matrix', 'F_out'),
        ('qknorm_pre_rope', 'soft_slot', 'C_up'))


def dump(src, data_dir, output):
    sys.path.insert(0, str(Path(src).resolve()))
    import torch
    from transformers import AutoTokenizer
    from calibrate import MODEL, REVISION, load_base
    from v2_memory import CONTROLS, collate, config, evaluate_batch, train_forward
    from v2_schedule import load_data, make_schedule
    import v2_train
    assert Path(v2_train.__file__).resolve().parent == Path(src).resolve()
    rows = load_data(Path(data_dir) / 'train.jsonl', 'train')
    dev = load_data(Path(data_dir) / 'dev.jsonl', 'dev')[:16]
    schedule = make_schedule(rows, 4_000_000)
    schedule['batches'] = schedule['batches'][:4]
    schedule['steps'], schedule['warmup_steps'] = 4, 1
    schedule['tokens'] = sum(len(rows[i]['support_input_ids']) + len(rows[i]['question_input_ids'])
                             for b in schedule['batches'] for i in b)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    pad = tok.pad_token_id
    try:
        from v2_memory import ISOLATED_CONTROLS
        from v2r_diagnostics import annotate
    except ImportError:
        ISOLATED_CONTROLS = ()
    else:
        annotate(tok, dev, 16)
    base = load_base()
    result = {}
    for arm in ARMS:
        key = '.'.join(arm)
        with v2_train.adapted(base, cfg=config(*arm)) as model:
            side = model.adapters()[14].side
            out = {'init': {n: p.detach().cpu().clone() for n, p in side.named_parameters()}}
            model.train()
            model.model.eval()
            batch = collate([rows[i] for i in schedule['batches'][0]], pad, 'cuda')
            loss, logits, _, _ = train_forward(model, batch)
            loss.backward()
            out['loss'] = loss.detach().cpu()
            out['logits'] = logits.detach().cpu()
            out['grads'] = {n: p.grad.detach().cpu().clone() for n, p in side.named_parameters()}
            model.zero_grad(set_to_none=True)
            with tempfile.TemporaryDirectory() as tmp:
                train = v2_train.train(model, rows, schedule, pad, 3e-3, 8, Path(tmp), dev[:4], 0)
                history = [json.loads(x) for x in (Path(tmp) / 'train.jsonl').read_text().splitlines()]
            out['history'] = [{k: h[k] for k in ('step', 'tokens', 'loss', 'lr', 'gradient_norm')} for h in history]
            out['train'] = {k: train[k] for k in ('steps', 'tokens', 'trainable_parameters',
                                                    'parameter_names_with_nonzero_gradient')}
            out['trained'] = {n: p.detach().cpu().clone() for n, p in side.named_parameters()}
            model.eval()
            dev_batch = collate(dev, pad, 'cuda')
            donors = collate(dev[-1:] + dev[:-1], pad, 'cuda')
            out['eval'] = {}
            isolated = ISOLATED_CONTROLS if arm[1] == 'soft_slot' else ()
            with torch.no_grad():
                for mode in CONTROLS + isolated:
                    details, eval_logits = evaluate_batch(model, dev_batch, mode,
                                                          **({'donor_batch': donors} if mode == 'shuffled' else {}))
                    out['eval'][mode] = {'logits': eval_logits.cpu(),
                                         'details': {k: v for k, v in details.items()
                                                     if not k.startswith('_') and isinstance(v, (int, float))}}
        result[key] = out
    torch.save(result, output)


def compare(paths, output):
    import torch
    a, b = (torch.load(p, weights_only=False) for p in paths)
    report, ok = {}, True

    def walk(x, y, path):
        nonlocal ok
        if isinstance(x, dict):
            if set(x) != set(y):
                # New code may add diagnostics; old keys must all exist and agree.
                missing = set(x) - set(y)
                if missing:
                    report[path + '.missing'] = sorted(missing)
                    ok = False
            for k in x:
                if k in y:
                    walk(x[k], y[k], f'{path}.{k}')
        elif isinstance(x, list):
            if len(x) != len(y):
                report[path] = 'length'
                ok = False
            for i, (u, v) in enumerate(zip(x, y)):
                walk(u, v, f'{path}[{i}]')
        elif isinstance(x, torch.Tensor):
            same = x.shape == y.shape and torch.equal(x, y)
            if not same:
                ok = False
                report[path] = float((x.float() - y.float()).abs().max()) if x.shape == y.shape else 'shape'
        elif x != y and not (isinstance(x, float) and x != x and y != y):
            ok = False
            report[path] = [x, y]

    walk(a, b, 'dump')
    summary = {'identical': ok, 'differences': report, 'reference': str(paths[0]), 'candidate': str(paths[1])}
    Path(output).write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    return ok


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    d = sub.add_parser('dump')
    d.add_argument('--src', required=True)
    d.add_argument('--data-dir', required=True)
    d.add_argument('--output', required=True)
    c = sub.add_parser('compare')
    c.add_argument('reference')
    c.add_argument('candidate')
    c.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'dump':
        dump(args.src, args.data_dir, args.output)
        return 0
    return 0 if compare((args.reference, args.candidate), args.output) else 1


if __name__ == '__main__':
    raise SystemExit(main())
