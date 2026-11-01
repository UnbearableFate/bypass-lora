"""V5 regressions. Run with the unchanged V1 suites in a registered PBS job."""
import hashlib
import json
import math
import random
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import v5_train as v5
from v1_data import derive_seed, build_training_stream
from v1_run import lr_factor, preserved_evaluation_state


def toy_splits():
    return {'splits': {task: {'train': [{'task':task, 'id':i} for i in range(n)]}
                       for task, n in [('piqa', 5), ('boolq', 3)]}}


def test_v5_order_independent_reference_and_filter_after_shuffle():
    splits = toy_splits()
    for seed in range(4):
        orders = []
        for epoch in range(3):
            rng = random.Random(derive_seed(20261029, 'concat_loop_v2', 'V5', 'train', seed)+epoch)
            p, b = list(range(5)), list(range(3))
            rng.shuffle(p)
            rng.shuffle(b)
            p.remove(2)
            expected = []
            for index in range(4):
                expected.append(('piqa', p[index]))
                if index < 3:
                    expected.append(('boolq', b[index]))
            actual = [(e['task'],e['id']) for e in v5.training_stream(splits, seed, epoch, [('piqa',2)])]
            assert actual == expected
            orders.append(actual)
        assert orders[0] != orders[1]
    assert v5.training_stream(splits, 0, 0) != build_training_stream(splits, 0)


def patch_encoder(monkeypatch):
    monkeypatch.setattr(v5, 'encode_training_example', lambda tok, row, max_length:
                        dict(row, input_ids=[1,2], labels=[-100,2]))


def test_three_epochs_cross_boundary_batch_and_monitor_points(monkeypatch):
    patch_encoder(monkeypatch)
    splits = toy_splits()
    batches = v5.plan_batches(splits, None, 3, [('piqa',2)])
    assert [len(b) for b in batches] == [2]*10+[1]
    assert [[e['epoch'] for e in b] for b in batches][3] == [0,1]
    meta = v5.plan_metadata(batches)
    assert meta['plan.steps'] == 11 and meta['plan.examples'] == 21
    assert meta['plan.epoch_end_steps'] == {'1':4,'2':7,'3':11}
    assert meta['plan.input_tokens'] == 42
    encoded_order = [(e['task'],e['id'],e['epoch']) for batch in batches for e in batch]
    assert meta['plan.order_sha256'] == hashlib.sha256(json.dumps(encoded_order).encode()).hexdigest()
    assert v5.plan_metadata(v5.plan_batches(splits,None,3,[('piqa',2)])) == meta
    capped = v5.plan_batches(splits, None, 3, [('piqa',2)], token_budget=5)
    assert len(capped) == 2 and v5.plan_metadata(capped)['plan.input_tokens'] == 8


def test_lr_selection_expands_raw_minimum_before_small_lr_tie_rule():
    candidates = {1e-4:1.009, 3e-4:1.005, 1e-3:1.0}
    assert v5.extension_lr(candidates) == 3e-3
    assert v5.select_lr(candidates) == 1e-4
    assert v5.select_lr(dict(candidates, **{})) == 1e-4
    candidates[3e-3] = .98
    assert v5.select_lr(candidates) == 3e-3
    assert v5.extension_lr({1e-4:1.,3e-4:float('nan'),1e-3:None}) == 3e-5
    assert v5.extension_lr({1e-4:1.1,3e-4:1.,1e-3:float('inf')}) is None
    with pytest.raises(ValueError, match='no finite'):
        v5.select_lr({1e-4:float('inf'),3e-4:None})


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([.3,-.2,.7]))


def patch_training(monkeypatch):
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:0)
    monkeypatch.setattr(v5,'batch_tensors',lambda examples, pad:examples[0])
    def losses(model, row):
        labels = torch.tensor(row['targets'])
        logits = model.weight.expand(len(labels),-1)
        return F.cross_entropy(logits,labels,reduction='none'), None
    monkeypatch.setattr(v5,'token_losses',losses)


def test_example_mean_objective_matches_independent_optimizer_reference(monkeypatch,tmp_path):
    patch_training(monkeypatch)
    batches = [[{'input_ids':[1,2], 'targets':[0], 'epoch':0,'epoch_end':False},
                {'input_ids':[1,2,3,4], 'targets':[1,1,2], 'epoch':0,'epoch_end':True}],
               [{'input_ids':[1,2,3], 'targets':[2,0], 'epoch':1,'epoch_end':True}],
               [{'input_ids':[1,2], 'targets':[1], 'epoch':2,'epoch_end':True}]]
    actual, reference = TinyModel(), TinyModel()
    optimizer = torch.optim.AdamW(reference.parameters(),lr=.01,betas=(.9,.999),eps=1e-8,weight_decay=.01)
    for step,batch in enumerate(batches):
        optimizer.zero_grad(set_to_none=True)
        example_losses = [F.cross_entropy(reference.weight.expand(len(row['targets']),-1),
                                          torch.tensor(row['targets']),reduction='mean') for row in batch]
        torch.stack(example_losses).mean().backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(),1.)
        optimizer.param_groups[0]['lr'] = .01*lr_factor(step,len(batches))
        optimizer.step()
    monitor_points = []
    def monitor(epoch,step,model):
        monitor_points.append((epoch,step))
        model.eval()
        random.random()
        torch.rand(2)
    result = v5.train(actual,batches,SimpleNamespace(pad_token_id=0),.01,tmp_path,monitor)
    torch.testing.assert_close(actual.weight, reference.weight,rtol=0,atol=0)
    assert monitor_points == [(1,1),(2,2),(3,3)]
    assert actual.training
    assert result['train.steps'] == 3 and result['train.examples'] == 4
    assert result['train.loss_tokens'] == 7 and result['train.input_tokens'] == 11


def test_nonfinite_training_stops_before_optimizer_and_retains_usage(monkeypatch,tmp_path):
    patch_training(monkeypatch)
    monkeypatch.setattr(v5,'token_losses',lambda model,row:(torch.tensor([float('nan')]),None))
    model = TinyModel()
    before = model.weight.detach().clone()
    row = {'input_ids':[1,2], 'targets':[0], 'epoch':0,'epoch_end':False}
    with pytest.raises(v5.TrainingDiverged):
        v5.train(model,[[row]],SimpleNamespace(pad_token_id=0),.01,tmp_path,None)
    assert torch.equal(before,model.weight)
    usage = json.loads((tmp_path/'usage.json').read_text())
    assert usage['train.steps'] == 0 and usage['train.input_tokens'] == 2


@pytest.mark.parametrize('arm',tuple(v5.ARMS))
def test_real_v5_arm_initialization_counts_seed_and_checkpoint(arm,tmp_path):
    from calibrate import fixture
    ids = fixture()
    base = v5.make_model('base',0)
    with torch.no_grad():
        reference = base(ids,attention_mask=torch.ones_like(ids),use_cache=False).logits
    del base
    torch.cuda.empty_cache()
    model = v5.make_model(arm,2)
    params = list(model.adapter_parameters())
    assert sum(p.numel() for p in params) == v5.ARMS[arm]
    assert all(p.dtype == torch.float32 and p.requires_grad for p in params)
    with torch.no_grad():
        initial = model(ids,attention_mask=torch.ones_like(ids),use_cache=False).logits
    assert torch.equal(initial,reference)
    assert all(not p.requires_grad for p in model.model.parameters()
               if not any(p is adapter for adapter in params))
    # A real update makes the save/reload test sensitive to missing adapter weights.
    model.train()
    loss = model(ids,attention_mask=torch.ones_like(ids),use_cache=False).logits.float().square().mean()
    loss.backward()
    with torch.no_grad():
        for p in params:
            assert p.grad is not None and torch.isfinite(p.grad).all()
            p.add_(p.grad,alpha=-.001)
    model.eval()
    with torch.no_grad():
        expected = model(ids,attention_mask=torch.ones_like(ids),use_cache=False).logits
    model.export_adapter(tmp_path)
    del model,params
    torch.cuda.empty_cache()
    reloaded = v5.make_model(arm,3,tmp_path)
    with torch.no_grad():
        observed = reloaded(ids,attention_mask=torch.ones_like(ids),use_cache=False).logits
    assert torch.equal(expected,observed)


def test_lock_requires_endpoint_and_real_calibration_manifests(tmp_path):
    manifests = []
    for arm in v5.ARMS:
        for lr,nll in [(1e-4,1.3),(3e-4,1.2),(1e-3,1.),(3e-3,1.005)]:
            path = tmp_path/f'{arm}-{lr}.json'
            v5.write_json(path,{'status':'complete','validator':{'passed':True},
                                'phase':'calibration','contract':v5.CONTRACT,'seed':0,'arm':arm,
                                'metrics':{'learning_rate':lr,'calibration.nll':nll}})
            manifests.append(path)
    args = SimpleNamespace(manifests=manifests,output=tmp_path/'lock.json')
    v5.lock_lrs(args)
    lock = json.loads(args.output.read_text())
    assert lock['locked_lrs'] == dict.fromkeys(v5.ARMS,1e-3)
    args.manifests = manifests[:-1]
    with pytest.raises(ValueError,match='required candidates'):
        v5.lock_lrs(args)
    bad = json.loads(manifests[0].read_text())
    bad['phase'] = 'smoke'
    v5.write_json(manifests[0],bad)
    args.manifests = manifests
    with pytest.raises(ValueError,match='invalid calibration'):
        v5.lock_lrs(args)
