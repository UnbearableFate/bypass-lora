"""E-0032 rule 1: segment-1 write masks, the sentence-end rule and the segmented X tap."""
from contextlib import contextmanager
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from bypass_model import BranchModel
from rb_tap_check import check_direct_hidden
from seg_data import sentence_ends, segment1_write, _TOKEN_TEXT
from seg_model import (make_model, forward, first_forward, parameter_digest, PARAMETERS, KEEP,
                       SEG05_INITIAL_SHA256, WRITE_RULE, X_TAP, E0032_ARMS, contract, config_for)


# ---- the frozen sentence-end rule on decoded token texts ----

class FakeTokenizer:
    """Each id decodes to a fixed text, so the rule is tested independently of BPE merges."""
    name_or_path = 'fake-sentence-rule'

    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        assert not skip_special_tokens and not clean_up_tokenization_spaces
        return ''.join(self.texts[i] for i in ids)


def flags(texts):
    _TOKEN_TEXT.clear()
    return sentence_ends(FakeTokenizer(texts), list(range(len(texts))))


@pytest.mark.parametrize('texts, expected', [
    (['Passage', ':', ' It', ' ran', '.', ' Then', ' stop', '?', ' Yes', '!', '\n'],
     [False, False, False, False, True, False, False, True, False, True, False]),
    # The next token must start with whitespace (newline counts); end of segment counts.
    # Whitespace inside the mark's own token does not count (literal frozen rule).
    (['A', '.', 'B', ' c', '.\n', 'D', '3', '.', '5', ' kg', '.', '\n'],
     [False, False, False, False, False, False, False, False, False, False, True, False]),
    # Closing quotes / right parentheses may follow inside the same token text.
    (['He', ' said', ' "', 'Stop', '."', ' Then', ' (', 'ok', '.)', ' x', '!\'', ' y', '?”', ' z'],
     [False, False, False, False, True, False, False, False, True, False, True, False, True, False]),
    # A quote token alone is not an end; trailing whitespace inside the token is ignored.
    (['Stop', '.', '"', ' Then', '. ', 'x', ' e', '.g', '.', ' a', ',', ' b', ';', '. ', ' f', '.'],
     [False, False, False, False, False, False, False, False, True, False, False, False, False, True, False, True]),
    # Opening quotes or other symbols after the mark do not qualify.
    (['a', '.“', ' b', '.]', ' c', '..', ' d'],
     [False, False, False, False, False, True, False]),
])
def test_sentence_end_rule_fixed_examples(texts, expected):
    assert flags(texts) == expected


def test_write_masks_always_include_the_last_token():
    tok = FakeTokenizer(['a', '.', ' b', ' c'])
    _TOKEN_TEXT.clear()
    assert segment1_write(tok, [0, 1, 2, 3], 'all') is None
    assert segment1_write(tok, [0, 1, 2, 3], 'last') == [False, False, False, True]
    assert segment1_write(tok, [0, 1, 2, 3], 'sentence_last') == [False, True, False, True]
    assert segment1_write(tok, [0, 1], 'sentence_last') == [False, True]
    with pytest.raises(ValueError):
        segment1_write(tok, [], 'last')
    with pytest.raises(ValueError):
        segment1_write(tok, [0], 'first')


def test_real_tokenizer_sentence_ends_follow_the_text():
    from seg_train import tokenizer
    tok = tokenizer()
    ids = tok('Passage: The cat sat. It ran away! Did it? Pi is 3.14 today\n', add_special_tokens=False)['input_ids']
    marks = sentence_ends(tok, ids)
    prefixes = [tok.decode(ids[:t+1]) for t, m in enumerate(marks) if m]
    assert prefixes == ['Passage: The cat sat.', 'Passage: The cat sat. It ran away!',
                        'Passage: The cat sat. It ran away! Did it?']
    assert segment1_write(tok, ids, 'last') == [False]*(len(ids)-1)+[True]


# ---- registration ----

def test_arm_registration_contract_and_configs():
    assert all(contract(s, a) == 'E-0032@1' for s in (0, 1, 2) for a in E0032_ARMS)
    assert contract(0) == 'E-0025@1' and contract(1, 'SEG-05') == 'E-0027@1'
    assert WRITE_RULE == {'SEG-W1': 'last', 'SEG-W2': 'sentence_last'}
    assert {a: PARAMETERS[a] for a in E0032_ARMS} == {'SEG-W1': 2752512, 'SEG-W2': 2752512,
        'SEG-X5': 2228224, 'SEG-X6': 2228224, 'SEG-X7': 2228224}
    assert config_for('SEG-W1') == config_for('SEG-W2') == config_for('SEG-05')
    from dataclasses import replace
    for arm, reference in zip(X_TAP, ('SEG-05', 'SEG-06', 'SEG-07')):
        assert config_for(arm) == replace(config_for(reference), tap='direct_hidden')
    assert config_for('SEG-05').tap == 'qknorm_pre_rope'


# ---- GPU ----

@pytest.fixture(scope='module')
def base():
    from calibrate import load_base
    return load_base()


@contextmanager
def model_for(base, arm, seed=0):
    with patch('seg_model.load_base', return_value=base):
        model = make_model(arm, seed=seed)
    try:
        yield model
    finally:
        for i, layer in model.adapters().items():
            for h in layer.handles: h.remove()
            base.model.layers[i] = layer.original
        base.zero_grad(set_to_none=True)


def ids():
    return (torch.tensor([[11, 29, 73, 181, 8, 64, 5]], device='cuda'),
            torch.tensor([[71, 9, 18, 41]], device='cuda'))


def perturb(model):
    generator = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for p in model.adapter_parameters():
            p.add_(torch.randn(p.shape, generator=generator).to(p.device)*.03)


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_initial_parameters_W_equal_SEG05_and_X_triplet_elementwise(base, seed):
    for arm in ('SEG-05', 'SEG-W1', 'SEG-W2'):
        with model_for(base, arm, seed) as model:
            assert parameter_digest(model) == SEG05_INITIAL_SHA256[seed]
    states = {}
    for arm in X_TAP:
        with model_for(base, arm, seed) as model:
            states[arm] = [p.detach().cpu().clone() for p in model.adapter_parameters()]
            assert sum(p.numel() for p in states[arm]) == PARAMETERS[arm]
            assert model.config.tap == 'direct_hidden' and model.config.rank == 128
            for layer in model.adapters().values():
                assert torch.count_nonzero(layer.side.fusion.P) == 0
    assert all(len(states[a]) == len(states['SEG-X5']) and
               all(torch.equal(x, y) for x, y in zip(states[a], states['SEG-X5'])) for a in X_TAP)


def test_X_tap_reads_the_input_layernorm_output(base):
    with model_for(base, 'SEG-X5') as model, torch.no_grad():
        assert check_direct_hidden(model, ids()[0])


def losses_and_grads(model, arm, a, b, first_write):
    model.zero_grad(set_to_none=True)
    out, parts = forward(model, arm, a, torch.ones_like(a), b, torch.ones_like(b), first_write=first_write)
    loss = F.cross_entropy(out.logits[0, :-1].float(), b[0, 1:])
    loss.backward()
    return out.logits.detach().clone(), [p.grad.detach().clone() for p in model.adapter_parameters()], parts


def test_all_true_segment1_mask_is_bitwise_the_unmasked_path(base):
    a, b = ids()
    with model_for(base, 'SEG-05') as model:
        model.train(); perturb(model)
        logits, grads, parts = losses_and_grads(model, 'SEG-05', a, b, None)
        masked, masked_grads, masked_parts = losses_and_grads(
            model, 'SEG-05', a, b, torch.ones_like(a, dtype=torch.bool))
        assert torch.equal(logits, masked)
        assert all(torch.equal(x, y) for x, y in zip(grads, masked_grads))
        for i in model.config.layers:
            assert int(parts['segment1'][i]['side']['writes']) == int(masked_parts['segment1'][i]['side']['writes']) == a.shape[1]


@pytest.mark.parametrize('arm, mask', [('SEG-W1', [0, 0, 0, 0, 0, 0, 1]), ('SEG-W2', [0, 0, 1, 0, 1, 0, 1])])
def test_masked_segment1_writes_only_the_mask_and_segment2_is_unchanged(base, arm, mask):
    a, b = ids()
    write = torch.tensor([mask], dtype=torch.bool, device='cuda')
    with model_for(base, arm) as model, torch.no_grad():
        model.eval(); perturb(model)
        with pytest.raises(ValueError):
            forward(model, arm, a, torch.ones_like(a), b, torch.ones_like(b))
        out, parts = forward(model, arm, a, torch.ones_like(a), b, torch.ones_like(b), first_write=write)
        first = first_forward(model, a, torch.ones_like(a), write)
        for i in model.config.layers:
            side1, side2 = parts['segment1'][i]['side'], parts['segment2'][i]['side']
            assert int(side1['writes']) == sum(mask) and int(side1['reads']) == a.shape[1]
            assert int(side2['writes']) == int(side2['reads']) == b.shape[1]
            assert int(first.cache.memory[i].counts[0]) == sum(mask)
            assert int(out.cache.memory[i].counts[0]) == sum(mask)+b.shape[1]
        # Segment 2 reads the carried masked state exactly as SEG-05 reads its state.
        direct = model(b, attention_mask=torch.ones_like(b), initial_memory=first.cache.memory, use_cache=False)
        assert torch.equal(direct.logits, out.logits)
        # Segment-1 positions before the first write read an empty state: zero increment.
        captured = {}
        layer = model.adapters()[model.config.layers[0]]
        handle = layer.side.register_forward_hook(lambda m, args, output: captured.setdefault('z', output[0]))
        try:
            first_forward(model, a, torch.ones_like(a), write)
        finally:
            handle.remove()
        before = mask.index(1)
        assert torch.count_nonzero(captured['z'][0, :before]) == 0
        assert torch.isfinite(captured['z']).all()


def test_W_swap_and_reset_interventions_with_masks(base):
    a, b = ids()
    write = torch.tensor([[0, 0, 0, 0, 0, 0, 1]], dtype=torch.bool, device='cuda')
    with model_for(base, 'SEG-W1') as model, torch.no_grad():
        model.eval(); perturb(model)
        normal, _ = forward(model, 'SEG-W1', a, torch.ones_like(a), b, torch.ones_like(b), first_write=write)
        donor = first_forward(model, a, torch.ones_like(a), write).cache.memory
        swapped, _ = forward(model, 'SEG-W1', a, torch.ones_like(a), b, torch.ones_like(b), kind='iv_swap',
                             donor_memory=donor, first_write=write)
        assert torch.equal(swapped.logits, normal.logits)
        reset, _ = forward(model, 'SEG-W1', a, torch.ones_like(a), b, torch.ones_like(b), kind='iv_reset',
                           first_write=write)
        assert torch.equal(reset.logits, model(b, use_cache=False).logits)


def test_F_phase_write_mask_validation(base):
    a, _ = ids()
    with model_for(base, 'SEG-X7') as model, torch.no_grad():
        with pytest.raises(ValueError):
            model(a, write_mask=torch.ones_like(a, dtype=torch.bool))
    with model_for(base, 'SEG-05') as model, torch.no_grad():
        with pytest.raises(ValueError):
            model(a, write_mask=torch.ones_like(a))
        with pytest.raises(ValueError):
            model(a, attention_mask=torch.tensor([[1, 1, 1, 1, 1, 1, 0]], device='cuda'),
                  write_mask=torch.ones_like(a, dtype=torch.bool))
        with pytest.raises(ValueError):
            model(a, write_mask=torch.ones_like(a, dtype=torch.bool), capture_history=True)
        b = ids()[1]
        with pytest.raises(ValueError):
            forward(model, 'SEG-06', a, torch.ones_like(a), b, torch.ones_like(b),
                    first_write=torch.ones_like(a, dtype=torch.bool))


@pytest.mark.parametrize('arm', E0032_ARMS)
def test_new_arms_start_at_base_and_iv_off_is_base(base, arm):
    a, b = ids()
    write = None if arm not in WRITE_RULE else torch.tensor([[0, 0, 1, 0, 0, 0, 1]], dtype=torch.bool, device='cuda')
    with torch.no_grad():
        expected = base(b, use_cache=False).logits
    with model_for(base, arm) as model, torch.no_grad():
        model.eval()
        initial, _ = forward(model, arm, a, torch.ones_like(a), b, torch.ones_like(b), first_write=write)
        assert torch.equal(initial.logits, expected)
        perturb(model)
        off, _ = forward(model, arm, a, torch.ones_like(a), b, torch.ones_like(b), kind='iv_off', first_write=write)
        assert torch.equal(off.logits, expected)


@pytest.mark.parametrize('arm', ['SEG-X6', 'SEG-X7'])
def test_X_reset_and_T0_carry_no_segment1_state(base, arm):
    a, b = ids()
    with model_for(base, arm) as model, torch.no_grad():
        model.eval(); perturb(model)
        observed = []
        h = model.model.register_forward_pre_hook(lambda m, args: observed.append(1))
        try:
            direct, parts = forward(model, arm, a, torch.ones_like(a), b, torch.ones_like(b))
        finally:
            h.remove()
        assert len(observed) == 1 and set(parts) == {'segment2'}
        discarded, _ = forward(model, arm, a, torch.ones_like(a), b, torch.ones_like(b), force_first=True)
        assert torch.equal(direct.logits, discarded.logits)
        assert torch.equal(direct.logits, model(b, use_cache=False).logits)
        if arm == 'SEG-X6':
            # Reset clears the state at the boundary: segment 2 starts from empty counts.
            assert all(int(s.counts[0]) == b.shape[1] for s in direct.cache.memory.values())
        with pytest.raises(ValueError):
            forward(model, arm, a, torch.ones_like(a), b, torch.ones_like(b), kind='iv_swap',
                    donor_memory={})


def test_X_keep_training_graph_reaches_segment1(base):
    a, b = ids()
    with model_for(base, 'SEG-X5') as model:
        model.train(); perturb(model)
        _, grads, _ = losses_and_grads(model, 'SEG-X5', a, b, None)
        assert all(torch.isfinite(g).all() and g.norm() > 0 for g in grads)
