import inspect
import unittest

import torch
import torch.nn.functional as F

from v2r16_objective import slot_marginal


class SlotMarginalTests(unittest.TestCase):
    def test_matches_finite_probability_enumeration(self):
        log_weights = torch.log(torch.tensor([[0.25, 0.75], [0.6, 0.4]]))
        logits = torch.tensor([
            [[2.0, -1.0, 0.5], [-0.5, 1.0, 0.0]],
            [[0.0, 0.2, 1.1], [1.4, -0.3, 0.1]],
        ])
        labels = torch.tensor([0, 2])
        nll, log_mix = slot_marginal(log_weights, logits, labels)

        probs = F.softmax(logits, dim=-1)
        enumerated = (log_weights.exp()[..., None] * probs).sum(dim=1)
        self.assertTrue(torch.allclose(log_mix.exp(), enumerated, atol=1e-7))
        self.assertTrue(torch.allclose(nll, -enumerated[torch.arange(2), labels].log().mean()))

    def test_mixes_probabilities_not_logits(self):
        weights = torch.log(torch.tensor([[0.5, 0.5]]))
        logits = torch.tensor([[[8.0, 0.0], [0.0, 2.0]]])
        _, log_mix = slot_marginal(weights, logits, torch.tensor([0]))
        probability_mix = (weights.exp()[..., None] * logits.softmax(-1)).sum(1)
        logit_mix = (weights.exp()[..., None] * logits).sum(1).softmax(-1)
        self.assertTrue(torch.allclose(log_mix.exp(), probability_mix))
        self.assertFalse(torch.allclose(log_mix.exp(), logit_mix))

    def test_gradients_match_explicit_enumeration_and_reach_both_routes(self):
        log_weights = torch.log(torch.tensor([[0.35, 0.65]], requires_grad=True))
        log_weights.retain_grad()
        logits = torch.tensor([[[1.0, -0.4, 0.2], [-0.2, 0.7, 0.1]]], requires_grad=True)
        labels = torch.tensor([1])
        nll, _ = slot_marginal(log_weights, logits, labels)
        grad_w, grad_logits = torch.autograd.grad(nll, (log_weights, logits))

        explicit_w = log_weights.detach().clone().requires_grad_()
        explicit_logits = logits.detach().clone().requires_grad_()
        terms = explicit_w[..., None] + F.log_softmax(explicit_logits, dim=-1)
        explicit_nll = -torch.logsumexp(terms, dim=1)[0, labels[0]]
        ref_w, ref_logits = torch.autograd.grad(explicit_nll, (explicit_w, explicit_logits))
        self.assertTrue(torch.allclose(grad_w, ref_w))
        self.assertTrue(torch.allclose(grad_logits, ref_logits))
        self.assertGreater(grad_w.abs().sum().item(), 0.0)
        self.assertGreater(grad_logits.abs().sum().item(), 0.0)

    def test_extreme_log_space_values_remain_finite(self):
        log_weights = torch.tensor([[-10000.0, 0.0]])
        logits = torch.tensor([[[1.0e30, -1.0e30], [-1.0e30, 1.0e30]]])
        nll, log_mix = slot_marginal(log_weights, logits, torch.tensor([1]))
        self.assertTrue(torch.isfinite(nll))
        self.assertTrue(torch.isfinite(log_mix).all())
        self.assertTrue(torch.allclose(log_mix, torch.tensor([[-10000.0, 0.0]])))

    def test_invalid_slots_do_not_affect_result_even_with_all_negative_infinity(self):
        weights = torch.log(torch.tensor([[0.2, 0.3, 0.5]]))
        logits = torch.tensor([[[1.0, 0.0], [-torch.inf, -torch.inf], [0.0, 1.0]]])
        valid = torch.tensor([[True, False, True]])
        nll, log_mix = slot_marginal(weights, logits, torch.tensor([0]), valid)
        changed_logits = logits.clone()
        changed_logits[0, 1] = torch.tensor([torch.inf, -torch.inf])
        changed_nll, changed_log_mix = slot_marginal(weights, changed_logits, torch.tensor([0]), valid)
        expected = torch.logsumexp(
            weights[:, [0, 2], None] + F.log_softmax(logits[:, [0, 2], :], dim=-1), dim=1
        )
        self.assertTrue(torch.allclose(log_mix, expected))
        self.assertTrue(torch.allclose(nll, changed_nll))
        self.assertTrue(torch.allclose(log_mix, changed_log_mix))

    def test_rejects_empty_rows_and_malformed_shapes(self):
        weights = torch.zeros(2, 3)
        logits = torch.zeros(2, 3, 4)
        labels = torch.tensor([0, 1])
        with self.assertRaises(ValueError):
            slot_marginal(weights, logits, labels, torch.tensor([[True, False, False], [False] * 3]))
        with self.assertRaises(ValueError):
            slot_marginal(weights[:, :2], logits, labels)

    def test_api_has_no_target_slot_or_auxiliary_label_argument(self):
        self.assertEqual(tuple(inspect.signature(slot_marginal).parameters),
                         ("log_weights", "slot_logits", "labels", "slot_valid"))


if __name__ == "__main__":
    unittest.main()
