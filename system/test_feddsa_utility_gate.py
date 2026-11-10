import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from flcore.clients.clientfeddsa import clientfeddsa
from flcore.servers.serverfeddsa import FedDSA


class UtilityGateTest(unittest.TestCase):
    def setUp(self):
        self.client = clientfeddsa.__new__(clientfeddsa)
        self.client.__dict__.update(
            num_classes=3,
            semantic_anchors=torch.eye(3),
            anchor_mode="single",
            new_anchor_mode="none",
            dual_disagreement_gate=False,
            reliability_anchor_source="mcl",
            scalar_reliability_state=False,
            agreement_weighting=False,
            prototype_utility=True,
            utility_local_gate=True,
            utility_weight_normalize=False,
            utility_positive_only=False,
            utility_temperature=0.1,
            utility_min_weight=0.25,
            utility_max_weight=1.5,
            class_utility=torch.zeros(3),
            lamda=1.0,
            mcl=1.0,
            local_gap=0.0,
            global_gap=0.0,
            mcl_gap_mode="min",
            mcl_variant="fedsa",
        )
        self.feats = torch.full((3, 3), 0.25, requires_grad=True)
        self.labels = torch.arange(3)
        self.logits = torch.eye(3)

    def anchor_loss(self):
        return self.client.anchor_loss(self.feats, self.labels, self.logits)

    def test_zero_utility_preserves_loss_and_gradient(self):
        self.client.utility_local_gate = False
        baseline, baseline_metrics = self.anchor_loss()
        baseline_grad = torch.autograd.grad(baseline, self.feats)[0]
        self.client.utility_local_gate = True
        for normalize in (False, True):
            with self.subTest(normalize=normalize):
                self.client.utility_weight_normalize = normalize
                loss, metrics = self.anchor_loss()
                torch.testing.assert_close(loss, baseline)
                for key in ("anchor_mse_loss", "mcl_loss"):
                    torch.testing.assert_close(metrics[key], baseline_metrics[key])
                torch.testing.assert_close(
                    torch.autograd.grad(loss, self.feats)[0], baseline_grad
                )
                self.assertAlmostEqual(metrics["utility_weight_mean"].item(), 1.0)

    def test_signed_utility_changes_loss_strength_with_bounds(self):
        neutral, _ = self.anchor_loss()
        losses = []
        for utility in (-100.0, -0.1, 0.0, 0.1, 100.0):
            self.client.class_utility.fill_(utility)
            loss, metrics = self.anchor_loss()
            self.assertTrue(torch.isfinite(loss).item())
            losses.append(loss.item())
            weight = metrics["utility_weight_mean"].item()
            self.assertGreaterEqual(weight, self.client.utility_min_weight)
            self.assertLessEqual(weight, self.client.utility_max_weight)
        self.assertEqual(losses, sorted(losses))
        self.assertLess(losses[1], neutral.item())
        self.assertGreater(losses[3], neutral.item())
        self.assertAlmostEqual(losses[0] / losses[2], 0.25)
        self.assertAlmostEqual(losses[-1] / losses[2], 1.5)

    def test_normalized_mean_does_not_hide_class_differences(self):
        self.client.utility_weight_normalize = True
        self.client.class_utility = torch.tensor([-0.1, 0.0, 0.1])
        _, metrics = self.anchor_loss()
        self.assertAlmostEqual(metrics["utility_weight_mean"].item(), 1.0)
        self.assertGreater(metrics["utility_weight_std"].item(), 0.1)

    def test_capture_records_clipping_without_changing_loss_or_gradient(self):
        self.client.class_utility = torch.tensor([-0.2, 0.0, 0.2])
        baseline, _ = self.anchor_loss()
        baseline_gradient = torch.autograd.grad(baseline, self.feats)[0]
        rng = torch.random.get_rng_state().clone()
        self.client.capture_utility_gate = True
        captured_loss, _ = self.anchor_loss()
        captured_gradient = torch.autograd.grad(captured_loss, self.feats)[0]
        captured = self.client.last_local_utility_gate
        torch.testing.assert_close(captured_loss, baseline)
        torch.testing.assert_close(captured_gradient, baseline_gradient)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertFalse(self.client.capture_utility_gate)
        torch.testing.assert_close(captured["weight"], torch.tensor([0.25, 1.0, 1.5]))
        self.assertLess(captured["before_clip"][0], 0.25)
        self.assertGreater(captured["before_clip"][2], 1.5)
        self.assertIsNone(captured["normalized_before_clip"])

        self.client.utility_weight_normalize = True
        self.client.capture_utility_gate = True
        self.anchor_loss()
        captured = self.client.last_local_utility_gate
        pre_normalization = (2.0 * torch.sigmoid(self.client.class_utility / 0.1)).clamp(0.25, 1.5)
        normalized = pre_normalization / pre_normalization.mean()
        torch.testing.assert_close(captured["before_normalization"], pre_normalization)
        torch.testing.assert_close(captured["normalized_before_clip"], normalized)
        torch.testing.assert_close(captured["weight"], normalized.clamp(0.25, 1.5))

    def test_diagnostics_preserve_class_ids_missing_values_and_round_changes(self):
        self.client.class_utility = torch.tensor([-0.2, 0.0, 0.2])
        self.client.capture_utility_gate = True
        self.anchor_loss()
        self.client.id = 7
        server = FedDSA.__new__(FedDSA)
        server.__dict__.update(
            num_classes=3, random_seed=0,
            args=SimpleNamespace(goal="test", utility_weight_normalize=False),
            previous_utility_gate_vectors={}, utility_temperature=0.1,
            utility_signal_mode="observed", utility_ema=0.25,
            utility_local_gate=True,
            utility_positive_only=False, utility_min_weight=0.25, utility_max_weight=1.5,
            last_utility_observation_count=torch.tensor([4.0, 0.0, 2.0]),
            last_utility_observed=torch.tensor([0.1, 0.0, float("nan")]),
            last_utility_signal=torch.tensor([0.1, 0.0, 0.2]),
            last_utility_client_count=torch.tensor([2.0, 0.0, 1.0]),
            last_utility_client_std=torch.tensor([0.2, 0.0, 0.0]),
            last_utility_positive_clients=torch.tensor([1.0, 0.0, 1.0]),
            last_utility_negative_clients=torch.tensor([1.0, 0.0, 0.0]),
            class_utility=torch.tensor([0.1, 0.0, 0.2]),
            utility_validation_probe=True,
            utility_validation_metrics={"pearson": 0.6, "spearman": 0.5, "pairs": 2},
            utility_validation_query_count=torch.tensor([2.0, 0.0, 1.0]),
            utility_validation_gain_count=torch.tensor([3.0, 0.0, 1.0]),
            utility_validation_query_value=torch.tensor([0.3, 0.0, 0.5]),
            utility_validation_gain_value=torch.tensor([0.4, 0.0, -0.1]),
            eval_gap=10, global_rounds=500,
        )
        with tempfile.TemporaryDirectory() as tmp:
            server.utility_gate_log_path = str(Path(tmp) / "gates.jsonl")
            server.write_utility_gate_diagnostics(1, self.client, torch.tensor([0.1, 0.0, 0.2]))
            server.write_utility_gate_diagnostics(2, self.client, torch.tensor([0.2, 0.0, 0.3]))
            server.write_utility_gate_diagnostics(3, None, None)
            rows = [json.loads(line) for line in Path(server.utility_gate_log_path).read_text().splitlines()]

        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["local"]["captured_client"], 7)
        self.assertEqual(rows[0]["local"]["first_clipping"]["below_min"], 1)
        self.assertEqual(rows[0]["local"]["first_clipping"]["above_max"], 1)
        self.assertIsNone(rows[0]["local"]["normalization_clipping"])
        self.assertNotIn("server", rows[0])
        self.assertEqual(rows[0]["anchor_candidate_mix"], 0.5)
        self.assertIsNone(rows[0]["utility"]["raw_observed"][1])
        self.assertIsNone(rows[0]["utility"]["raw_observed"][2])
        self.assertEqual(rows[0]["utility"]["contributing_clients"], [2.0, 0.0, 1.0])
        self.assertIsNone(rows[0]["utility"]["client_sample_weighted_std"][1])
        self.assertEqual(rows[0]["probe"]["paired_validation_count"], [3.0, 0.0, 1.0])
        self.assertEqual(rows[0]["probe"]["summary"]["pairs"], 2)
        self.assertIsNone(rows[0]["probe"]["validation_gain"][1])
        self.assertNotIn("server_gate_by_gain", rows[0]["probe"])
        self.assertAlmostEqual(rows[0]["probe"]["local_weight_by_gain"]["nonnegative_minus_negative"], -1.25, places=6)
        self.assertEqual(rows[1]["local"]["summary"]["change"]["mean_abs"], 0.0)
        self.assertIsNone(rows[2]["local"]["summary"])

    def test_observed_utility_logs_client_support_and_disagreement(self):
        server = FedDSA.__new__(FedDSA)
        server.__dict__.update(
            num_classes=3, feat_dim=3, device="cpu", prototype_utility=True,
            utility_validation_probe=False, utility_signal_mode="observed", utility_ema=0.25,
            class_utility=torch.zeros(3), new_anchor_metrics={}, role="Server",
            save_folder_name="temp_gate_test", aggregate_protos=lambda uploaded: {},
            selected_clients=[
                SimpleNamespace(protos={}, anchor_utility=torch.tensor([0.2, -0.1, 123.0]),
                                anchor_utility_count=torch.tensor([1.0, 3.0, 0.0])),
                SimpleNamespace(protos={}, anchor_utility=torch.tensor([-0.2, 0.1, 1.2]),
                                anchor_utility_count=torch.tensor([3.0, 1.0, 2.0])),
            ],
        )
        with patch("flcore.servers.serverfeddsa.save_item"):
            server.receive_protos()
        self.assertNotIn("harmful_intervention_suppression", server.utility_validation_metrics)
        torch.testing.assert_close(server.last_utility_observed, torch.tensor([-0.1, -0.05, 1.2]))
        torch.testing.assert_close(server.last_utility_observation_count, torch.tensor([4.0, 4.0, 2.0]))
        torch.testing.assert_close(server.last_utility_client_count, torch.tensor([2, 2, 1]))
        torch.testing.assert_close(server.last_utility_client_std, torch.tensor([0.03 ** 0.5, 0.0075 ** 0.5, 0.0]))
        torch.testing.assert_close(server.last_utility_positive_clients, torch.tensor([1, 1, 1]))
        torch.testing.assert_close(server.last_utility_negative_clients, torch.tensor([1, 1, 0]))


if __name__ == "__main__":
    unittest.main()
