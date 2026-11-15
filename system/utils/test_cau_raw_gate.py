"""Run with: python -m unittest utils.test_cau_raw_gate (from system/)."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from utils.cau_raw_gate import raw_cau_payload, raw_cau_weights
from utils.feddsa_utils import configure_utility_module


class RawGateTests(unittest.TestCase):
    def test_options_preserve_legacy_and_resolve_raw_policy(self):
        legacy = SimpleNamespace(fast_anchor_proto_trust="off", utility_temperature=.25,
                                 utility_weight_normalize=True, utility_min_weight=.2,
                                 utility_max_weight=1.6, utility_positive_only=True)
        configure_utility_module(legacy)
        self.assertEqual(legacy.cau_gate_policy, "legacy")
        self.assertEqual(legacy.cau_gate_tau, .25)
        self.assertTrue(legacy.utility_weight_normalize)
        self.assertEqual(legacy.utility_min_weight, .2)
        self.assertTrue(legacy.utility_positive_only)
        # Preserve the legacy client's existing temperature clamping behavior.
        for temperature in (0., -1.):
            args = SimpleNamespace(fast_anchor_proto_trust="off", utility_temperature=temperature)
            configure_utility_module(args)
            configure_utility_module(args)  # Main, server and clients resolve the same args.
        for policy in ("raw", "raw_shuffled"):
            raw = SimpleNamespace(fast_anchor_proto_trust="off", cau_gate_policy=policy,
                                  utility_temperature=.25, cau_gate_tau=.1)
            configure_utility_module(raw)
            self.assertFalse(raw.utility_weight_normalize)
            self.assertFalse(raw.utility_positive_only)
            self.assertEqual((raw.utility_min_weight, raw.utility_max_weight), (.75, 1.25))
            self.assertEqual(raw.cau_gate_tau, .1)
        for tau in (0., -1., float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                configure_utility_module(SimpleNamespace(cau_gate_tau=tau))

    def test_sign_neutrality_bounds_and_uncertainty(self):
        for device in ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",):
            raw = torch.tensor([-10., -.1, 0., .1, 10.], device=device)
            noise = torch.full_like(raw, .02)
            count = torch.full_like(raw, 4.)
            weight, _ = raw_cau_weights(raw, noise, count, .1)
            self.assertTrue(bool((weight[raw < 0] <= 1).all()))
            self.assertTrue(bool((weight[raw > 0] >= 1).all()))
            self.assertEqual(float(weight[2]), 1.)
            self.assertTrue(bool(((weight >= .75) & (weight <= 1.25)).all()))
            neutral, _ = raw_cau_weights(raw, noise, torch.zeros_like(count), .1)
            self.assertTrue(torch.equal(neutral, torch.ones_like(raw)))
            noisy, _ = raw_cau_weights(raw, noise * 10, count, .1)
            self.assertTrue(bool(((noisy - 1).abs() <= (weight - 1).abs()).all()))

    def test_shuffle_preserves_distribution_and_global_rng(self):
        server = SimpleNamespace(class_utility=torch.zeros(20),
                                 raw_class_utility=torch.linspace(-.2, .2, 20),
                                 raw_gate_noise_ema=torch.full((20,), .02),
                                 raw_gate_effective_clients=torch.full((20,), 4.),
                                 random_seed=2, current_round=37)
        server.raw_gate_effective_clients[0] = 0
        state = torch.get_rng_state().clone()
        normal = raw_cau_payload(server, "raw", .1)
        shuffled = raw_cau_payload(server, "raw_shuffled", .1)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertTrue(torch.equal(normal["weight"].sort().values,
                                    shuffled["weight"].sort().values))
        self.assertTrue(torch.equal(shuffled["weight"],
                                    normal["weight"][shuffled["source_class"]]))
        self.assertFalse(torch.equal(normal["weight"], shuffled["weight"]))
        self.assertEqual(int(shuffled["source_class"][0]), 0)
        again = raw_cau_payload(server, "raw_shuffled", .1)
        self.assertTrue(torch.equal(shuffled["weight"], again["weight"]))


def server_fixture(policy="raw"):
    from flcore.servers.serverfeddsa import FedDSA
    server = FedDSA.__new__(FedDSA)
    server.args = SimpleNamespace(cau_gate_policy=policy, cau_gate_tau=.1)
    server.cau_gate_policy = policy
    server.cau_gate_tau = .1
    server.device = torch.device("cpu")
    server.num_classes, server.feat_dim = 3, 2
    server.prototype_utility = server.utility_local_gate = True
    server.raar_enabled = server.utility_validation_probe = False
    server.utility_ema, server.utility_signal_mode = .25, "zscore"
    server.class_utility = torch.zeros(3)
    server.class_utility_count = torch.zeros(3)
    server.semantic_anchors = torch.tensor([[0., 0.], [2., 0.], [0., 2.]])
    server.stable_semantic_anchors = server.semantic_anchors.clone()
    server.new_anchor_mode, server.anchor_mode = "none", "single"
    server.scalar_reliability_state = False
    server.role, server.save_folder_name = "server", "unused"
    server.new_anchor_metrics = {}
    server.current_round, server.random_seed = 0, 2
    server.selected_clients = [
        SimpleNamespace(id=0, protos={0: torch.zeros(2)}, proto_sample_count={0: 2},
                        anchor_utility=torch.tensor([-.2, .4, 0.]),
                        anchor_utility_count=torch.tensor([2., 4., 0.])),
        SimpleNamespace(id=1, protos={0: torch.ones(2)}, proto_sample_count={0: 6},
                        anchor_utility=torch.tensor([-.6, .2, 0.]),
                        anchor_utility_count=torch.tensor([6., 4., 0.])),
    ]
    return server


def client_fixture():
    from flcore.clients.clientfeddsa import clientfeddsa
    client = clientfeddsa.__new__(clientfeddsa)
    state = dict(device=torch.device("cpu"), num_classes=3, new_anchor_mode="none",
                 anchor_mode="single", dual_disagreement_gate=False,
                 reliability_anchor_source="mcl", scalar_reliability_state=False,
                 prototype_utility=True, utility_local_gate=True, cau_gate_policy="raw",
                 raw_cau_gate_payload=None, utility_positive_only=False,
                 utility_weight_normalize=False, utility_min_weight=.75,
                 utility_max_weight=1.25, utility_temperature=.1,
                 agreement_weighting=True, capture_utility_gate=True,
                 mcl_variant="fedsa", mcl_gap_class_cap=True,
                 local_gap=.2, global_gap=.3, mcl_gap_mode="min")
    for key, value in state.items():
        setattr(client, key, value)
    return client


class GateIntegrationTests(unittest.TestCase):
    def test_first_broadcast_is_neutral_then_aggregation_drives_both_anchor_losses(self):
        server = server_fixture()
        client = client_fixture()
        server.clients = [client]
        server.send_semantic_state()
        self.assertTrue(torch.equal(client.raw_cau_gate_payload["weight"], torch.ones(3)))
        with patch("flcore.servers.serverfeddsa.save_item"):
            server.receive_protos()
        torch.testing.assert_close(server.raw_class_utility, torch.tensor([-.125, .075, 0.]))
        torch.testing.assert_close(server.raw_gate_effective_clients, torch.tensor([1.6, 2., 0.]))
        server.current_round = 1
        server.send_semantic_state()
        weight = client.raw_cau_gate_payload["weight"]
        self.assertLess(float(weight[0]), 1.)
        self.assertGreater(float(weight[1]), 1.)
        self.assertEqual(float(weight[2]), 1.)
        features = torch.tensor([[.1, 0.], [1.9, 0.], [.1, 1.8], [.1, 0.]], requires_grad=True)
        labels = torch.tensor([0, 0, 2, 1])
        logits = torch.tensor([[2., 0., 0.], [0., 2., 0.], [0., 0., 2.], [0., 2., 0.]])
        pcw, _ = client.anchor_agreement_weight(features, labels, server.semantic_anchors, logits)
        self.assertFalse(torch.equal(pcw, torch.ones_like(pcw)))
        actual, _ = client.anchor_loss(features, labels, logits)
        composed = pcw * weight[labels]
        expected = client.anchor_mse(features, labels, server.semantic_anchors, composed)
        expected += client.mcl_loss(features, labels, server.semantic_anchors, composed)[0]
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(bool(torch.isfinite(torch.autograd.grad(actual, features)[0]).all()))
        # A class with no current query support must not reuse stale gate support.
        for producer in server.selected_clients:
            producer.anchor_utility_count.zero_()
        with patch("flcore.servers.serverfeddsa.save_item"):
            server.receive_protos()
        server.send_semantic_state()
        self.assertTrue(torch.equal(client.raw_cau_gate_payload["weight"], torch.ones(3)))

    def test_disabled_cau_and_gate_bypass_payload_and_do_not_touch_raar_state(self):
        for disable in ("prototype_utility", "utility_local_gate"):
            server = server_fixture()
            client = client_fixture()
            server.clients = [client]
            server.send_semantic_state()
            self.assertIsNotNone(client.raw_cau_gate_payload)
            setattr(server, disable, False)
            anchor = server.semantic_anchors.clone()
            server.send_semantic_state()
            self.assertIsNone(client.raw_cau_gate_payload)
            self.assertTrue(torch.equal(server.semantic_anchors, anchor))


if __name__ == "__main__":
    unittest.main()
