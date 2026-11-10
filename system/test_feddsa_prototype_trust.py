import unittest

import torch

from flcore.servers.serverfeddsa import FedDSA


class PrototypeTrustTest(unittest.TestCase):
    def test_server_update_is_independent_of_class_utility_and_temperature(self):
        initial = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        expected = torch.tensor([[2.0, 4.0], [3.0, 4.0], [3.0, 5.0]])
        for utility in (-100.0, 0.0, 100.0):
            for temperature in (0.1, 0.25):
                with self.subTest(utility=utility, temperature=temperature):
                    server = FedDSA.__new__(FedDSA)
                    server.__dict__.update(
                        device="cpu", semantic_anchors=initial.clone(),
                        global_protos={0: torch.tensor([3.0, 6.0]),
                                       2: torch.tensor([1.0, 4.0]),
                                       -1: torch.zeros(2), 3: torch.zeros(2)},
                        prototype_utility=True, dual_disagreement_gate=False,
                        anchor_mode="single", new_anchor_mode="none", moving_anchor_ema=1.0,
                        class_utility=torch.tensor([utility, 0.0, -utility]),
                        utility_temperature=temperature,
                    )
                    server.update_semantic_anchors()
                    torch.testing.assert_close(server.semantic_anchors, expected)
                    self.assertFalse(server.semantic_anchors.requires_grad)

    def test_moving_single_retains_its_configured_update_rate(self):
        for rate in (0.0, 0.25, 1.0):
            with self.subTest(rate=rate):
                server = FedDSA.__new__(FedDSA)
                server.__dict__.update(
                    device="cpu", semantic_anchors=torch.tensor([[1.0], [10.0]]),
                    global_protos={0: torch.tensor([3.0])},
                    prototype_utility=False, dual_disagreement_gate=False,
                    anchor_mode="single", new_anchor_mode="none", moving_anchor_ema=rate,
                )
                server.update_semantic_anchors()
                torch.testing.assert_close(
                    server.semantic_anchors, torch.tensor([[1.0 + 2.0 * rate], [10.0]])
                )

    def test_empty_upload_keeps_anchors_unchanged(self):
        server = FedDSA.__new__(FedDSA)
        server.semantic_anchors = torch.eye(2)
        server.global_protos = {}
        server.update_semantic_anchors()
        torch.testing.assert_close(server.semantic_anchors, torch.eye(2))

    def test_utility_update_keeps_only_prototype_trust(self):
        for mode in ("auto", "off"):
            with self.subTest(mode=mode):
                server = FedDSA.__new__(FedDSA)
                server.__dict__.update(
                    device="cpu", semantic_anchors=torch.tensor([[1.0], [10.0]]),
                    stable_semantic_anchors=torch.tensor([[0.0], [9.0]]),
                    global_protos={0: torch.tensor([2.0])},
                    proto_disagreement={0: 1.0}, proto_support={0: 1.0},
                    fast_radius_rho=0.6, fast_anchor_proto_trust=mode,
                    dual_disagreement_gate=False, prototype_utility=True,
                    anchor_mode="single", new_anchor_mode="none", moving_anchor_ema=1.0,
                )
                server.update_semantic_anchors()
                self.assertAlmostEqual(server.semantic_anchors[0].item(), 1.5)
                server.apply_prototype_trust()
                beta = 4.0 / (4.0 + 0.36 + 1e-12) * 0.5 if mode == "auto" else 1.0
                torch.testing.assert_close(
                    server.semantic_anchors, torch.tensor([[1.5 * beta], [10.0]])
                )


if __name__ == "__main__":
    unittest.main()
