import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from flcore.servers.serverfeddsa import FedDSA


class NumericalGuardTest(unittest.TestCase):
    def server(self):
        server = FedDSA.__new__(FedDSA)
        client = SimpleNamespace(id=7, protos={}, anchor_utility=torch.tensor([.1, .2, .3]),
                                 anchor_utility_count=torch.ones(3))
        server.__dict__.update(num_classes=3, feat_dim=3, device="cpu", prototype_utility=True,
                               utility_validation_probe=False, utility_signal_mode="zscore", utility_ema=.25,
                               class_utility=torch.zeros(3), new_anchor_metrics={}, role="Server",
                               current_round=245, save_folder_name="temp_numeric_test",
                               aggregate_protos=lambda uploads: {}, selected_clients=[client])
        return server, client

    def receive(self, server):
        with patch("flcore.servers.serverfeddsa.save_item"):
            server.receive_protos()

    def test_finite_update_is_unchanged(self):
        server, client = self.server()
        expected = (client.anchor_utility - client.anchor_utility.mean()) / client.anchor_utility.std(unbiased=False) * .25
        self.receive(server)
        torch.testing.assert_close(server.class_utility, expected)

    def test_bad_upload_does_not_poison_state(self):
        for name in ("anchor_utility", "anchor_utility_count", "validation_anchor_gain", "validation_anchor_gain_count"):
            with self.subTest(name=name):
                server, client = self.server()
                client.validation_anchor_gain = torch.ones(3)
                client.validation_anchor_gain_count = torch.ones(3)
                setattr(client, name, torch.tensor([.1, float("nan"), .3]))
                with self.assertRaisesRegex(FloatingPointError, "round=245, client=7"):
                    self.receive(server)
                torch.testing.assert_close(server.class_utility, torch.zeros(3))

    def test_nonfinite_prototype_is_rejected(self):
        server, client = self.server()
        client.protos = {0: torch.tensor([1., float("inf"), 0.])}
        with self.assertRaisesRegex(FloatingPointError, r"prototype\[0\]"):
            self.receive(server)

    def test_finite_inputs_with_overflowing_sum_are_rejected(self):
        server, client = self.server()
        client.anchor_utility = torch.full((3,), 3e38)
        server.selected_clients.append(SimpleNamespace(id=8, protos={}, anchor_utility=client.anchor_utility,
                                                       anchor_utility_count=torch.ones(3)))
        with self.assertRaisesRegex(FloatingPointError, "weighted_utility_sum"):
            self.receive(server)
        torch.testing.assert_close(server.class_utility, torch.zeros(3))


if __name__ == "__main__":
    unittest.main()
