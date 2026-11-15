"""Server-only RAAR evidence must be independent of the private local CAU gate."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from flcore.servers.serverfeddsa import FedDSA
from utils.raar_head_evidence import head_gains
from utils.test_local_marginal_cau import local_fixture, server_fixture


def evidence_server(local_cau=False):
    server = server_fixture()
    for key, value in dict(raar_enabled=True, raar_evidence_enabled=True,
                           local_marginal_cau=local_cau, utility_validation_probe=False,
                           raar_evidence_ema=.25, raar_evidence_assignment="observed",
                           current_round=0, random_seed=2, role="server", save_folder_name="unused",
                           utility_trust_temperature=.001, utility_trust_penalty=.25,
                           fast_radius_rho=.6).items():
        setattr(server, key, value)
    server.new_anchor_metrics = {}
    server.selected_clients = [
        SimpleNamespace(id=0, protos={0: torch.zeros(2)}, proto_sample_count={0: 2},
                        raar_utility=torch.tensor([-.2, .4, 0.]),
                        raar_utility_count=torch.tensor([2., 4., 0.])),
        SimpleNamespace(id=1, protos={0: torch.ones(2)}, proto_sample_count={0: 6},
                        raar_utility=torch.tensor([-.6, .2, 0.]),
                        raar_utility_count=torch.tensor([6., 4., 0.])),
    ]
    return server


class RAAREvidenceTests(unittest.TestCase):
    def test_server_raw_evidence_ema_count_aggregation_and_current_missing_mask(self):
        for local_cau in (False, True):
            server = evidence_server(local_cau)
            with patch("flcore.servers.serverfeddsa.save_item"):
                server.receive_protos()
            torch.testing.assert_close(server.raar_utility_ema, torch.tensor([-.125, .075, 0.]))
            torch.testing.assert_close(server.raar_utility_count, torch.tensor([8., 8., 0.]))
            torch.testing.assert_close(server.global_protos[0], torch.tensor([.75, .75]))
            torch.testing.assert_close(server.negative_raar_utility(torch.ones(3)), torch.tensor([1., 0., 0.]))
            for client in server.selected_clients:
                client.raar_utility_count.zero_()
            with patch("flcore.servers.serverfeddsa.save_item"):
                server.receive_protos()
            self.assertTrue(torch.equal(server.raar_utility_count, torch.zeros(3)))
            self.assertTrue(torch.equal(server.negative_raar_utility(torch.ones(3)), torch.zeros(3)))
            self.assertFalse(hasattr(server, "class_utility"))

    def test_virtual_raar_evidence_ignores_local_gate_and_local_gap_selection(self):
        client, model, batches = local_fixture()
        client.candidate_gain_enabled = True
        client.protos = {0: torch.zeros(2), 1: torch.tensor([2., 0.]), 2: torch.tensor([0., 2.])}
        client.local_cau_state = {"weight": torch.tensor([.75, 1.25, 1.])}
        client.local_cau_round = client.current_round
        state = copy.deepcopy(model.state_dict())
        expected = None
        for enabled in (False, True):
            for gap in (False, True):
                client.local_marginal_cau, client.local_marginal_gap = enabled, gap
                current = head_gains(client, model, batches[0], batches[1], batches[2])
                if expected is None:
                    expected = current
                for (value, count), (v, n) in zip(current, expected):
                    self.assertTrue(torch.equal(value, v) and torch.equal(count, n))
                for key, value in model.state_dict().items():
                    self.assertTrue(torch.equal(value, state[key]))
                self.assertTrue(all(p.grad is None for p in model.parameters()))
                self.assertTrue(client.anchor_candidate_evidence["mcl_gap_class_cap"])
                self.assertEqual(client.anchor_candidate_evidence["mcl_gap"], .2)

    def test_local_cau_alone_does_not_run_or_upload_server_evidence(self):
        client, model, _ = local_fixture()
        client.candidate_gain_enabled = False
        with patch("utils.raar_head_evidence.head_gains", side_effect=AssertionError("RAAR disabled")):
            self.assertIsNone(client.compute_raar_evidence(model))
        self.assertIsNone(client.raar_utility)
        self.assertIsNone(client.raar_utility_count)


if __name__ == "__main__":
    unittest.main()
