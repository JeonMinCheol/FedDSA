"""Private local CAU: derivatives, fresh-state lifecycle and sampler isolation."""
import copy
import os
import pickle
import random
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F
from utils.feddsa_utils import configure_utility_module
from utils.local_marginal_cau import local_marginal_gate
def client_fixture():
    from flcore.clients.clientfeddsa import clientfeddsa
    client = clientfeddsa.__new__(clientfeddsa)
    for key, value in dict(device=torch.device("cpu"), num_classes=3,
                           new_anchor_mode="none", anchor_mode="single",
                           dual_disagreement_gate=False, reliability_anchor_source="mcl",
                           scalar_reliability_state=False, agreement_weighting=True,
                           capture_utility_gate=True, mcl_variant="fedsa",
                           mcl_gap_class_cap=True, local_gap=.2, global_gap=.3,
                           mcl_gap_mode="min", local_cau_state=None, local_cau_round=None).items():
        setattr(client, key, value)
    return client


def server_fixture():
    from flcore.servers.serverfeddsa import FedDSA
    server = FedDSA.__new__(FedDSA)
    for key, value in dict(args=SimpleNamespace(local_marginal_cau=True),
                           device=torch.device("cpu"), num_classes=3, feat_dim=2,
                           new_anchor_mode="none", anchor_mode="single",
                           scalar_reliability_state=False).items():
        setattr(server, key, value)
    server.semantic_anchors = torch.tensor([[0., 0.], [2., 0.], [0., 2.]])
    server.stable_semantic_anchors = server.semantic_anchors.clone()
    return server


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_projector = torch.nn.Linear(4, 2)
        self.shared_head = torch.nn.Linear(2, 3)
        self.backbone = torch.nn.Dropout(.5)

    def extract_backbone(self, value):
        return self.backbone(value)

    def agreement_logits(self, features):
        return 5 * F.linear(F.normalize(features, dim=1), F.normalize(self.shared_head.weight, dim=1))


def local_fixture(gap=True):
    client = client_fixture()
    for key, value in dict(local_marginal_cau=True, local_marginal_gap=gap,
                           pcw_prediction_source="classifier_logits", utility_inner_lr=.01,
                           lamda=1., mcl=1., id=0, random_seed=7, current_round=10,
                           batch_size=4, save_folder_name="local_marginal_test").items():
        setattr(client, key, value)
    client.semantic_anchors = torch.tensor([[0., 0.], [2., 0.], [0., 2.]])
    client.stable_semantic_anchors = None
    generator = torch.Generator().manual_seed(51)
    batches = [(torch.randn(4, 4, generator=generator), torch.tensor([0, 1, 2, 0])) for _ in range(3)]
    client.load_train_data = lambda drop_last=False, **kwargs: iter(batches)
    return client, ToyModel(), batches


def rng_state():
    return torch.get_rng_state().clone(), pickle.dumps(np.random.get_state()), pickle.dumps(random.getstate())


class LocalMarginalTests(unittest.TestCase):
    def test_options_and_invalid_combinations(self):
        args = SimpleNamespace(local_marginal_cau=True, fast_anchor_proto_trust="off")
        configure_utility_module(args)
        configure_utility_module(args)
        self.assertTrue(args.local_marginal_cau)
        self.assertFalse(args.raar_enabled)
        for removed in ("prototype_utility", "cau_gate_policy", "utility_local_gate"):
            self.assertFalse(hasattr(args, removed))
        for override in (dict(use_amp=True),
                         dict(new_anchor_mode="boundary"), dict(scalar_reliability_state=True)):
            with self.assertRaises(ValueError):
                configure_utility_module(SimpleNamespace(local_marginal_cau=True, **override))

    def test_source_class_derivative_with_pcw_gap_and_missing_class(self):
        from utils.mcl_gap import add_mcl_gap
        generator = torch.Generator().manual_seed(34)
        for gap in (0., 4.):
            for cap in (False, True):
                projector = torch.nn.Linear(4, 2, dtype=torch.float64)
                head = torch.nn.Linear(2, 3, dtype=torch.float64)
                x = torch.randn(7, 4, generator=generator, dtype=torch.float64)
                y = torch.tensor([0, 0, 1, 1, 0, 1, 0])
                q = torch.randn(5, 4, generator=generator, dtype=torch.float64)
                qy = torch.tensor([0, 1, 2, 0, 1])
                anchors = torch.randn(3, 2, generator=generator, dtype=torch.float64)
                pcw = torch.linspace(.3, .9, 7, dtype=torch.float64)
                params = [projector.weight, projector.bias, head.weight, head.bias]
                w = torch.ones(3, dtype=torch.float64, requires_grad=True)
                z = F.linear(x, params[0], params[1])
                ce = F.cross_entropy(F.linear(z, params[2], params[3]), y)
                dist = (z[:, None] - anchors[None]).square().sum(-1).sqrt()
                dist = add_mcl_gap(dist, y, anchors, gap, class_cap=cap)
                per = F.mse_loss(z, anchors[y], reduction="none").mean(1)
                per = per + F.cross_entropy(-dist, y, reduction="none")
                grads = torch.autograd.grad(ce + (per * pcw * w[y]).mean(), params, create_graph=True)
                update = [p - .01 * grad for p, grad in zip(params, grads)]
                loss = F.cross_entropy(F.linear(F.linear(q, update[0], update[1]), update[2], update[3]), qy)
                expected = -torch.autograd.grad(loss, w)[0]
                actual = local_marginal_gate(x, y, q, qy, projector, head, anchors, pcw, .01, 1., 1., gap, cap)
                torch.testing.assert_close(actual["score"], expected, atol=1e-12, rtol=1e-9)
                self.assertEqual(float(actual["weight"][2]), 1.)
                self.assertFalse(bool(actual["observed"][2]))
                self.assertTrue(bool(((actual["weight"] >= .75) & (actual["weight"] <= 1.25)).all()))

    def test_current_private_gate_preserves_model_rng_and_resets_after_absence(self):
        for gap in (False, True):
            client, model, batches = local_fixture(gap)
            model.train()
            model.backbone.eval()
            for parameter in model.parameters():
                parameter.grad = torch.ones_like(parameter)
            state = copy.deepcopy(model.state_dict())
            gradients = [p.grad.clone() for p in model.parameters()]
            modes = [m.training for m in model.modules()]
            client.raar_utility = torch.tensor([.1, .2, .3])
            client.anchor_candidate_evidence = {"sentinel": True}
            rng = rng_state()
            with tempfile.TemporaryDirectory() as temp:
                cwd = os.getcwd()
                os.chdir(temp)
                try:
                    client.prepare_local_marginal_gate(model)
                    gate = client.local_cau_state["weight"].clone()
                    self.assertFalse(hasattr(client, "set_raw_cau_gate"))
                    self.assertFalse(hasattr(client, "set_class_utility"))
                    record = client.latest_local_marginal_record
                    self.assertEqual(record["gap_in_signal"], gap)
                    self.assertTrue(record["independent_gate_check"]["validation_gap_included"])
                    for name, value in model.state_dict().items():
                        self.assertTrue(torch.equal(value, state[name]))
                    for p, grad in zip(model.parameters(), gradients):
                        self.assertTrue(torch.equal(p.grad, grad))
                    self.assertEqual([m.training for m in model.modules()], modes)
                    current = rng_state()
                    self.assertTrue(torch.equal(rng[0], current[0]))
                    self.assertEqual(rng[1:], current[1:])
                    self.assertEqual(client.anchor_candidate_evidence, {"sentinel": True})
                    self.assertTrue(torch.equal(client.raar_utility, torch.tensor([.1, .2, .3])))
                    client.current_round = 200
                    client.load_train_data = lambda **kwargs: iter(batches[:1])
                    client.prepare_local_marginal_gate(model)
                    self.assertTrue(torch.equal(client.local_cau_state["weight"], torch.ones(3)))
                finally:
                    os.chdir(cwd)

    def test_sampler_and_server_never_overwrite_local_gate_or_training_order(self):
        client, model, batches = local_fixture()
        client._read_client_data_cached = lambda **kwargs: [(x, y) for bx, by in batches for x, y in zip(bx, by)]
        del client.load_train_data
        before = list(client.load_train_data(phase="train/0", drop_last=False))
        with tempfile.TemporaryDirectory() as temp:
            cwd = os.getcwd()
            os.chdir(temp)
            try:
                client.prepare_local_marginal_gate(model)
            finally:
                os.chdir(cwd)
        after = list(client.load_train_data(phase="train/0", drop_last=False))
        for (x, y), (a, b) in zip(before, after):
            self.assertTrue(torch.equal(x, a) and torch.equal(y, b))
        server = server_fixture()
        server.args.local_marginal_cau = True
        server.clients = [client]
        weights = client.local_cau_state["weight"].clone()
        server.send_semantic_state()
        self.assertTrue(torch.equal(weights, client.local_cau_state["weight"]))

    def test_probe_uses_the_selected_actual_training_pcw_score_source(self):
        for source in ("classifier_logits", "cosine"):
            client, model, batches = local_fixture()
            client.pcw_prediction_source = source
            with torch.no_grad():
                z = model.shared_projector(batches[0][0])
                expected = (model.agreement_logits(z) if source == "cosine"
                            else model.shared_head(z))
            with tempfile.TemporaryDirectory() as temp:
                cwd = os.getcwd()
                os.chdir(temp)
                try:
                    with patch.object(client, "anchor_agreement_weight",
                                      wraps=client.anchor_agreement_weight) as agreement:
                        client.prepare_local_marginal_gate(model)
                    self.assertTrue(torch.equal(agreement.call_args.args[3], expected))
                finally:
                    os.chdir(cwd)


if __name__ == "__main__":
    unittest.main()
