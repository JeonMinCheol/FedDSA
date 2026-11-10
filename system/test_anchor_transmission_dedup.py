import unittest
from types import SimpleNamespace

import torch

from flcore.clients.clientsa import clientSA
from flcore.servers.serverfeddsa import FedDSA
from flcore.servers.serversa import FedSA
from test_feddsa_utility_gate import UtilityGateTest


class AnchorTransmissionTest(unittest.TestCase):
    def dsa_lifecycle(self, duplicate, partial):
        server = FedDSA.__new__(FedDSA)
        server.__dict__.update(
            semantic_anchors=torch.eye(3), stable_semantic_anchors=torch.eye(3),
            class_utility=torch.zeros(3), new_anchor_mode='none', anchor_mode='single',
            num_classes=3, feat_dim=3, scalar_reliability_state=False,
            prototype_utility=True, clients=[],
        )
        sent = []
        for cid in range(4):
            fixture = UtilityGateTest()
            fixture.setUp()
            client = fixture.client
            client.device = 'cpu'
            client.id = cid
            client.agreement_weighting = True
            client.utility_temperature = .25
            client.utility_weight_normalize = True
            for name in ['set_semantic_anchors', 'set_class_utility']:
                original = getattr(client, name)
                def setter(value, _original=original, _name=name):
                    sent.append((_name, value.numel()*value.element_size()))
                    _original(value)
                setattr(client, name, setter)
            server.clients.append(client)
        server.send_semantic_state()
        observations = []
        for round_index in range(6):
            selected = server.clients[round_index % 2::2] if partial else server.clients
            # Evaluation runs before the optional legacy resend.
            observations.extend((c.semantic_anchors.clone(), c.class_utility.clone())
                                for c in server.clients)
            if duplicate:
                server.send_semantic_state(selected)
            for client in selected:
                features = (torch.arange(9).reshape(3, 3).float()/10 + round_index/100).requires_grad_()
                loss, _ = client.anchor_loss(features, torch.arange(3), torch.eye(3))
                gradient = torch.autograd.grad(loss, features)[0]
                observations.append((loss.detach().clone(), gradient.clone()))
            server.semantic_anchors.add_(.01*(round_index+1))
            server.class_utility.add_(torch.tensor([-.01, .005, .01]))
            server.send_semantic_state()
        return observations, sent

    def test_full_and_partial_participation_keep_states_losses_and_gradients(self):
        for partial in [False, True]:
            with self.subTest(partial=partial):
                old, old_sent = self.dsa_lifecycle(True, partial)
                new, new_sent = self.dsa_lifecycle(False, partial)
                self.assertEqual(len(old), len(new))
                for left, right in zip(old, new):
                    for a, b in zip(left, right):
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
                self.assertEqual(len(new_sent), (1+6)*4*2)
                self.assertEqual(len(old_sent)-len(new_sent), 6*(2 if partial else 4)*2)

    def test_feddsa_sends_full_anchors_and_utility_even_when_unchanged(self):
        sent = []
        server = FedDSA.__new__(FedDSA)
        server.__dict__.update(
            semantic_anchors=torch.eye(3), stable_semantic_anchors=torch.eye(3),
            class_utility=torch.tensor([-.1, 0., .1]), new_anchor_mode='none',
            anchor_mode='single', num_classes=3, feat_dim=3,
            scalar_reliability_state=False, prototype_utility=True,
            clients=[SimpleNamespace(
                set_semantic_anchors=lambda x: sent.append(('anchors', x.clone())),
                set_stable_semantic_anchors=lambda x: None,
                set_class_utility=lambda x: sent.append(('utility', x.clone())),
            )],
        )
        for _ in range(2):
            server.send_semantic_state()
        self.assertEqual([kind for kind, _ in sent], ['anchors', 'utility'] * 2)
        self.assertEqual([x.numel() * x.element_size() for _, x in sent], [36, 12] * 2)
        for index in [0, 1]:
            torch.testing.assert_close(sent[index][1], sent[index + 2][1], rtol=0, atol=0)

    def sa_server(self):
        server = FedSA.__new__(FedSA)
        server.semantic_anchors = torch.eye(3)
        server.anchor_momentum = .9999
        server.global_protos = {c: torch.full((3,), 5.) for c in range(3)}
        server.clients = []
        counts = []
        def client():
            c = clientSA.__new__(clientSA)
            c.device = torch.device('cpu')
            original = c.set_semantic_anchors
            def receive(value):
                counts.append(value.numel()*value.element_size())
                original(value)
            c.set_semantic_anchors = receive
            return c
        server.clients = [client(), client()]
        return server, counts, client

    def test_fedsa_unchanged_anchor_is_sent_every_round(self):
        server, counts, _ = self.sa_server()
        server.anchor_momentum = 1.0
        server.send_semantic_anchors()
        initial = server.semantic_anchors.clone()
        for _ in range(3):
            server.update_semantic_anchors()
            server.send_semantic_anchors()
        self.assertEqual(len(counts), 8)
        torch.testing.assert_close(server.semantic_anchors, initial, rtol=0, atol=0)
        for c in server.clients:
            torch.testing.assert_close(c.semantic_anchors, initial, rtol=0, atol=0)

    def test_fedsa_official_ema_updates_observed_classes_and_broadcasts_changes(self):
        server, counts, _ = self.sa_server()
        del server.global_protos[1]
        initial = server.semantic_anchors.clone()
        server.send_semantic_anchors()
        server.update_semantic_anchors()
        expected = torch.tensor([[1.0004, .0005, .0005], [0., 1., 0.], [.0005, .0005, 1.0004]])
        torch.testing.assert_close(server.semantic_anchors, expected)
        torch.testing.assert_close(server.clients[0].semantic_anchors, initial, rtol=0, atol=0)
        for step in [1, 2]:
            if step == 2:
                server.update_semantic_anchors()
                expected[[0, 2]] = (.9999**2 * initial[[0, 2]] + (1 - .9999**2) * 5.)
                torch.testing.assert_close(server.semantic_anchors, expected)
            server.send_semantic_anchors()
            for c in server.clients:
                torch.testing.assert_close(c.semantic_anchors, server.semantic_anchors, rtol=0, atol=0)
        self.assertEqual(len(counts), 6)

    def test_fedsa_momentum_zero_replaces_only_observed_classes(self):
        server, _, _ = self.sa_server()
        server.anchor_momentum = 0.0
        server.global_protos = {2: torch.tensor([2., 3., 4.], dtype=torch.float64)}
        server.update_semantic_anchors()
        torch.testing.assert_close(server.semantic_anchors, torch.tensor([[1., 0., 0.], [0., 1., 0.], [2., 3., 4.]]), rtol=0, atol=0)
        self.assertFalse(server.semantic_anchors.requires_grad)

    def test_fedsa_inplace_and_replacement_updates_are_delivered_without_aliasing(self):
        server, counts, _ = self.sa_server()
        server.send_semantic_anchors()
        server.semantic_anchors.add_(.1)
        self.assertFalse(torch.equal(server.semantic_anchors, server.clients[0].semantic_anchors))
        server.send_semantic_anchors()
        server.semantic_anchors = server.semantic_anchors + .1
        server.send_semantic_anchors()
        self.assertEqual(len(counts), 6)
        for c in server.clients:
            torch.testing.assert_close(c.semantic_anchors, server.semantic_anchors, rtol=0, atol=0)

    def test_fedsa_new_recipients_receive_the_full_anchor(self):
        server, counts, new_client = self.sa_server()
        server.send_semantic_anchors()
        server.clients.append(new_client())
        server.send_semantic_anchors()
        self.assertEqual(len(counts), 5)
        torch.testing.assert_close(server.clients[-1].semantic_anchors,
                                   server.semantic_anchors, rtol=0, atol=0)

    def test_fedsa_training_uses_official_broadcast_schedule_with_evaluation(self):
        server, counts, _ = self.sa_server()
        for cid, client in enumerate(server.clients):
            client.id = cid
            client.test_metrics = lambda: (0, 1, 0.)
        server.__dict__.update(
            global_rounds=2, eval_gap=1, auto_break=False, Budget=[], rs_test_acc=[0.],
            select_clients=lambda: server.clients,
            run_clients=lambda clients, **kwargs: [],
            log_train_diagnostics=lambda *args: None,
            receive_protos=lambda: None,
            cal_class_margins=lambda *args: None,
            proto2anchors=lambda protos: torch.empty(0),
            maybe_save_tsne=lambda: None, save_results=lambda: None,
        )
        server.evaluate = server.test_metrics
        server.train()
        self.assertEqual(len(counts), (1 + 3) * len(server.clients))


if __name__ == '__main__':
    unittest.main()
