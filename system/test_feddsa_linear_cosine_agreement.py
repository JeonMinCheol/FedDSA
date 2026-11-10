import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from flcore.clients.clientfeddsa import clientfeddsa
from flcore.trainmodel.models import build_model_from_spec


class LinearCosineAgreementTest(unittest.TestCase):
    def model(self, enabled=True):
        args = SimpleNamespace(algorithm='FedDSA', feature_dim=3, num_classes=3,
                               dataset='Cifar100', shared_classifier_scale=5.,
                               experimental_linear_cosine_agreement=enabled)
        return build_model_from_spec(
            'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600)', args)

    def test_linear_prediction_and_cosine_agreement_share_one_head(self):
        model = self.model()
        self.assertEqual(model.shared_classifier_mode, 'linear')
        self.assertIsNotNone(model.shared_head.bias)
        with torch.no_grad():
            model.shared_head.weight.copy_(torch.tensor([[10., 0., 0.], [1., 2., 0.], [0., 0., 1.]]))
            model.shared_head.bias.copy_(torch.tensor([0., 0., 20.]))
        z = torch.tensor([[1., 1., 0.], [1., 0., 1.]])
        linear = F.linear(z, model.shared_head.weight, model.shared_head.bias)
        cosine = F.normalize(z, dim=1) @ F.normalize(model.shared_head.weight, dim=1).T
        torch.testing.assert_close(model.classify(z), linear)
        torch.testing.assert_close(model.agreement_logits(z), cosine)
        self.assertFalse(torch.equal(linear.argmax(1), cosine.argmax(1)))
        torch.testing.assert_close(model.agreement_logits(3*z), cosine)
        torch.testing.assert_close(model.classify(3*z), 3*linear-2*model.shared_head.bias)
        self.assertEqual([n for n, _ in model.named_parameters() if 'head' in n],
                         ['shared_head.weight', 'shared_head.bias'])
        with torch.no_grad():
            x = torch.randn(2, 3, 32, 32)
            torch.testing.assert_close(model(x), model.shared_head(model.shared_representation(x)))
        default = self.model(False)
        self.assertIsNone(default.shared_head.bias)
        torch.testing.assert_close(default.classify(z), 5*default.agreement_logits(z))

    def test_training_sends_linear_logits_to_ce_and_cosine_logits_to_agreement(self):
        torch.manual_seed(17)
        model = self.model()
        x, y = torch.randn(3, 3, 32, 32), torch.arange(3)
        with torch.no_grad():
            z = model.shared_representation(x)
            expected_task, expected_agreement = model.classify(z), model.agreement_logits(z)
        seen = {}
        client = clientfeddsa.__new__(clientfeddsa)
        client.__dict__.update(model=model, device=torch.device('cpu'), learning_rate=.01,
            amp_enabled=False, agreement_weighting=True, local_epochs=1, train_slow=False,
            prototype_utility=False, meta_enabled=False, validation_enabled=False,
            latest_utility_metrics={}, protos={}, role='Client_0', save_folder_name='unit_test',
            train_time_cost={'num_rounds': 0, 'total_cost': 0.})
        client.update_gap_stats = lambda: None
        client.load_train_data = lambda **kw: [(x, y)]
        client.move_input = lambda x, y: (x, y)
        client.forward_rep_logits = lambda m, x: (m.shared_representation(x), m(x))
        client.collect_protos = lambda **kw: None
        client.release_model_from_device = lambda m: None
        def task(logits, labels):
            seen['task'] = logits.detach().clone()
            return F.cross_entropy(logits, labels)
        def anchor(z, labels, logits):
            seen['agreement'] = logits.detach().clone()
            self.assertFalse(logits.requires_grad)
            return .01*z.square().mean(), {}
        client.loss, client.anchor_loss = task, anchor
        with patch('flcore.clients.clientfeddsa.save_item'):
            client.train()
        torch.testing.assert_close(seen['task'], expected_task)
        torch.testing.assert_close(seen['agreement'], expected_agreement)

    def test_utility_matches_independent_linear_sgd_forks_without_mutating_model(self):
        torch.manual_seed(31)
        model = self.model()
        batches = [(torch.randn(6, 3, 32, 32), torch.tensor([0, 1, 2, 0, 1, 2])) for _ in range(3)]
        client = clientfeddsa.__new__(clientfeddsa)
        client.__dict__.update(prototype_utility=True, semantic_anchors=torch.eye(3),
            utility_validation_probe=True, device=torch.device('cpu'), utility_inner_lr=.01,
            lamda=1., mcl=1., num_classes=3)
        client.load_train_data = lambda **kw: batches
        client.move_input = lambda x, y: (x, y)
        initial = copy.deepcopy(model.state_dict())
        expected = [[], []]
        for use_anchor in [False, True]:
            branch = copy.deepcopy(model).eval()
            params = list(branch.shared_projector.parameters()) + list(branch.shared_head.parameters())
            optimizer = torch.optim.SGD(params, lr=.01)
            with torch.no_grad():
                features = branch.extract_backbone(batches[0][0])
            z = branch.shared_projector(features)
            labels = batches[0][1]
            loss = F.cross_entropy(F.linear(z, branch.shared_head.weight, branch.shared_head.bias), labels)
            if use_anchor:
                loss = loss + F.mse_loss(z, client.semantic_anchors[labels])
                loss = loss + F.cross_entropy(-torch.cdist(z, client.semantic_anchors), labels)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                for index in range(2):
                    x, y = batches[index+1]
                    logits = F.linear(branch.shared_representation(x), branch.shared_head.weight, branch.shared_head.bias)
                    expected[index].append(F.cross_entropy(logits, y, reduction='none'))
        model.train()
        client.compute_anchor_utility(model)
        self.assertTrue(model.training)
        for k, value in initial.items():
            torch.testing.assert_close(model.state_dict()[k], value)
        for index, actual in enumerate([client.anchor_utility, client.validation_anchor_gain]):
            gain = expected[index][0]-expected[index][1]
            means = torch.stack([gain[batches[index+1][1] == c].mean() for c in range(3)])
            torch.testing.assert_close(actual, means, atol=2e-6, rtol=1e-3)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
