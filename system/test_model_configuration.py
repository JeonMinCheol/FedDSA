import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from flcore.trainmodel.models import build_model_from_spec
from flcore.trainmodel.models import BaseHeadSplit
from test_feddsa_config_cleanup import current_parser


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x):
        return self.fc(x)


class ModelConfigurationTest(unittest.TestCase):
    def test_cli_has_only_linear_and_no_anchor_mode(self):
        parser, _ = current_parser()
        args = parser.parse_args([])
        self.assertEqual(args.shared_classifier_scale, 5.0)
        for option in ['--head_mode', '--projector_mode']:
            self.assertNotIn(option, parser._option_string_actions)
        self.assertNotIn('--anchor_mode', parser._option_string_actions)
        self.assertFalse(args.use_amp)

    def test_feddsa_real_builder_and_logits_are_cosine(self):
        args = SimpleNamespace(feature_dim=5, num_classes=3, algorithm='FedDSA',
                               dataset='Cifar100', shared_classifier_scale=5.0)
        model = build_model_from_spec('TinyBackbone()', args, {'TinyBackbone': TinyBackbone})
        self.assertEqual(model.shared_classifier_mode, 'cosine')
        self.assertIsInstance(model.shared_projector, nn.Linear)
        self.assertIsInstance(model.shared_head, nn.Linear)
        self.assertIsNone(model.shared_head.bias)
        x = torch.randn(6, 4)
        features = model.shared_representation(x)
        expected = 5.0 * (torch.nn.functional.normalize(features, dim=1)
                          @ torch.nn.functional.normalize(model.shared_head.weight, dim=1).t())
        torch.testing.assert_close(model(x), expected)
        torch.testing.assert_close(model.classify(2 * features), expected)
        model.shared_classifier_scale = 100.0
        torch.testing.assert_close(model(x), 20 * expected)

    def test_baseline_default_heads_are_linear(self):
        for algorithm in ['LG-FedAvg', 'FedGH', 'FedProto', 'FedTGP', 'FedRG', 'FedSA']:
            with self.subTest(algorithm=algorithm):
                args = SimpleNamespace(algorithm=algorithm, feature_dim=4, num_classes=3,
                                       models=['tiny'])
                with patch('flcore.trainmodel.models.build_model_from_spec', return_value=TinyBackbone()):
                    model = BaseHeadSplit(args)
                self.assertIsInstance(model.head, nn.Linear)
                self.assertIsNotNone(model.head.bias)
                features = torch.randn(6, 4)
                torch.testing.assert_close(model.head(2 * features),
                                           2 * model.head(features) - model.head.bias)


if __name__ == '__main__':
    unittest.main()
