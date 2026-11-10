import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from flcore.clients.clientfeddsa import clientfeddsa
from flcore.clients.clientsa import clientSA


class UnitAnchorLossTest(unittest.TestCase):
    def setUp(self):
        self.features = torch.full((3, 3), 0.25, requires_grad=True)
        self.labels = torch.arange(3)
        self.anchors = torch.eye(3)

    def test_feddsa_uses_both_losses_with_unit_coefficients(self):
        client = clientfeddsa.__new__(clientfeddsa)
        client.__dict__.update(
            num_classes=3, semantic_anchors=self.anchors,
            anchor_mode="single", new_anchor_mode="none",
            dual_disagreement_gate=False, reliability_anchor_source="mcl",
            scalar_reliability_state=False, agreement_weighting=False,
            prototype_utility=True, utility_local_gate=True,
            utility_weight_normalize=False, utility_positive_only=False,
            utility_temperature=0.1, utility_min_weight=0.25,
            utility_max_weight=1.5, class_utility=torch.zeros(3),
            local_gap=0.0, global_gap=0.0, mcl_gap_mode="min", mcl_variant="fedsa",
            lamda=0.0, mcl=0.0,
        )
        loss, metrics = client.anchor_loss(self.features, self.labels, torch.eye(3))
        mse = F.mse_loss(self.features, self.anchors)
        mcl = F.cross_entropy(-torch.cdist(self.features, self.anchors), self.labels)
        torch.testing.assert_close(loss, mse + mcl)
        torch.testing.assert_close(metrics["anchor_mse_loss"], mse)
        torch.testing.assert_close(metrics["mcl_loss"], mcl)
        torch.testing.assert_close(torch.autograd.grad(loss, self.features, retain_graph=True)[0],
                                   torch.autograd.grad(mse + mcl, self.features)[0])

    def test_feddsa_rejects_nonunit_shared_options(self):
        for mse_weight, mcl_weight in [(0.0, 1.0), (1.0, 2.0)]:
            with self.subTest(mse_weight=mse_weight, mcl_weight=mcl_weight):
                args = SimpleNamespace(lamda=mse_weight, margin_contrastive=mcl_weight)
                with self.assertRaisesRegex(ValueError, "unit MSE and MCL"):
                    clientfeddsa(args, id=0, train_samples=1, test_samples=1)

    def test_fedsa_still_respects_configured_coefficients(self):
        client = clientSA.__new__(clientSA)
        client.__dict__.update(semantic_anchors=self.anchors, num_classes=3,
                               cc=0.0, lamda=2.0, mcl=3.0,
                               loss_mse=nn.MSELoss(), loss=nn.CrossEntropyLoss(), local_gap=0.0, global_gap=0.0)
        loss, metrics = client.cal_anchors_loss(self.features, None, self.labels)
        mse = F.mse_loss(self.features, self.anchors)
        mcl = F.cross_entropy(-torch.cdist(self.features, self.anchors), self.labels)
        torch.testing.assert_close(loss, 2.0 * mse + 3.0 * mcl)
        torch.testing.assert_close(metrics["fedsa_mse_loss"], 2.0 * mse)
        torch.testing.assert_close(metrics["fedsa_mcl_loss"], 3.0 * mcl)


if __name__ == "__main__":
    unittest.main()
