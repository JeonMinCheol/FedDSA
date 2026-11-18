import os
import sys
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flcore.clients.clientetfproto import clientETFProto, simplex_etf  # noqa: E402
from flcore.servers.serveretfproto import ETFProtoFL  # noqa: E402


def amcl(p, y, labels, protos, margin=0.5, tau=0.07, num_classes=5):
    fake = SimpleNamespace(num_classes=num_classes, device=torch.device("cpu"), margin=margin, current_margin=margin, tau=tau)
    return clientETFProto.amcl_loss(fake, p, y, labels, protos)


class ETFProtoFLTest(unittest.TestCase):
    def test_simplex_geometry(self):
        etf = simplex_etf(10, 10)
        self.assertTrue(torch.allclose(etf.norm(dim=0), torch.ones(10), atol=1e-6))
        cos = etf.t() @ etf
        off = cos[~torch.eye(10, dtype=torch.bool)]
        self.assertTrue(torch.allclose(off, torch.full_like(off, -1 / 9), atol=1e-6))

    def test_amcl_matches_paper_formula(self):
        torch.manual_seed(0)
        p, protos = torch.randn(4, 3), torch.randn(5, 3)
        labels, y = torch.arange(5), torch.tensor([0, 1, 2, 4])
        sim = 0.5 * (1 + F.normalize(p, dim=1) @ F.normalize(protos, dim=1).t())
        logits = sim / 0.07
        logits[torch.arange(4), y] -= 0.5 / 0.07
        self.assertTrue(torch.allclose(amcl(p, y, labels, protos), F.cross_entropy(logits, y), atol=1e-5))
        self.assertGreater(amcl(p, y, labels, protos), amcl(p, y, labels, protos, margin=0.0))

    def test_amcl_skips_classes_without_global_prototype(self):
        torch.manual_seed(1)
        p, protos = torch.randn(3, 3), torch.randn(2, 3)
        labels = torch.tensor([1, 3])  # classes 0 and 4 have no global prototype yet
        full = amcl(p[1:2], torch.tensor([3]), labels, protos)
        self.assertTrue(torch.allclose(amcl(p, torch.tensor([0, 3, 4]), labels, protos), amcl(p[1:2], torch.tensor([3]), labels, protos)))
        self.assertTrue(torch.isfinite(full))
        self.assertEqual(float(amcl(p, torch.tensor([0, 0, 4]), labels, protos)), 0.0)

    def test_local_angular_gap(self):
        fake = SimpleNamespace(protos={0: torch.tensor([1.0, 0.0]), 1: torch.tensor([0.0, 2.0]), 2: torch.tensor([-1.0, 0.0])})
        # nearest neighbours are all orthogonal (cos 0) -> (1 - 0) / 2 = 0.5
        self.assertAlmostEqual(clientETFProto.local_angular_gap(fake), 0.5, places=6)
        self.assertIsNone(clientETFProto.local_angular_gap(SimpleNamespace(protos={0: torch.ones(2)})))
        shifted = SimpleNamespace(protos={c: v + torch.tensor([50.0, 50.0]) for c, v in fake.protos.items()})
        # a large shared offset hides the separation unless prototypes are centred first
        self.assertLess(clientETFProto.local_angular_gap(shifted), 0.01)
        self.assertGreater(clientETFProto.local_angular_gap(shifted, centered=True), 0.2)

    def test_exclusion_keeps_classes_only_excluded_clients_hold(self):
        cl = [SimpleNamespace(id=0, protos={0: torch.tensor([1., 0.])}, proto_counts={0: 2.}),
              SimpleNamespace(id=1, protos={0: torch.tensor([9., 9.]), 2: torch.tensor([3., 3.])}, proto_counts={0: 5., 2: 4.})]
        srv = SimpleNamespace(exclude={1}, agg_weight="count", selected_clients=cl, clients=cl)
        ETFProtoFL.receive_protos(srv)
        self.assertTrue(torch.equal(srv.global_protos[0], torch.tensor([1., 0.])))
        self.assertTrue(torch.equal(srv.global_protos[2], torch.tensor([3., 3.])))

    def test_gap_weighted_aggregation(self):
        cl = [SimpleNamespace(id=0, protos={0: torch.tensor([1., 0.])}, proto_counts={0: 1.}, upload_gap=0.3),
              SimpleNamespace(id=1, protos={0: torch.tensor([0., 1.])}, proto_counts={0: 1.}, upload_gap=0.1)]
        srv = SimpleNamespace(exclude=set(), agg_weight="gap", selected_clients=cl, clients=cl)
        ETFProtoFL.receive_protos(srv)
        self.assertTrue(torch.allclose(srv.global_protos[0], torch.tensor([0.75, 0.25])))


if __name__ == "__main__":
    unittest.main()
