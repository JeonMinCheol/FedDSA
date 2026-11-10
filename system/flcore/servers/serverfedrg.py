"""Port of FedRG/system/flcore/{servers/serverfedrg,optimizers/aggregator}.py."""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from flcore.clients.clientbase import save_item
from flcore.clients.clientfedrg import clientRG
from flcore.clients.clientprototype import optional_item
from flcore.servers.serverprototype import PrototypeServer


class GraphConstructor(nn.Module):
    def __init__(self, feature_dim, node_dim, alpha):
        super().__init__()
        self.lin1 = nn.Linear(feature_dim, node_dim)
        self.lin2 = nn.Linear(feature_dim, node_dim)
        self.alpha = alpha

    def forward(self, features):
        a = torch.tanh(self.alpha * self.lin1(features))
        b = torch.tanh(self.alpha * self.lin2(features))
        return F.relu(torch.tanh(self.alpha * (a @ b.T - b @ a.T)))


def rep_aggregate(args, prototypes, reliability_weights):
    features = torch.stack(prototypes).detach().to(args.device)
    size = len(prototypes)
    if args.rg_graph_target == 'uniform':
        # The official aggregator explicitly overwrites reliability with this matrix.
        weights = torch.ones(size, size, device=features.device) / size
    else:
        weights = reliability_weights.to(features.device)
    target = F.normalize(weights, dim=1)
    graph = GraphConstructor(features.shape[1], args.node_dim, args.adjalpha).to(args.device)
    optimizer = torch.optim.SGD(graph.parameters(), lr=args.gnn_learning_rate,
                                weight_decay=args.gnn_weight_decay)
    for _ in range(args.gc_epoch):
        loss = F.mse_loss(F.normalize(graph(features), dim=1), target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        adj = graph(features)
        row_sum = adj.sum(1, keepdim=True)
        adj = adj / row_sum.clamp_min(torch.finfo(adj.dtype).tiny)
        # Antisymmetric adjacency has a zero diagonal. Isolated nodes (including
        # a class on one client) retain their own representation instead of zero.
        adj = torch.where(row_sum > 0, adj, torch.eye(size, device=features.device))
        return (adj @ features).detach()


@torch.no_grad()
def calculate_performance(head, prototype, label):
    probabilities = head(prototype).softmax(-1)
    other = probabilities.clone()
    other[label] = 0
    p = probabilities[label]
    return (1 - (1 - p).clamp_min(0).sqrt()) * p * (1 - other.max())


class FedRG(PrototypeServer):
    def __init__(self, args, times):
        super().__init__(args, times)
        if args.gc_epoch < 1 or args.node_dim < 1:
            raise ValueError("gc_epoch and node_dim must be positive")
        self.set_slow_clients()
        self.set_clients(clientRG)
        self.head = optional_item(self.role, 'head', self.save_folder_name, self.device)
        if self.head is None:
            # Avoid aliasing client 0's classifier, which makes its updates special.
            self.head = copy.deepcopy(self.clients[0].load_model().head)
        self.optimizer_head = torch.optim.SGD(self.head.parameters(), lr=args.server_learning_rate)

    def run_round(self, round_index):
        self.train_selected_clients()
        active = self.active_clients()
        self.uploaded_protos = [optional_item(c.role, 'protos', self.save_folder_name, self.device)
                                for c in active]
        samples = [(proto, label) for protos in self.uploaded_protos
                   for label, proto in protos.items()]
        self.head.train()
        for features, labels in DataLoader(samples, self.batch_size, shuffle=True):
            loss = F.cross_entropy(self.head(features.to(self.device)), labels.to(self.device))
            self.optimizer_head.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer_head.step()
        self.head.eval()
        reps = {c.id: optional_item(c.role, 'reps', self.save_folder_name, self.device) or {}
                for c in active}
        for label in range(self.num_classes):
            nodes = [(c.id, protos[label]) for c, protos in zip(active, self.uploaded_protos)
                     if label in protos]
            if not nodes:
                continue
            features = [proto for _, proto in nodes]
            reliability = torch.stack([calculate_performance(self.head, p, label) for p in features])
            normalized = F.normalize(torch.stack(features), dim=1)
            weights = reliability[:, None] * (normalized @ normalized.T)
            aggregated = rep_aggregate(self.args, features, weights)
            for (cid, _), value in zip(nodes, aggregated):
                reps[cid][label] = value.detach()
        for client in active:
            client.set_rep(reps[client.id])
        save_item(self.head, self.role, 'head', self.save_folder_name)
