"""Shared data/model handling for the integrated prototype algorithms."""

from contextlib import contextmanager

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from flcore.clients.clientbase import Client, item_exists, load_item
from utils.data_utils import read_client_data


def move_input(x, device):
    if isinstance(x, (tuple, list)):
        # Text models expect the sequence lengths on the CPU.
        return [x[0].to(device), *x[1:]]
    return x.to(device)


def optional_item(role, name, folder, device):
    if not item_exists(role, name, folder):
        return None
    from flcore.clients.clientbase import move_loaded_item
    return move_loaded_item(load_item(role, name, folder), device)


def prototype_mse(features, labels, prototypes):
    """Missing classes contribute zero; normalize by the full batch, as upstream."""
    target = features.detach().clone()
    for label in labels.unique().tolist():
        if label in prototypes and torch.is_tensor(prototypes[label]):
            target[labels == label] = prototypes[label].detach().to(features.device)
    return torch.nn.functional.mse_loss(features, target)


@contextmanager
def singleton_batchnorm(model, size):
    """Keep rare singleton batches; use running BN statistics for that batch."""
    layers = [m for m in model.modules()
              if isinstance(m, torch.nn.modules.batchnorm._BatchNorm) and m.training and size == 1]
    try:
        for layer in layers:
            layer.eval()
        yield
    finally:
        for layer in layers:
            layer.train()


class PrototypeClient(Client):
    def load_model(self):
        model = load_item(self.role, 'model', self.save_folder_name)
        if model is None:
            raise FileNotFoundError(f"Missing model for {self.role}: {self.save_folder_name}")
        return model.to(self.device)

    def load_train_data(self, batch_size=None, shuffle=True):
        data = read_client_data(self.dataset, self.id, is_train=True, few_shot=self.few_shot)
        if not data:
            raise ValueError(f"Client {self.id} has no training samples")
        return self.create_data_loader(data, batch_size or self.batch_size,
                                       drop_last=False, shuffle=shuffle)

    @torch.no_grad()
    def collect_prototypes(self, model):
        model.eval()
        sums, counts = {}, {}
        for x, y in self.load_train_data(shuffle=False):
            features = model.base(move_input(x, self.device))
            for label in y.unique().tolist():
                value = features[y.to(self.device) == label].sum(0)
                sums[label] = sums.get(label, 0) + value
                counts[label] = counts.get(label, 0) + (y == label).sum().item()
        return {label: (value / counts[label]).detach() for label, value in sums.items()}

    def test_metrics(self):
        model = self.load_model().eval()
        correct, count, probabilities, labels = 0, 0, [], []
        with torch.no_grad():
            for x, y in self.load_test_data():
                output = model(move_input(x, self.device))
                correct += (output.argmax(1).cpu() == y.cpu()).sum().item()
                count += y.numel()
                probabilities.append(output.cpu().numpy())
                labels.append(y.cpu().numpy())
        if not count:
            return 0, 0, 0.0
        truth = np.eye(self.num_classes)[np.concatenate(labels)]
        auc = roc_auc_score(truth, np.concatenate(probabilities), average='micro')
        return correct, count, auc

    def train_metrics(self):
        model = self.load_model().eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for x, y in self.load_train_data(shuffle=False):
                loss = self.loss(model(move_input(x, self.device)), y.to(self.device))
                total += loss.item() * y.numel()
                count += y.numel()
        return total, count
