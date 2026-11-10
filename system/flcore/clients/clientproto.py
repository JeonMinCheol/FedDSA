import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client, load_item, save_item
from collections import defaultdict


class clientProto(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        torch.manual_seed(0)
        self.device = torch.device(args.device)

        self.loss_mse = nn.MSELoss()
        self.lamda = args.lamda
        self.use_amp = bool(getattr(args, "use_amp", self.device.type == "cuda"))
        self.amp_enabled = bool(self.use_amp and self.device.type == "cuda")
        self.feature_dim = int(args.feature_dim)

    def _move_input(self, x, y):
        if isinstance(x, (list, tuple)):
            x = x[0]
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True).long().view(-1)
        return x, y

    def _proto_for_labels(self, global_protos, y, rep):
        if not global_protos:
            return None

        proto_new = rep.detach().clone()
        matched = False
        for label in y.detach().unique().tolist():
            proto = self._coerce_proto(global_protos.get(int(label), None), rep.shape[1], rep.device, rep.dtype)
            if proto is None:
                continue
            mask = y == int(label)
            proto_new[mask] = proto.view(1, -1).expand(int(mask.sum().item()), -1)
            matched = True
        return proto_new if matched else None

    def _coerce_proto(self, proto, feature_dim=None, device=None, dtype=torch.float32):
        if proto is None or isinstance(proto, list):
            return None

        if feature_dim is None:
            feature_dim = self.feature_dim
        if device is None:
            device = self.device

        proto = proto.to(device=device, dtype=dtype) if torch.is_tensor(proto) else torch.as_tensor(
            proto,
            device=device,
            dtype=dtype,
        )
        proto = torch.nan_to_num(proto, nan=0.0, posinf=0.0, neginf=0.0)

        if proto.numel() == feature_dim:
            return proto.reshape(feature_dim)
        if proto.ndim >= 2 and proto.shape[-1] == feature_dim:
            return proto.reshape(-1, feature_dim).mean(dim=0)
        return None

    def _global_proto_matrix(self, global_protos):
        if not global_protos:
            return None, None

        labels = []
        protos = []
        for label in sorted(global_protos.keys()):
            proto = self._coerce_proto(global_protos[label], self.feature_dim, self.device, torch.float32)
            if proto is None:
                continue
            labels.append(int(label))
            protos.append(proto)

        if len(protos) == 0:
            return None, None
        return torch.tensor(labels, device=self.device, dtype=torch.long), torch.stack(protos, dim=0)

    @torch.no_grad()
    def _protos_from_sums(self, proto_sum, proto_count):
        if proto_sum is None or proto_count is None:
            return {}

        return {
            int(label): (proto_sum[label] / proto_count[label].clamp_min(1.0)).detach()
            for label in torch.nonzero(proto_count > 0, as_tuple=False).view(-1).tolist()
        }

    def train(self):
        trainloader = self.load_train_data()
        model = getattr(self, "model", None)
        if model is None:
            model = load_item(self.role, 'model', self.save_folder_name)
            self.model = model
        model.to(self.device)
        if hasattr(self, "global_protos"):
            global_protos = self.global_protos
        else:
            global_protos = load_item('Server', 'global_protos', self.save_folder_name)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        grad_scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)
        model.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        proto_sum = None
        proto_count = None
        for step in range(max_local_epochs):
            for x, y in trainloader:
                x, y = self._move_input(x, y)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                    rep = model.base(x)
                    output = model.head(rep)
                    loss = self.loss(output, y)

                    proto_new = self._proto_for_labels(global_protos, y, rep)
                    if proto_new is not None:
                        loss = loss + self.loss_mse(proto_new.float(), rep.float()) * self.lamda

                with torch.no_grad():
                    rep_detached = rep.detach().float()
                    if proto_sum is None:
                        proto_sum = torch.zeros(
                            self.num_classes,
                            rep_detached.shape[1],
                            device=rep_detached.device,
                            dtype=torch.float32,
                        )
                        proto_count = torch.zeros(self.num_classes, device=rep_detached.device, dtype=torch.float32)
                    proto_sum.index_add_(0, y, rep_detached)
                    proto_count.index_add_(0, y, torch.ones_like(y, dtype=proto_count.dtype))

                optimizer.zero_grad(set_to_none=True)
                if self.amp_enabled:
                    grad_scaler.scale(loss).backward()
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        self.protos = self._protos_from_sums(proto_sum, proto_count)
        if self.persist_client_models:
            save_item(model, self.role, 'model', self.save_folder_name)

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


    def test_metrics(self):
        testloader = self.load_test_data()
        model = getattr(self, "model", None)
        if model is None:
            model = load_item(self.role, 'model', self.save_folder_name)
            self.model = model
        model.to(self.device)
        if hasattr(self, "global_protos"):
            global_protos = self.global_protos
        else:
            global_protos = load_item('Server', 'global_protos', self.save_folder_name)
        model.eval()

        test_acc = 0
        test_num = 0
        
        proto_labels, proto_matrix = self._global_proto_matrix(global_protos)
        if proto_matrix is not None:
            with torch.no_grad():
                for x, y in testloader:
                    x, y = self._move_input(x, y)
                    with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                        rep = model.base(x)

                    distances = torch.cdist(rep.float(), proto_matrix)
                    pred = proto_labels[torch.argmin(distances, dim=1)]
                    test_acc += (torch.sum(pred == y)).item()
                    test_num += y.shape[0]

            return test_acc, test_num, 0
        else:
            return 0, 1e-5, 0

    def train_metrics(self):
        trainloader = self.load_train_data()
        model = getattr(self, "model", None)
        if model is None:
            model = load_item(self.role, 'model', self.save_folder_name)
            self.model = model
        model.to(self.device)
        if hasattr(self, "global_protos"):
            global_protos = self.global_protos
        else:
            global_protos = load_item('Server', 'global_protos', self.save_folder_name)
        model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                x, y = self._move_input(x, y)
                with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                    rep = model.base(x)
                    output = model.head(rep)
                    loss = self.loss(output, y)

                    proto_new = self._proto_for_labels(global_protos, y, rep)
                    if proto_new is not None:
                        loss = loss + self.loss_mse(proto_new.float(), rep.float()) * self.lamda
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num


# https://github.com/yuetan031/fedproto/blob/main/lib/utils.py#L205
def agg_func(protos):
    """
    Returns the average of the weights.
    """

    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]

    return protos
