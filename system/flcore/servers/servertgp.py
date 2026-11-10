import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from flcore.clients.clienttgp import clientTGP
from flcore.servers.serverbase import Server
from flcore.clients.clientbase import load_item, save_item
from collections import defaultdict
from torch.utils.data import DataLoader


class FedTGP(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientTGP)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.num_classes = args.num_classes

        self.server_learning_rate = args.local_learning_rate
        self.batch_size = args.batch_size
        self.server_batch_size = int(getattr(args, "server_batch_size", self.batch_size))
        self.server_epochs = args.server_epochs
        self.margin_threthold = args.margin_threthold

        self.feature_dim = args.feature_dim
        self.server_hidden_dim = self.feature_dim
        
        self.TGP = None
        if args.save_folder_name == 'temp' or 'temp' not in args.save_folder_name:
            TGP = Trainable_Global_Prototypes(
                self.num_classes, 
                self.server_hidden_dim, 
                self.feature_dim, 
                self.device
            ).to(self.device)
            self.TGP = TGP
            save_item(TGP, self.role, 'TGP', self.save_folder_name)
            print(TGP)
        self.CEloss = nn.CrossEntropyLoss()
        self.MSEloss = nn.MSELoss()

        self.gap = torch.ones(self.num_classes, device=self.device) * 1e9
        self.min_gap = None
        self.max_gap = None
        self.global_protos = None
        self.use_amp = bool(getattr(args, "use_amp", str(self.device).startswith("cuda")))
        self.amp_enabled = bool(self.use_amp and str(self.device).startswith("cuda"))

        for client in self.clients:
            client.global_protos = self.global_protos

    def _coerce_proto(self, proto):
        if proto is None or isinstance(proto, list):
            return None

        proto = proto.to(device=self.device, dtype=torch.float32) if torch.is_tensor(proto) else torch.as_tensor(
            proto,
            device=self.device,
            dtype=torch.float32,
        )
        proto = torch.nan_to_num(proto, nan=0.0, posinf=0.0, neginf=0.0)

        if proto.numel() == self.feature_dim:
            return proto.reshape(self.feature_dim)
        if proto.ndim >= 2 and proto.shape[-1] == self.feature_dim:
            return proto.reshape(-1, self.feature_dim).mean(dim=0)
        return None

    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate heterogeneous models")
                self.evaluate()

            self.run_clients(self.selected_clients)


            self.receive_protos()
            self.update_TGP()

            self.Budget.append(time.time() - s_t)
            print('-'*50, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        if len(self.Budget) > 1:
            print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        

    def receive_protos(self):
        assert (len(self.selected_clients) > 0)

        self.uploaded_ids = []
        self.uploaded_protos = []
        uploaded_protos_per_client = []
        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            protos = getattr(client, "protos", None)
            if protos is None:
                protos = load_item(client.role, 'protos', client.save_folder_name)
            cleaned_protos = {}
            for k in protos.keys():
                proto = self._coerce_proto(protos[k])
                if proto is None:
                    continue
                cleaned_protos[int(k)] = proto
                self.uploaded_protos.append((proto, int(k)))
            uploaded_protos_per_client.append(cleaned_protos)

        # calculate class-wise minimum distance
        self.gap = torch.ones(self.num_classes, device=self.device) * 1e9
        avg_protos = proto_cluster(uploaded_protos_per_client)
        if len(avg_protos) > 1:
            labels = torch.tensor(sorted(avg_protos.keys()), device=self.device, dtype=torch.long)
            proto_tensor = torch.stack(
                [avg_protos[int(label)].to(device=self.device, dtype=torch.float32) for label in labels.tolist()],
                dim=0,
            )
            distances = torch.cdist(proto_tensor, proto_tensor)
            distances.fill_diagonal_(float("inf"))
            self.gap[labels] = torch.min(distances, dim=1).values
        elif len(avg_protos) == 1:
            only_label = next(iter(avg_protos.keys()))
            self.gap[only_label] = 1.0

        self.min_gap = torch.min(self.gap[self.gap < 1e8]) if bool((self.gap < 1e8).any().item()) else torch.tensor(1.0, device=self.device)
        self.gap = torch.where(self.gap > 1e8, self.min_gap, self.gap)
        self.max_gap = torch.max(self.gap)
        print('class-wise minimum distance', self.gap)
        print('min_gap', self.min_gap)
        print('max_gap', self.max_gap)
            
    def update_TGP(self):
        TGP = self.TGP
        if TGP is None:
            TGP = load_item(self.role, 'TGP', self.save_folder_name)
            self.TGP = TGP
        TGP.to(self.device)
        TGP_opt = torch.optim.SGD(TGP.parameters(), lr=self.server_learning_rate)
        grad_scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)
        TGP.train()
        class_ids = torch.arange(self.num_classes, device=self.device, dtype=torch.long)
        last_loss = None

        for e in range(self.server_epochs):
            proto_loader = DataLoader(self.uploaded_protos, self.server_batch_size,
                                      drop_last=False, shuffle=True)
            for proto, y in proto_loader:
                proto = proto.to(self.device, non_blocking=True).float()
                y = y.to(self.device, non_blocking=True).long()

                with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                    proto_gen = TGP(class_ids)
                    dist = torch.cdist(proto, proto_gen.float())
                    one_hot = F.one_hot(y, self.num_classes).to(self.device)
                    margin = min(float(self.max_gap.item()), self.margin_threthold)
                    dist = dist + one_hot * margin
                    loss = self.CEloss(-dist, y)

                TGP_opt.zero_grad(set_to_none=True)
                if self.amp_enabled:
                    grad_scaler.scale(loss).backward()
                    grad_scaler.step(TGP_opt)
                    grad_scaler.update()
                else:
                    loss.backward()
                    TGP_opt.step()
                last_loss = loss

        if last_loss is not None:
            print(f'Server loss: {last_loss.item()}')
        self.uploaded_protos = []
        save_item(TGP, self.role, 'TGP', self.save_folder_name)

        TGP.eval()
        with torch.no_grad():
            generated = TGP(class_ids).detach()
        global_protos = {
            int(class_id): generated[int(class_id)].detach().clone().reshape(self.feature_dim)
            for class_id in range(self.num_classes)
        }
        self.global_protos = global_protos
        for client in self.clients:
            client.global_protos = self.global_protos
        save_item(global_protos, self.role, 'global_protos', self.save_folder_name)


def proto_cluster(protos_list):
    proto_clusters = defaultdict(list)
    for protos in protos_list:
        for k in protos.keys():
            proto_clusters[k].append(protos[k])

    for k in proto_clusters.keys():
        protos = torch.stack(proto_clusters[k])
        proto_clusters[k] = torch.mean(protos, dim=0).detach()

    return proto_clusters
            

class Trainable_Global_Prototypes(nn.Module):
    def __init__(self, num_classes, server_hidden_dim, feature_dim, device):
        super().__init__()

        self.device = device

        self.embedings = nn.Embedding(num_classes, feature_dim)
        layers = [nn.Sequential(
            nn.Linear(feature_dim, server_hidden_dim), 
            nn.ReLU()
        )]
        self.middle = nn.Sequential(*layers)
        self.fc = nn.Linear(server_hidden_dim, feature_dim)

    def forward(self, class_id):
        if torch.is_tensor(class_id):
            class_id = class_id.to(device=self.device, dtype=torch.long)
        else:
            class_id = torch.tensor(class_id, device=self.device, dtype=torch.long)

        emb = self.embedings(class_id)
        mid = self.middle(emb)
        out = self.fc(mid)

        return out
