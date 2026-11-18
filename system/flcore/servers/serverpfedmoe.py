from flcore.clients.clientbase import load_item, save_item
from flcore.clients.clientpfedmoe import clientPFedMoE
from flcore.servers.serverbase import Server
from flcore.servers.serverfml import FML
from flcore.trainmodel.models import BaseHeadSplit


class PFedMoE(FML):
    """pFedMoE server (Yi et al. 2024): sample-weighted FedAvg (Eq. 9) of the shared homogeneous small feature extractor
    only; same round loop as FML. The shared model's own head is never used (each client predicts with its own header)."""

    def __init__(self, args, times):
        Server.__init__(self, args, times)
        global_model = BaseHeadSplit(args, is_global=True) if hasattr(args, 'global_model') else BaseHeadSplit(args, 0)
        save_item(global_model.to(args.device), self.role, 'global_model', self.save_folder_name)
        self.set_slow_clients()
        self.set_clients(clientPFedMoE)
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        self.Budget = []

    def aggregate_parameters(self):
        assert len(self.uploaded_ids) > 0
        g = load_item(self.role, 'global_model', self.save_folder_name)
        states = [(load_item(self.clients[cid].role, 'global_model', self.clients[cid].save_folder_name).state_dict(), w)
                  for cid, w in zip(self.uploaded_ids, self.uploaded_weights)]  # weights n_k / n, summing to 1
        g.load_state_dict({k: sum(w * st[k].float() for st, w in states).to(v.dtype)
                           if k.startswith("base.") and v.is_floating_point() else v
                           for k, v in g.state_dict().items()})
        save_item(g, self.role, 'global_model', self.save_folder_name)

    def evaluate(self, *args, **kwargs):
        out = super().evaluate(*args, **kwargs)
        d = [c.diag for c in self.clients if getattr(c, "diag", None)]
        if d:
            print(f"pFedMoEDiag mean gate weight on the shared expert={sum(x[0] for x in d) / sum(x[1] for x in d):.3f}")
        return out
