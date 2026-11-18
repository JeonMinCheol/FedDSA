import numpy as np
from flcore.clients.clientapen import clientAPEN
from flcore.clients.clientbase import save_item
from flcore.servers.serverbase import Server
from flcore.servers.serverfml import FML
from flcore.trainmodel.models import BaseHeadSplit


class FedAPEN(FML):
    """FedAPEN server: plain FedAvg of the shared model (same loop as FML); clients differ only in training/inference."""

    def __init__(self, args, times):
        Server.__init__(self, args, times)
        global_model = BaseHeadSplit(args, is_global=True) if hasattr(args, 'global_model') else BaseHeadSplit(args, 0)
        save_item(global_model.to(args.device), self.role, 'global_model', self.save_folder_name)
        self.set_slow_clients()
        self.set_clients(clientAPEN)
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        self.Budget = []

    def evaluate(self, *args, **kwargs):
        out = super().evaluate(*args, **kwargs)
        d = [c.diag for c in self.clients if getattr(c, "diag", None)]
        if d:
            n = sum(x[2] for x in d)
            print(f"APENDiag private={sum(x[0] for x in d) / n:.4f} shared={sum(x[1] for x in d) / n:.4f} "
                  f"w_private mean={np.mean([x[3] for x in d]):.3f} min={min(x[3] for x in d):.3f} max={max(x[3] for x in d):.3f}")
        return out
