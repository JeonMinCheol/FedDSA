import copy
from flcore.clients.clientbase import load_item
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
        gargs = args
        if getattr(args, "apen_native_head", False):  # shared model with its own classifier shape (private models unchanged)
            gargs = copy.copy(args)
            gargs.fedsa_projector = "native"
        global_model = BaseHeadSplit(gargs, is_global=True) if hasattr(args, 'global_model') else BaseHeadSplit(gargs, 0)
        save_item(global_model.to(args.device), self.role, 'global_model', self.save_folder_name)
        self.set_slow_clients()
        self.set_clients(clientAPEN)
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        self.Budget = []

    def evaluate(self, *args, **kwargs):
        out = super().evaluate(*args, **kwargs)
        swap = [int(v) for v in str(getattr(self.args, "apen_swap_rounds", "") or "").split(",") if v.strip()]
        if swap:  # checkpoint swap (review round 2): keep shared snapshots, pair them with the final private models
            r = (len(self.rs_test_acc) - 1) * self.eval_gap
            if r in swap:
                if not hasattr(self, "_shared_snaps"):
                    self._shared_snaps = {}
                g = load_item(self.role, 'global_model', self.save_folder_name)
                self._shared_snaps[r] = {k: v.detach().cpu().clone() for k, v in g.state_dict().items()}
            if r >= self.global_rounds and getattr(self, "_shared_snaps", None):
                for rs, st in sorted(self._shared_snaps.items()) + [(r, {k: v.detach().cpu().clone() for k, v in
                                                                        load_item(self.role, 'global_model', self.save_folder_name).state_dict().items()})]:
                    res = [c.swap_eval(st) for c in self.clients]
                    n = sum(x["n"] for x in res)
                    acc = {k: 100 * sum(x[k] for x in res) / n for k in ("shared", "shared_prior", "ensemble")}
                    print(f"APENSwap shared@R{rs} private@R{r}: shared={acc['shared']:.2f} shared+prior={acc['shared_prior']:.2f} "
                          f"ensemble={acc['ensemble']:.2f} margin={acc['ensemble'] - acc['shared_prior']:+.2f} "
                          f"w*_mean={np.mean([x['w'] for x in res]):.2f} tau*_mean={np.mean([x['tau'] for x in res]):.2f} "
                          f"shared_nll_val={sum(x['nll_val'] for x in res) / sum(x['n_val'] for x in res):.4f} "
                          f"shared_nll_test={sum(x['nll_test'] for x in res) / n:.4f}")
        d = [c.diag for c in self.clients if getattr(c, "diag", None)]
        if d:
            n = sum(x[2] for x in d)
            print(f"APENDiag private={sum(x[0] for x in d) / n:.4f} shared={sum(x[1] for x in d) / n:.4f} "
                  f"shared+prior={sum(x[4] for x in d) / n:.4f} "
                  f"w_private mean={np.mean([x[3] for x in d]):.3f} min={min(x[3] for x in d):.3f} max={max(x[3] for x in d):.3f}")
            # per-client private/shared/weight/shared+log local prior, to check whether the ensemble gain is concentrated on a few clients
            print("APENDiagClient " + " ".join(f"{c.id}:{c.diag[0] / c.diag[2]:.4f}/{c.diag[1] / c.diag[2]:.4f}/{c.diag[3]:.3f}/{c.diag[4] / c.diag[2]:.4f}"
                                               for c in self.clients if getattr(c, "diag", None)))
            dn = [c for c in self.clients if getattr(c, "diag_nll", None)]
            if dn:
                print("APENNLL " + " ".join(f"{k}={sum(c.diag_nll[k] for c in dn) / n:.4f}" for k in dn[0].diag_nll)
                      + " | APENVal " + " ".join(f"{k}={sum(c.diag_val[k] for c in dn) / max(1, sum(c.diag_val['n'] for c in dn)):.4f}"
                                                 for k in ("ensemble", "shared_prior")))
            if getattr(self.args, "apen_dump_perclass", False):  # per-client per-class counts, last eval kept (analysis)
                import json
                out_p = f"logs/{self.args.goal}_perclass.json"
                json.dump({"round": len(self.rs_test_acc) - 1,
                           "clients": {c.id: c.diag_cls for c in self.clients if getattr(c, "diag_cls", None)}},
                          open(out_p, "w"))
        return out
