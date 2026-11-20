"""FedScout server: sample-weighted FedAvg of the scout backbone (personal projector/head, realigned after every
average), memorization gate (restore the best held-out backbone, freeze it and stop sharing), and evaluation of the
fused prediction scout + alpha * learner with one alpha picked on the pooled held-out splits."""
import copy
import time

import numpy as np
import torch
from sklearn import metrics
from sklearn.preprocessing import label_binarize

from flcore.clients.clientscout import clientScout
from flcore.servers.serverbase import Server
from flcore.trainmodel.models import BaseHeadSplit


class FedScout(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.args = args
        self.set_slow_clients()
        self.set_clients(clientScout)
        arch = args.scout_arch
        assert arch[0] == "c" and arch[1:].isdigit(), "--scout_arch must be c<depth> (CIFAR ResNet-depth)"
        cargs = copy.copy(args)
        cargs.models = [f'CIFARResNet({int(arch[1:])}, width={float(args.scout_width)}, num_classes=args.num_classes)']
        cargs.fedsa_projector = args.fedsa_projector  # linear, seqpool or native (--scout_projector)
        scout = BaseHeadSplit(cargs, 0)
        for c in self.clients:
            c.scout = copy.deepcopy(scout)  # same initial weights everywhere
        sd = scout.state_dict()
        self.shared_keys = [k for k in sd if args.scout_share == "all" or k.startswith("base.0.")]
        self.shared_floats = sum(sd[k].numel() for k in self.shared_keys if sd[k].is_floating_point())
        self.comm_floats = 0  # upload + download, all clients
        self.frozen = False
        self.best, self.best_round, self.best_state = -1.0, 0, None
        self.taus = [float(t) for t in str(args.scout_cascade_taus).split(",") if t]
        # learner weight grid for the fusion scout + alpha * learner, ascending (ties -> smaller); alpha > 1 lets the
        # learner outweigh the scout (FedAPEN learns w_private ~ 0.57, i.e. alpha ~ 1.3)
        self.alphas = [float(x) for x in str(args.scout_alphas).split(",") if x]
        self.Budget = []
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print(f"FedScout: scout {arch} x{args.scout_width} ({sum(p.numel() for p in scout.parameters()) / 1e3:.1f}K params), "
              f"shared per round {self.shared_floats / 1e3:.1f}K floats ({args.scout_share}), gate {args.scout_gate}, "
              f"realign {args.scout_realign}, residual {args.scout_residual}, screen tau {args.scout_screen_tau}, "
              f"projector {args.scout_projector} ({args.fedsa_projector})")

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            for c in self.clients:
                c.set_round(i)
            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate FedScout (fused = scout + alpha * residual learner)")
                self.evaluate()
            for client, result in self.run_clients(self.selected_clients, return_exceptions=True):
                if isinstance(result, Exception):
                    raise RuntimeError(f"FedScout client {client.id} failed during local training.") from result
            if not self.frozen and self.args.scout_share != "none":
                self.aggregate()
            if self.args.scout_gate > 0 and self.args.scout_share != "none":  # the gate restores one shared backbone
                self.gate(i)
            self.log_train(i)
            self.Budget.append(time.time() - s_t)
            print("-" * 50, self.Budget[-1])
            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break
        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        if len(self.Budget) > 1:
            print(sum(self.Budget[1:]) / len(self.Budget[1:]))
        self.save_results()

    @torch.no_grad()
    def aggregate(self):
        total = float(sum(c.train_samples for c in self.selected_clients))
        states = [(c.scout.state_dict(), c.train_samples / total) for c in self.selected_clients]
        avg = {k: (sum(w * st[k].float().to(self.device) for st, w in states).to(states[0][0][k].dtype)
                   if states[0][0][k].is_floating_point() else states[0][0][k].clone()) for k in self.shared_keys}
        for c in self.clients:
            c.scout.load_state_dict(avg, strict=self.args.scout_share == "all")
        self.comm_floats += (len(self.selected_clients) + len(self.clients)) * self.shared_floats
        if self.args.scout_realign > 0 and self.args.scout_share == "backbone":
            with torch.enable_grad():
                for c in self.clients:
                    c.realign(self.args.scout_realign)

    def gate(self, rnd):
        """Memorization gate: track the received scout on the clients' held-out splits; once it has not improved for
        --scout_gate rounds (or at --scout_freeze_at), restore the best backbone, freeze it and stop sharing."""
        if self.frozen:
            return
        stats = [c.gate_stats() for c in self.clients]
        n = sum(s[2] for s in stats)
        val, tr = sum(s[0] for s in stats) / n, sum(s[1] for s in stats) / n
        print(f"[Round {rnd}] ScoutGate held_out={val:.4f} train={tr:.4f} gap={tr - val:.4f}")
        if val > self.best:
            self.best, self.best_round = val, rnd
            self.best_state = {k: v.detach().cpu().clone() for k, v in self.clients[0].scout.state_dict().items()
                               if k.startswith("base.0.")}
        if rnd - self.best_round >= self.args.scout_gate or 0 < self.args.scout_freeze_at <= rnd:
            for c in self.clients:
                own = c.scout.state_dict()
                c.scout.load_state_dict({k: v.to(own[k].device, own[k].dtype) for k, v in self.best_state.items()}, strict=False)
                c.frozen = True
                if self.args.scout_freeze_head:
                    c.realign(max(1, self.args.scout_realign))  # refit my head to the restored backbone once, then fix it
                    c.head_frozen = True
            self.frozen = True
            print(f"[Round {rnd}] ScoutGate freeze: restored the round-{self.best_round} backbone (held-out {self.best:.4f})")

    def log_train(self, rnd):
        w = [(c.latest_train_metrics, c.train_samples) for c in self.selected_clients if c.latest_train_metrics]
        tot = sum(n for _, n in w) or 1
        avg = {k: sum(m[k] * n for m, n in w) / tot for k in (w[0][0] if w else {})}
        print(f"[Round {rnd}] ScoutTrain " + " ".join(f"{k}={v:.4f}" for k, v in avg.items())
              + f" comm_MB={4 * self.comm_floats / 1e6:.1f} frozen={int(self.frozen)}")

    @torch.no_grad()
    def test_metrics(self):
        """Fused accuracy with one alpha chosen on the pooled held-out splits (first max -> smaller alpha), plus the
        scout alone, the learner alone, every alpha, and the confidence cascade (learner only where the scout's top-1
        probability is below tau)."""
        outs = {c.id: c.logits() for c in self.clients}
        held = np.zeros(len(self.alphas))
        for o in outs.values():
            lt, lp, y = o["held"]
            held += [int(((lt + a * lp).argmax(1) == y).sum()) for a in self.alphas]
        a_star = self.alphas[int(np.argmax(held))]
        ids, nums, corrects, aucs = [], [], [], []
        tot = {"scout": 0, "scout_prior": 0, "learner": 0, "logit_sum": 0, **{f"a{a:g}": 0 for a in self.alphas}}
        cas = {t: [0, 0] for t in self.taus}  # correct, learner runs
        for cid, o in outs.items():
            lt, lp, y = o["test"]
            z = lt + a_star * lp
            ids.append(cid)
            nums.append(len(y))
            corrects.append(float((z.argmax(1) == y).sum()))
            lb = label_binarize(y.numpy(), classes=np.arange(self.num_classes))
            aucs.append(float(metrics.roc_auc_score(lb, z.numpy(), average="micro")) * len(y))
            tot["scout"] += int((lt.argmax(1) == y).sum())
            tot["scout_prior"] += int(((lt + o["log_prior"]).argmax(1) == y).sum())  # scout + log local class frequencies
            tot["learner"] += int((lp.argmax(1) == y).sum())
            tot["logit_sum"] += int(((lt + lp).argmax(1) == y).sum())
            for a in self.alphas:
                tot[f"a{a:g}"] += int(((lt + a * lp).argmax(1) == y).sum())
            conf = lt.softmax(1).amax(1)
            for t in self.taus:
                run = conf < t
                cas[t][0] += int((torch.where(run, z.argmax(1), lt.argmax(1)) == y).sum())
                cas[t][1] += int(run.sum())
        n = sum(nums)
        print(f"ScoutEval alpha={a_star:g} fused={sum(corrects) / n:.4f} "
              + " ".join(f"{k}={v / n:.4f}" for k, v in tot.items())
              + " " + " ".join(f"cascade{t:g}={c / n:.4f} run{t:g}={r / n:.4f}" for t, (c, r) in cas.items())
              + f" held_counts={','.join(str(int(h)) for h in held)}")
        return ids, nums, corrects, aucs
