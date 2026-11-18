"""ETF-ProtoFL server: sample-count weighted averaging of unnormalised local prototypes.

The paper's weight gamma_c^i = |D_{i,c}| / (N_c * |N_c|) is read here as a sample-count
weighted mean over the clients that hold class c (weights normalised to sum to 1).
"""
import time
from collections import defaultdict

from flcore.clients.clientetfproto import clientETFProto
from flcore.servers.serverbase import Server


class ETFProtoFL(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.set_slow_clients()
        self.set_clients(clientETFProto)
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")
        self.Budget = []
        self.global_protos = None
        # oracle diagnostic: leave these clients out of prototype aggregation (they still train/evaluate)
        self.agg_weight = getattr(args, "etf_agg_weight", "count")
        self.exclude = {int(c) for c in str(getattr(args, "etf_exclude_clients", "")).split(",") if c.strip()}

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate heterogeneous models")
                self.evaluate()
                hc = sum(getattr(c, "last_head_correct", 0) for c in self.clients)
                print(f"ETF-head Test Accuracy (diagnostic): {hc / max(1, sum(c.test_samples for c in self.clients)):.4f}")
            for client in self.selected_clients:
                client.global_protos = self.global_protos
            self.run_clients(self.selected_clients)
            if i % self.eval_gap == 0:
                print(f"[Round {i}] Per-client margin: " + " ".join(
                    f"{c.id}:{c.current_margin:.4f}" for c in self.selected_clients))
            self.receive_protos()
            self.Budget.append(time.time() - s_t)
            print('-' * 50, self.Budget[-1])
            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break
        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        self.save_results()

    def receive_protos(self):
        def aggregate(clients):
            sums, weights = defaultdict(float), defaultdict(float)
            for client in clients:
                gap = getattr(client, "upload_gap", None)
                scale = 1.0 if self.agg_weight == "count" or gap is None else max(gap, 1e-6)
                for c, proto in client.protos.items():
                    n = client.proto_counts[c] * scale  # gap: well-separated clients count more
                    sums[c] = sums[c] + n * proto
                    weights[c] += n
            return {c: sums[c] / weights[c] for c in sums}
        kept = [c for c in self.selected_clients if c.id not in self.exclude]
        protos = aggregate(kept)
        if self.exclude:
            # classes held only by excluded clients would otherwise vanish for everyone
            fallback = aggregate([c for c in self.selected_clients if c.id in self.exclude])
            protos.update({c: p for c, p in fallback.items() if c not in protos})
        self.global_protos = protos
        for client in self.clients:
            client.global_protos = self.global_protos
