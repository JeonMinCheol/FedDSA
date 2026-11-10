"""Round lifecycle shared by the three integrated implementations."""

import time

import numpy as np

from flcore.servers.serverbase import Server


class PrototypeServer(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        if not 0 < self.join_ratio <= 1 or self.num_join_clients < 1:
            raise ValueError("join_ratio must select at least one client and be <= 1")
        if not 0 <= self.client_drop_rate < 1:
            raise ValueError("client_drop_rate must be in [0, 1)")
        if int((1 - self.client_drop_rate) * self.num_join_clients) < 1:
            raise ValueError("client_drop_rate leaves no uploading clients")
        if self.eval_gap < 1 or self.global_rounds < 0 or self.local_epochs < 1:
            raise ValueError("Require eval_gap >= 1, global_rounds >= 0, local_epochs >= 1")
        self.Budget = []

    def evaluate(self, acc=None, loss=None):
        _, counts, correct, auc = self.test_metrics()
        if sum(counts) == 0:
            raise ValueError("Evaluation requires nonempty client test splits")
        accuracy = sum(correct) / sum(counts)
        mean_auc = sum(auc) / sum(counts)
        (self.rs_test_acc if acc is None else acc).append(accuracy)
        self.rs_test_auc.append(mean_auc)
        print(f"Averaged Test Accuracy: {accuracy:.4f}")
        print(f"Averaged Test AUC: {mean_auc:.4f}")
        print(f"Std Test Accuracy: {np.std([a / n for a, n in zip(correct, counts)]):.4f}")

    def train(self):
        # Match flcore's convention: round 0 includes an update.
        for round_index in range(self.global_rounds + 1):
            start = time.time()
            self.selected_clients = self.select_clients()
            if round_index % self.eval_gap == 0:
                print(f"\n-------------Round number: {round_index}-------------")
                self.evaluate()
            self.run_round(round_index)
            self.Budget.append(time.time() - start)
            print(f"Round time: {self.Budget[-1]:.3f}s")
            if self.auto_break and self.check_done([self.rs_test_acc], top_cnt=self.top_cnt):
                break
        # Include the final trained model, including in one-round smoke runs.
        print("\nEvaluate final models")
        self.evaluate()
        print(f"Best accuracy: {max(self.rs_test_acc):.6f}")
        costs = self.Budget[1:] or self.Budget
        print(f"Average round time: {sum(costs) / len(costs):.3f}s")
        self.save_results()

    def active_clients(self):
        self.receive_ids()
        return [self.clients[cid] for cid in self.uploaded_ids]
