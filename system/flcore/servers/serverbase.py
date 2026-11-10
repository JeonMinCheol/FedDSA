import torch
import os
import numpy as np
import h5py
import copy
import time
import random
import shutil
import glob
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.data_utils import read_client_data
from flcore.clients.clientbase import load_item, save_item


class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.current_round = 0
        self.max_parallel_clients = int(getattr(args, "max_parallel_clients", 0))
        self.random_seed = int(getattr(args, "random_seed", 0)) + int(times)
        self.rng = np.random.RandomState(self.random_seed)
        self.py_rng = random.Random(self.random_seed)
        self.few_shot = args.few_shot
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.top_cnt = args.top_cnt
        self.auto_break = args.auto_break
        self.role = 'Server'
        if args.save_folder_name == 'temp':
            args.save_folder_name_full = f'{args.save_folder_name}/{args.dataset}/{args.algorithm}/{time.time()}/'
        elif 'temp' in args.save_folder_name:
            args.save_folder_name_full = args.save_folder_name
        else:
            args.save_folder_name_full = f'{args.save_folder_name}/{args.dataset}/{args.algorithm}/'
        self.save_folder_name = args.save_folder_name_full

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []
        self.anchor_3d_snapshots = []

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

    def set_clients(self, clientObj):
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.dataset, i, is_train=True, few_shot=self.few_shot)
            test_data = read_client_data(self.dataset, i, is_train=False, few_shot=self.few_shot)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=train_slow, 
                            send_slow=send_slow)
            self.clients.append(client)

    # random select slow clients
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = self.rng.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)

    def select_clients(self):
        if self.random_join_ratio:
            self.current_num_join_clients = self.rng.choice(
                range(self.num_join_clients, self.num_clients + 1),
                1,
                replace=False,
            )[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        candidates = self.clients
        selected_clients = list(
            self.rng.choice(candidates, self.current_num_join_clients, replace=False)
        )

        return selected_clients

    def _parallel_client_workers(self, clients):
        num_clients = len(clients)
        if num_clients <= 0:
            return 0
        if self.max_parallel_clients < 0:
            return num_clients
        if self.max_parallel_clients == 0:
            if str(self.device).startswith("cuda"):
                return 1
            return max(1, min(num_clients, os.cpu_count() or 1))
        return max(1, min(num_clients, self.max_parallel_clients))

    def run_clients(self, clients=None, client_fn=None, return_exceptions=False):
        clients = list(self.selected_clients if clients is None else clients)
        if len(clients) == 0:
            return []

        if client_fn is None:
            client_fn = lambda client: client.train()

        max_workers = self._parallel_client_workers(clients)
        if max_workers <= 1:
            results = []
            for client in clients:
                try:
                    result = client_fn(client)
                except Exception as exc:
                    if not return_exceptions:
                        raise
                    result = exc
                results.append((client, result))
            return results

        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(client_fn, client): client for client in clients}
            for future in as_completed(futures):
                client = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    if not return_exceptions:
                        raise
                    result = exc
                results.append((client, result))
        return results

    def send_parameters(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            
            client.set_parameters()

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_ids(self):
        assert (len(self.selected_clients) > 0)

        active_clients = self.py_rng.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        tot_samples = 0
        for client in active_clients:
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        assert (len(self.uploaded_ids) > 0)

        client = self.clients[self.uploaded_ids[0]]
        global_model = copy.deepcopy(load_item(client.role, 'model', client.save_folder_name))
        for param in global_model.parameters():
            param.data.zero_()
            
        for w, cid in zip(self.uploaded_weights, self.uploaded_ids):
            client = self.clients[cid]
            client_model = load_item(client.role, 'model', client.save_folder_name)
            for server_param, client_param in zip(global_model.parameters(), client_model.parameters()):
                server_param.data += client_param.data.clone() * w

        save_item(global_model, self.role, 'global_model', self.save_folder_name)
        
    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_test_acc)):
            algo = algo + "_" + self.goal + "_" + str(self.times)
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
        
        if 'temp' in self.save_folder_name:
            try:
                shutil.rmtree(self.save_folder_name)
                print('Deleted.')
            except:
                print('Already deleted.')

    def maybe_save_tsne(self):
        if not bool(getattr(self.args, "save_tsne", False)):
            return

        split_arg = str(getattr(self.args, "tsne_split", "test")).lower()
        splits = ["train", "test"] if split_arg == "both" else [split_arg]
        for split in splits:
            try:
                self.save_tsne_figure(split=split)
            except Exception as exc:
                print(f"TSNE figure skipped for {self.algorithm} ({split}): {exc}")

    def record_anchor_3d_snapshot(self, round_num, anchors=None):
        if not bool(getattr(self.args, "save_anchor_3d", False)):
            return
        interval = int(getattr(self.args, "anchor_3d_interval", 10))
        interval = max(interval, 1)
        if int(round_num) % interval != 0 and int(round_num) != int(self.global_rounds):
            return

        if anchors is None:
            anchors = getattr(self, "semantic_anchors", None)
        if anchors is None:
            return
        anchors = torch.as_tensor(anchors).detach().float().cpu()
        if anchors.ndim != 2 or anchors.size(0) < 2:
            return
        anchors = torch.nan_to_num(anchors, nan=0.0, posinf=0.0, neginf=0.0)
        self.anchor_3d_snapshots.append((int(round_num), anchors.numpy().astype(np.float32)))

    def maybe_save_anchor_3d(self):
        if not bool(getattr(self.args, "save_anchor_3d", False)):
            return
        if not self.anchor_3d_snapshots:
            print(f"Anchor 3D figure skipped for {self.algorithm}: no snapshots")
            return

        save_dir = str(getattr(self.args, "anchor_3d_save_dir", "../figures/anchor3d"))
        os.makedirs(save_dir, exist_ok=True)
        run_name = os.path.basename(os.path.normpath(str(self.save_folder_name)))
        if not run_name:
            run_name = f"{self.dataset}-{self.algorithm}-{self.goal}"
        stem = self._safe_filename(f"{run_name}-anchor3d")
        feature_path = os.path.join(save_dir, f"{stem}.npz")
        figure_path = os.path.join(save_dir, f"{stem}.png")

        rounds = np.asarray([round_num for round_num, _ in self.anchor_3d_snapshots], dtype=np.int64)
        anchors = np.stack([anchor for _, anchor in self.anchor_3d_snapshots], axis=0).astype(np.float32)
        meta = {
            "algorithm": str(self.algorithm),
            "dataset": str(self.dataset),
            "run_name": run_name,
            "random_seed": int(getattr(self.args, "random_seed", 0)),
            "times": int(getattr(self, "times", 0)),
            "num_snapshots": int(len(self.anchor_3d_snapshots)),
            "num_classes": int(anchors.shape[1]),
            "feature_dim": int(anchors.shape[2]),
        }
        np.savez_compressed(
            feature_path,
            rounds=rounds,
            anchors=anchors,
            metadata=json.dumps(meta, sort_keys=True),
        )
        self.plot_anchor_3d_snapshots(rounds, anchors, figure_path, meta)
        print(f"Saved anchor 3D figure: {figure_path}")
        print(f"Saved anchor 3D snapshots: {feature_path}")

    def plot_anchor_3d_snapshots(self, rounds, anchors, figure_path, meta):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(os.path.dirname(os.path.abspath(figure_path)), exist_ok=True)
        num_snapshots, num_classes, dim = anchors.shape
        max_classes = int(getattr(self.args, "anchor_3d_max_classes", 100))
        max_classes = max(2, min(max_classes, num_classes))
        class_indices = np.linspace(0, num_classes - 1, max_classes).round().astype(np.int64)

        selected = anchors[:, class_indices, :]
        flat = selected.reshape(-1, dim).astype(np.float64)
        mean = flat.mean(axis=0, keepdims=True)
        centered = flat - mean
        try:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            basis = vh[: min(3, vh.shape[0])].T
        except np.linalg.LinAlgError:
            basis = np.eye(dim, min(3, dim), dtype=np.float64)
        if basis.shape[1] < 3:
            basis = np.pad(basis, ((0, 0), (0, 3 - basis.shape[1])), mode="constant")

        projected = ((selected.reshape(-1, dim) - mean) @ basis).reshape(num_snapshots, max_classes, 3)
        finite_values = projected[np.isfinite(projected)]
        if finite_values.size == 0:
            finite_values = np.asarray([0.0], dtype=np.float64)
        low, high = np.percentile(finite_values, [1, 99])
        bound = max(abs(float(low)), abs(float(high)), 1e-6)
        bound *= 1.08

        max_panels = int(getattr(self.args, "anchor_3d_max_panels", 6))
        max_panels = max(1, min(max_panels, num_snapshots))
        if num_snapshots <= max_panels:
            panel_indices = np.arange(num_snapshots)
        else:
            panel_indices = np.linspace(0, num_snapshots - 1, max_panels).round().astype(np.int64)
            panel_indices = np.unique(panel_indices)

        cols = min(3, len(panel_indices))
        rows = int(np.ceil(len(panel_indices) / cols))
        fig = plt.figure(figsize=(5.0 * cols, 4.5 * rows))
        scatter = None
        colors = class_indices.astype(np.float32)

        for plot_idx, snapshot_idx in enumerate(panel_indices, start=1):
            ax = fig.add_subplot(rows, cols, plot_idx, projection="3d")
            coords = projected[snapshot_idx]
            scatter = ax.scatter(
                coords[:, 0],
                coords[:, 1],
                coords[:, 2],
                c=colors,
                cmap="turbo",
                s=18,
                alpha=0.82,
                linewidths=0,
            )
            original = selected[snapshot_idx]
            centered_original = original - original.mean(axis=0, keepdims=True)
            radius = float(np.linalg.norm(centered_original, axis=1).mean())
            ax.text2D(
                0.04,
                0.94,
                f"r={radius:.2f}",
                transform=ax.transAxes,
                fontsize=9,
                color="#2f2f2f",
                ha="left",
                va="top",
                bbox={
                    "boxstyle": "round,pad=0.22",
                    "facecolor": "white",
                    "edgecolor": "#dddddd",
                    "alpha": 0.82,
                },
            )
            ax.set_xlim(-bound, bound)
            ax.set_ylim(-bound, bound)
            ax.set_zlim(-bound, bound)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.view_init(elev=22, azim=-58)

        if scatter is not None:
            cbar = fig.colorbar(scatter, ax=fig.axes, shrink=0.62, pad=0.02)
            cbar.set_label("Class")
        fig.savefig(figure_path, dpi=220, bbox_inches="tight")
        plt.close(fig)

    def save_tsne_figure(self, split="test"):
        features, labels, client_ids = self.collect_tsne_features(split=split)
        if features.shape[0] < 3:
            print(f"TSNE figure skipped for {self.algorithm} ({split}): not enough samples")
            return None
        anchor_features, anchor_labels = self.collect_tsne_anchors(
            normalize=bool(getattr(self.args, "tsne_normalize", False)),
            feature_dim=features.shape[1],
        )

        save_dir = str(getattr(self.args, "tsne_save_dir", "../figures/tsne"))
        os.makedirs(save_dir, exist_ok=True)
        stem = self._tsne_artifact_stem(split)
        feature_path = os.path.join(save_dir, f"{stem}-features.npz")
        figure_path = os.path.join(save_dir, f"{stem}.png")

        meta = self._tsne_metadata(split, features.shape[0], anchor_features.shape[0])
        np.savez_compressed(
            feature_path,
            features=features.astype(np.float32),
            labels=labels.astype(np.int64),
            clients=client_ids.astype(np.int64),
            anchor_features=anchor_features.astype(np.float32),
            anchor_labels=anchor_labels.astype(np.int64),
            metadata=json.dumps(meta, sort_keys=True),
        )

        embedding, anchor_embedding = self.compute_tsne_embedding_with_anchors(features, anchor_features)
        self.plot_tsne_panels(
            [(embedding, labels, client_ids, self._tsne_title(meta), anchor_embedding, anchor_labels)],
            figure_path,
            title=None,
        )
        print(f"Saved TSNE figure: {figure_path}")
        print(f"Saved TSNE features: {feature_path}")

        self.try_save_tsne_comparison(feature_path, meta, save_dir)
        return figure_path

    def collect_tsne_features(self, split="test"):
        max_samples = int(getattr(self.args, "tsne_max_samples", 3000))
        max_per_client = int(getattr(self.args, "tsne_max_per_client", 150))
        normalize = bool(getattr(self.args, "tsne_normalize", False))
        split = str(split).lower()
        if split not in {"train", "test"}:
            split = "test"

        features = []
        labels = []
        client_ids = []
        rng = np.random.RandomState(self.random_seed + (17 if split == "test" else 29))

        for client in self.clients:
            client_features, client_labels = self.collect_client_tsne_features(client, split)
            if client_features.shape[0] == 0:
                continue
            if max_per_client > 0 and client_features.shape[0] > max_per_client:
                keep = rng.choice(client_features.shape[0], max_per_client, replace=False)
                client_features = client_features[keep]
                client_labels = client_labels[keep]
            features.append(client_features)
            labels.append(client_labels)
            client_ids.append(np.full(client_labels.shape[0], int(client.id), dtype=np.int64))

        if not features:
            return (
                np.empty((0, int(getattr(self.args, "feature_dim", 0))), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
                np.empty((0,), dtype=np.int64),
            )

        features = np.concatenate(features, axis=0).astype(np.float32)
        labels = np.concatenate(labels, axis=0).astype(np.int64)
        client_ids = np.concatenate(client_ids, axis=0).astype(np.int64)

        if max_samples > 0 and features.shape[0] > max_samples:
            keep = rng.choice(features.shape[0], max_samples, replace=False)
            features = features[keep]
            labels = labels[keep]
            client_ids = client_ids[keep]

        if normalize:
            denom = np.linalg.norm(features, axis=1, keepdims=True)
            features = features / np.maximum(denom, 1e-12)

        return features, labels, client_ids

    def collect_tsne_anchors(self, normalize=False, feature_dim=None):
        anchors = getattr(self, "semantic_anchors", None)
        empty_dim = int(feature_dim or getattr(self.args, "feature_dim", 0))
        if anchors is None:
            return (
                np.empty((0, empty_dim), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
            )

        anchors = torch.as_tensor(anchors).detach().float().cpu()
        if anchors.ndim != 2 or anchors.size(0) == 0:
            return (
                np.empty((0, empty_dim), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
            )
        if feature_dim is not None and int(feature_dim) > 0 and anchors.size(1) != int(feature_dim):
            print(
                f"TSNE anchors skipped for {self.algorithm}: "
                f"anchor dim {anchors.size(1)} != feature dim {int(feature_dim)}"
            )
            return (
                np.empty((0, int(feature_dim)), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
            )

        anchor_features = anchors.numpy().astype(np.float32)
        anchor_labels = np.arange(anchor_features.shape[0], dtype=np.int64)
        if normalize:
            denom = np.linalg.norm(anchor_features, axis=1, keepdims=True)
            anchor_features = anchor_features / np.maximum(denom, 1e-12)
        return anchor_features, anchor_labels

    def collect_client_tsne_features(self, client, split):
        loader = client.load_train_data(drop_last=False) if split == "train" else client.load_test_data()
        model = getattr(client, "model", None)
        if model is None:
            model = load_item(client.role, "model", client.save_folder_name)
        if model is None:
            return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)

        model.to(self.device)
        model.eval()
        feature_batches = []
        label_batches = []

        try:
            with torch.no_grad():
                for x, y in loader:
                    if isinstance(x, (list, tuple)):
                        x = x[0]
                    x = x.to(self.device, non_blocking=True)
                    y = y.to(self.device, non_blocking=True).long().view(-1)
                    if hasattr(model, "shared_representation"):
                        rep = model.shared_representation(x)
                    else:
                        rep = model.base(x) if hasattr(model, "base") else model(x)
                    rep = rep.detach().float()
                    if rep.dim() > 2:
                        rep = rep.view(rep.size(0), -1)
                    feature_batches.append(rep.cpu().numpy())
                    label_batches.append(y.detach().cpu().numpy())
        finally:
            client.release_model_from_device(model)

        if not feature_batches:
            return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)
        return (
            np.concatenate(feature_batches, axis=0).astype(np.float32),
            np.concatenate(label_batches, axis=0).astype(np.int64),
        )

    def compute_tsne_embedding(self, features):
        from sklearn.manifold import TSNE

        num_samples = int(features.shape[0])
        perplexity = float(getattr(self.args, "tsne_perplexity", 30.0))
        perplexity = min(perplexity, max(2.0, (num_samples - 1) / 3.0))
        seed = int(getattr(self.args, "tsne_seed", 0))
        if seed == 0:
            seed = int(self.random_seed)
        embedding = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            max_iter=int(getattr(self.args, "tsne_max_iter", 1000)),
            random_state=seed,
        ).fit_transform(features)
        return embedding.astype(np.float32)

    def compute_tsne_embedding_with_anchors(self, features, anchor_features):
        if anchor_features is None or anchor_features.shape[0] == 0:
            return self.compute_tsne_embedding(features), np.empty((0, 2), dtype=np.float32)
        if features.ndim != 2 or anchor_features.ndim != 2 or features.shape[1] != anchor_features.shape[1]:
            return self.compute_tsne_embedding(features), np.empty((0, 2), dtype=np.float32)

        combined = np.concatenate([features, anchor_features], axis=0).astype(np.float32)
        embedding = self.compute_tsne_embedding(combined)
        num_features = int(features.shape[0])
        return embedding[:num_features], embedding[num_features:]

    def plot_tsne_panels(self, panels, figure_path, title=None):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        num_panels = len(panels)
        fig_width = max(5.5 * num_panels, 6.0)
        fig, axes = plt.subplots(1, num_panels, figsize=(fig_width, 5.2), squeeze=False)
        axes = axes[0]
        last_scatter = None
        for ax, panel in zip(axes, panels):
            if len(panel) >= 6:
                embedding, labels, client_ids, panel_title, anchor_embedding, anchor_labels = panel[:6]
            else:
                embedding, labels, client_ids, panel_title = panel[:4]
                anchor_embedding = np.empty((0, 2), dtype=np.float32)
                anchor_labels = np.empty((0,), dtype=np.int64)
            last_scatter = ax.scatter(
                embedding[:, 0],
                embedding[:, 1],
                c=labels,
                cmap="turbo",
                s=7,
                alpha=0.78,
                linewidths=0,
                rasterized=True,
            )
            if anchor_embedding is not None and anchor_embedding.shape[0] > 0:
                ax.scatter(
                    anchor_embedding[:, 0],
                    anchor_embedding[:, 1],
                    c=anchor_labels,
                    cmap="turbo",
                    marker="*",
                    s=95,
                    alpha=0.98,
                    edgecolors="black",
                    linewidths=0.75,
                    label="Semantic anchors",
                    zorder=5,
                )
                ax.legend(loc="best", fontsize=8, frameon=True)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_aspect("equal", adjustable="datalim")

        if last_scatter is not None:
            cbar = fig.colorbar(last_scatter, ax=axes.tolist(), shrink=0.78, pad=0.02)
            cbar.set_label("Class")
        fig.savefig(figure_path, dpi=220, bbox_inches="tight")
        plt.close(fig)

    def try_save_tsne_comparison(self, current_path, current_meta, save_dir):
        if self.algorithm not in {"FedSA", "FedSTAR"}:
            return

        other_path, other_meta = self.find_tsne_counterpart(current_path, current_meta, save_dir)
        if other_path is None:
            print("TSNE comparison skipped: counterpart FedSA/FedSTAR features not found yet.")
            return

        current = np.load(current_path, allow_pickle=True)
        other = np.load(other_path, allow_pickle=True)
        current_features = current["features"].astype(np.float32)
        other_features = other["features"].astype(np.float32)
        current_labels = current["labels"].astype(np.int64)
        other_labels = other["labels"].astype(np.int64)
        current_clients = current["clients"].astype(np.int64)
        other_clients = other["clients"].astype(np.int64)
        current_anchor_features = (
            current["anchor_features"].astype(np.float32)
            if "anchor_features" in current.files
            else np.empty((0, current_features.shape[1]), dtype=np.float32)
        )
        other_anchor_features = (
            other["anchor_features"].astype(np.float32)
            if "anchor_features" in other.files
            else np.empty((0, other_features.shape[1]), dtype=np.float32)
        )
        current_anchor_labels = (
            current["anchor_labels"].astype(np.int64)
            if "anchor_labels" in current.files
            else np.empty((0,), dtype=np.int64)
        )
        other_anchor_labels = (
            other["anchor_labels"].astype(np.int64)
            if "anchor_labels" in other.files
            else np.empty((0,), dtype=np.int64)
        )

        current_embedding, current_anchor_embedding = self.compute_tsne_embedding_with_anchors(
            current_features,
            current_anchor_features,
        )
        other_embedding, other_anchor_embedding = self.compute_tsne_embedding_with_anchors(
            other_features,
            other_anchor_features,
        )
        safe_current = self._safe_filename(current_meta.get("run_name", self.algorithm))
        safe_other = self._safe_filename(other_meta.get("run_name", other_meta.get("algorithm", "other")))
        compare_path = os.path.join(save_dir, f"compare-{safe_current}-vs-{safe_other}-{current_meta['split']}.png")
        panels = [
            (
                current_embedding,
                current_labels,
                current_clients,
                self._tsne_title(current_meta),
                current_anchor_embedding,
                current_anchor_labels,
            ),
            (
                other_embedding,
                other_labels,
                other_clients,
                self._tsne_title(other_meta),
                other_anchor_embedding,
                other_anchor_labels,
            ),
        ]
        self.plot_tsne_panels(
            panels,
            compare_path,
            title=None,
        )
        print(f"Saved TSNE comparison: {compare_path}")

    def find_tsne_counterpart(self, current_path, current_meta, save_dir):
        current_algorithm = current_meta.get("algorithm", "")
        target_algorithm = "FedSA" if current_algorithm == "FedSTAR" else "FedSTAR"
        match_keys = [
            "dataset",
            "model_family",
            "feature_dim",
            "split",
            "random_seed",
            "times",
            "normalize",
        ]
        candidates = []
        for path in glob.glob(os.path.join(save_dir, "*-features.npz")):
            if os.path.abspath(path) == os.path.abspath(current_path):
                continue
            meta = self._load_tsne_meta(path)
            if not meta or meta.get("algorithm") != target_algorithm:
                continue
            if all(meta.get(key) == current_meta.get(key) for key in match_keys):
                candidates.append((os.path.getmtime(path), path, meta))
        if not candidates:
            return None, None
        candidates.sort(reverse=True)
        return candidates[0][1], candidates[0][2]

    def _load_tsne_meta(self, path):
        try:
            data = np.load(path, allow_pickle=True)
            raw = data["metadata"]
            if hasattr(raw, "item"):
                raw = raw.item()
            return json.loads(str(raw))
        except Exception:
            return None

    def _tsne_metadata(self, split, num_samples, num_anchors=0):
        return {
            "algorithm": str(self.algorithm),
            "algorithm_label": self._tsne_algorithm_label(),
            "dataset": str(self.dataset),
            "model_family": str(getattr(self.args, "model_family", "")),
            "feature_dim": int(getattr(self.args, "feature_dim", 0)),
            "num_clients": int(getattr(self.args, "num_clients", 0)),
            "num_classes": int(getattr(self.args, "num_classes", 0)),
            "split": str(split),
            "random_seed": int(getattr(self.args, "random_seed", 0)),
            "times": int(getattr(self, "times", 0)),
            "goal": str(getattr(self.args, "goal", "")),
            "run_name": os.path.basename(os.path.normpath(str(self.save_folder_name))),
            "num_samples": int(num_samples),
            "num_anchors": int(num_anchors),
            "max_samples": int(getattr(self.args, "tsne_max_samples", 3000)),
            "max_per_client": int(getattr(self.args, "tsne_max_per_client", 150)),
            "normalize": bool(getattr(self.args, "tsne_normalize", False)),
        }

    def _tsne_algorithm_label(self):
        if self.algorithm == "FedSTAR":
            mode = str(getattr(self.args, "star_anchor_mode", "point"))
            if mode in {"qk", "qwk_diag", "qwk_lowrank", "qwk_full"}:
                tau = float(getattr(self.args, "star_qk_temperature", 0.2))
                if mode == "qwk_lowrank":
                    rank = int(getattr(self.args, "star_qwk_rank", 16))
                    return f"FedSTAR-{mode} tau={tau:g} r={rank}"
                return f"FedSTAR-{mode} tau={tau:g}"
            return f"FedSTAR-{mode}"
        if self.algorithm == "FedSA":
            lam = float(getattr(self.args, "lamda", 0.0))
            mcl = float(getattr(self.args, "margin_contrastive", 0.0))
            cc = float(getattr(self.args, "classifier_calibration", 0.0))
            return f"FedSA lam={lam:g} mcl={mcl:g} cc={cc:g}"
        if self.algorithm == "FedTGP":
            lam = float(getattr(self.args, "lamda", 0.0))
            mode = str(getattr(self.args, "tgp_stats_mode", "exact"))
            return f"FedTGP lam={lam:g} stats={mode}"
        return str(self.algorithm)

    def _tsne_title(self, meta):
        label = meta.get("algorithm_label", meta.get("algorithm", "run"))
        anchor_count = int(meta.get("num_anchors", 0) or 0)
        anchor_part = f" anchors={anchor_count}" if anchor_count > 0 else ""
        return f"{label} | {meta.get('split', 'test')} | n={meta.get('num_samples', 0)}{anchor_part}"

    def _tsne_artifact_stem(self, split):
        run_name = os.path.basename(os.path.normpath(str(self.save_folder_name)))
        if not run_name:
            run_name = f"{self.dataset}-{self.algorithm}-{self.goal}"
        return self._safe_filename(f"{run_name}-{split}-tsne")

    def _safe_filename(self, value):
        value = str(value)
        allowed = []
        for char in value:
            if char.isalnum() or char in {"-", "_", ".", "="}:
                allowed.append(char)
            else:
                allowed.append("_")
        return "".join(allowed).strip("_")

    def test_metrics(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            print(f'Client {c.id}: Acc: {ct*1.0/max(ns, 1)}, AUC: {auc}')
            tot_auc.append(auc*ns)
            num_samples.append(ns)
        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc

    def train_metrics(self):
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl*1.0)
            print(f'Client {c.id}: Loss: {cl*1.0/ns}')

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    # evaluate selected clients
    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()
        # stats_train = self.train_metrics()

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        test_auc = sum(stats[3])*1.0 / sum(stats[1])
        # train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]
        
        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
        
        # if loss == None:
        #     self.rs_train_loss.append(train_loss)
        # else:
        #     loss.append(train_loss)

        # print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accuracy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accuracy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))

    def print_(self, test_acc, test_auc, train_loss):
        print("Average Test Accuracy: {:.4f}".format(test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt is not None and div_value is not None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt is not None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value is not None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True
