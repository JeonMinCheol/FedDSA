import copy
import torch
import torch.nn as nn
import numpy as np
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data
from flcore.trainmodel.models import BaseHeadSplit, build_model_from_spec


_ITEM_STORE = {}
_MEMORY_ONLY_ITEMS = os.getenv("HTFLLIB_MEMORY_ONLY_ITEMS", "1").lower() not in {"0", "false", "no"}


def _item_file(role, item_name, item_path=None):
    return os.path.abspath(os.path.join(item_path, role + "_" + item_name + ".pt"))


def _is_temp_path(item_path):
    if item_path is None:
        return False
    base = os.path.basename(os.path.abspath(item_path))
    return base == "temp" or base.startswith("temp_")


def _use_memory_store(item_path):
    return _MEMORY_ONLY_ITEMS and _is_temp_path(item_path)


class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        self.random_seed = int(getattr(args, "random_seed", 0))
        self.rng = np.random.RandomState(self.random_seed + int(id))
        torch.manual_seed(self.random_seed)
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer
        self.role = 'Client_' + str(self.id)
        self.save_folder_name = args.save_folder_name_full

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs
        self.few_shot = args.few_shot
        self.current_round = 0
        self.cache_client_data = not bool(getattr(args, "no_client_data_cache", False))
        self.persist_client_models = bool(getattr(args, "persist_client_models", False))
        self._client_data_cache = {}

        model_file = _item_file(self.role, 'model', self.save_folder_name)
        has_initial_model = model_file in _ITEM_STORE or (not _is_temp_path(self.save_folder_name) and os.path.exists(model_file))
        semantic_model_algorithms = {"FedDSA"}
        memory_model_algorithms = {"FedDSA", "FedProto"}
        init_device = "cpu" if args.algorithm in semantic_model_algorithms else self.device
        if _is_temp_path(self.save_folder_name) or args.save_folder_name == 'temp' or 'temp' not in args.save_folder_name or not has_initial_model:
            if args.models_folder_name:
                model = load_item(self.role, 'model', args.models_folder_name).to(init_device)
                print('load pre-trained model from', args.models_folder_name)
            elif args.algorithm in semantic_model_algorithms:
                model_spec = args.models[self.id % len(args.models)]
                model = build_model_from_spec(model_spec, args).to(init_device)
            else:
                model = BaseHeadSplit(args, self.id).to(init_device)
            save_item(model, self.role, 'model', self.save_folder_name)
            if args.algorithm in memory_model_algorithms:
                self.model = model
        elif args.algorithm in memory_model_algorithms:
            self.model = load_item(self.role, 'model', self.save_folder_name)

        self.train_slow = kwargs['train_slow']
        self.send_slow = kwargs['send_slow']
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        self.loss = nn.CrossEntropyLoss()

    def set_round(self, round_idx):
        self.current_round = int(round_idx)

    def _read_client_data_cached(self, is_train=True):
        cache_key = (bool(is_train), int(self.few_shot) if is_train else 0)
        if not self.cache_client_data:
            return read_client_data(self.dataset, self.id, is_train=is_train, few_shot=self.few_shot)
        if cache_key not in self._client_data_cache:
            self._client_data_cache[cache_key] = read_client_data(
                self.dataset,
                self.id,
                is_train=is_train,
                few_shot=self.few_shot,
            )
        return self._client_data_cache[cache_key]

    def load_train_data(self, batch_size=None, drop_last=True):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = self._read_client_data_cached(is_train=True)
        self.current_train_samples = len(train_data)
        return DataLoader(
            train_data,
            batch_size,
            drop_last=drop_last,
            shuffle=bool(train_data),
            pin_memory=str(self.device).startswith("cuda"),
        )

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = self._read_client_data_cached(is_train=False)
        self.current_test_samples = len(test_data)
        return DataLoader(
            test_data,
            batch_size,
            drop_last=False,
            shuffle=False,
            pin_memory=str(self.device).startswith("cuda"),
        )

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def release_model_from_device(self, model=None):
        if model is None and hasattr(self, "model"):
            model = self.model
        if model is not None and str(self.device).startswith("cuda"):
            model.to("cpu")
            torch.cuda.empty_cache()

    def test_metrics(self):
        testloader = self.load_test_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        # model.to(self.device)
        model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        
        with torch.no_grad():
            for x, y in testloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        
        return test_acc, test_num, auc

    def train_metrics(self):
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        # model.to(self.device)
        model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num

    # def get_next_train_batch(self):
    #     try:
    #         # Samples a new batch for persionalizing
    #         (x, y) = next(self.iter_trainloader)
    #     except StopIteration:
    #         # restart the generator if the previous generator is exhausted.
    #         self.iter_trainloader = iter(self.trainloader)
    #         (x, y) = next(self.iter_trainloader)

    #     if type(x) == type([]):
    #         x = x[0]
    #     x = x.to(self.device)
    #     y = y.to(self.device)

    #     return x, y


def save_item(item, role, item_name, item_path=None):
    item_file = _item_file(role, item_name, item_path)
    if _use_memory_store(item_path):
        _ITEM_STORE[item_file] = item
        return
    if not os.path.exists(item_path):
        os.makedirs(item_path)
    torch.save(item, item_file)

def item_exists(role, item_name, item_path=None):
    item_file = _item_file(role, item_name, item_path)
    return item_file in _ITEM_STORE or (
        not _use_memory_store(item_path) and os.path.isfile(item_file)
    )


def load_item(role, item_name, item_path=None):
    item_file = _item_file(role, item_name, item_path)
    if item_file in _ITEM_STORE:
        return _ITEM_STORE[item_file]
    if _use_memory_store(item_path):
        return None

    try:
        item = torch.load(item_file, weights_only=False)
        if _use_memory_store(item_path):
            _ITEM_STORE[item_file] = item
        return item
    except FileNotFoundError:
        print(role, item_name, 'Not Found')
        return None
    except (EOFError, RuntimeError) as exc:
        print(role, item_name, f'could not be loaded from {item_file}: {exc}')
        return None
