import copy
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn

from flcore.clients import clientbase
from flcore.clients.clientfeddsa import clientfeddsa


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(8, 4)
        self.shared_projector = nn.Linear(4, 3)
        self.shared_head = nn.Linear(3, 3)
        self.shared_classifier_mode = "linear"

    def extract_backbone(self, x):
        return self.base(x)

    def shared_representation(self, x):
        return self.shared_projector(self.extract_backbone(x))

    def classify(self, x):
        return self.shared_head(x)


def make_client(model, device, resident, batches):
    client = clientfeddsa.__new__(clientfeddsa)
    client.__dict__.update(
        model=model, device=torch.device(device), keep_client_models_on_gpu=resident,
        num_classes=3, semantic_anchors=torch.eye(3, device=device),
        anchor_mode="single", new_anchor_mode="none",
        scalar_reliability_state=False, dual_disagreement_gate=False,
        reliability_anchor_source="mcl", agreement_weighting=True,
        prototype_utility=True, utility_local_gate=True, utility_weight_normalize=True,
        utility_positive_only=False, utility_temperature=0.5, utility_inner_lr=0.01,
        utility_min_weight=0.25, utility_max_weight=1.5,
        utility_validation_probe=True, class_utility=torch.zeros(3, device=device),
        validation_anchor_grad=None, meta_enabled=False, validation_enabled=True,
        lamda=1.0, mcl=1.0, local_gap=0.0, global_gap=0.0,
        mcl_gap_mode="min", mcl_variant="fedsa", use_amp=False, amp_enabled=False,
        learning_rate=0.01, local_epochs=1, train_slow=False,
        loss=nn.CrossEntropyLoss(),
        role="Client_0", save_folder_name="temp_vram_unit",
        train_time_cost={"num_rounds": 0, "total_cost": 0.0},
        load_train_data=lambda **kwargs: batches, load_test_data=lambda: batches,
        update_gap_stats=lambda: None,
    )
    return client


class ModelResidencyTest(unittest.TestCase):
    def test_resident_release_clears_gradients_without_offloading_or_emptying_cache(self):
        client = clientfeddsa.__new__(clientfeddsa)
        client.device = torch.device("cuda")
        client.keep_client_models_on_gpu = True
        client.model = Mock()
        with patch("torch.cuda.empty_cache") as empty:
            client.release_model_from_device()
        client.model.zero_grad.assert_called_once_with(set_to_none=True)
        client.model.to.assert_not_called()
        empty.assert_not_called()

    def test_offload_option_preserves_existing_release_behavior(self):
        client = clientfeddsa.__new__(clientfeddsa)
        client.device = torch.device("cuda")
        client.keep_client_models_on_gpu = False
        client.model = Mock()
        with patch("torch.cuda.empty_cache") as empty:
            client.release_model_from_device()
        client.model.to.assert_called_once_with("cpu")
        client.model.zero_grad.assert_not_called()
        empty.assert_called_once_with()

    def test_cpu_release_does_not_touch_cuda(self):
        client = clientfeddsa.__new__(clientfeddsa)
        client.device = torch.device("cpu")
        client.keep_client_models_on_gpu = True
        client.model = Mock()
        with patch("torch.cuda.empty_cache") as empty:
            client.release_model_from_device()
        client.model.to.assert_not_called()
        empty.assert_not_called()

    def test_temp_store_retains_the_same_model_without_disk_io(self):
        model = TinyModel()
        with patch.dict(clientbase._ITEM_STORE, {}, clear=True), \
                patch.object(clientbase, "_MEMORY_ONLY_ITEMS", True), \
                patch("torch.save", side_effect=AssertionError("Unexpected disk write")):
            clientbase.save_item(model, "Client_0", "model", "temp_vram_unit")
            self.assertIs(clientbase.load_item("Client_0", "model", "temp_vram_unit"), model)

    def check_phases(self, device):
        torch.manual_seed(7)
        original = TinyModel()
        batches = [(torch.randn(6, 8), torch.arange(6) % 3) for _ in range(3)]
        clients = [make_client(copy.deepcopy(original), device, flag, batches) for flag in [False, True]]
        with patch.dict(clientbase._ITEM_STORE, {}, clear=True), \
                patch.object(clientbase, "_MEMORY_ONLY_ITEMS", True), \
                patch("torch.save", side_effect=AssertionError("Unexpected disk write")):
            for _ in range(2):
                outputs = []
                for client in clients:
                    model = client.model
                    client.train()
                    self.assertIs(clientbase.load_item(client.role, "model", client.save_folder_name), model)
                    client.collect_protos()
                    evaluation = client.test_metrics()
                    train_loss = client.train_metrics()
                    self.assertIs(client.model, model)
                    expected = "cuda" if client.keep_client_models_on_gpu and device == "cuda" else "cpu"
                    self.assertEqual(next(model.parameters()).device.type, expected)
                    if expected == "cuda":
                        self.assertTrue(all(p.grad is None for p in model.parameters()))
                    outputs.append((evaluation, train_loss, client.anchor_utility.clone(),
                                    client.validation_anchor_gain.clone(),
                                    {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}))
                self.assertEqual(outputs[0][:2], outputs[1][:2])
                for left, right in zip(outputs[0][2:4], outputs[1][2:4]):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)
                for key in outputs[0][4]:
                    torch.testing.assert_close(outputs[0][4][key], outputs[1][4][key], rtol=0, atol=0)

    def test_train_probe_proto_and_evaluation_match_on_cpu(self):
        self.check_phases("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device required")
    def test_cuda_residency_preserves_train_probe_proto_and_evaluation_results(self):
        self.check_phases("cuda")


if __name__ == "__main__":
    unittest.main()
