import copy
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn

from flcore.clients import clientbase
from flcore.clients.clientsa import clientSA


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(8, 4)
        self.head = nn.Linear(4, 3)

    def forward(self, x):
        return self.head(self.base(x))


def make_client(model, device, resident, batches):
    client = clientSA.__new__(clientSA)
    client.__dict__.update(
        model=model, device=torch.device(device), keep_client_models_on_gpu=resident,
        num_classes=3, semantic_anchors=torch.eye(3, 4, device=device),
        global_protos={i: torch.eye(3, 4)[i] for i in range(3)}, protos=None,
        local_gap=0.0, global_gap=0.0, loss=nn.CrossEntropyLoss(), loss_mse=nn.MSELoss(),
        lamda=10.0, mcl=1.0, cc=0.0, amp_enabled=False, learning_rate=0.01,
        local_epochs=1, train_slow=False, role=f'Client_{int(resident)}',
        save_folder_name='temp_fedsa_residency_test',
        train_time_cost={'num_rounds': 0, 'total_cost': 0.0},
        load_train_data=lambda **kwargs: batches, load_test_data=lambda: batches,
    )
    return client


class FedSAResidencyTest(unittest.TestCase):
    def test_cuda_release_preserves_model_and_clears_gradients(self):
        c = clientSA.__new__(clientSA)
        c.device, c.keep_client_models_on_gpu, c.model = torch.device('cuda'), True, Mock()
        with patch('torch.cuda.empty_cache') as empty:
            c.release_model_from_device()
        c.model.zero_grad.assert_called_once_with(set_to_none=True)
        c.model.to.assert_not_called()
        empty.assert_not_called()

    def test_offload_option_preserves_old_behavior(self):
        c = clientSA.__new__(clientSA)
        c.device, c.keep_client_models_on_gpu, c.model = torch.device('cuda'), False, Mock()
        with patch('torch.cuda.empty_cache') as empty:
            c.release_model_from_device()
        c.model.to.assert_called_once_with('cpu')
        empty.assert_called_once_with()

    def check_phases(self, device):
        torch.manual_seed(17)
        model = TinyModel()
        batches = [(torch.randn(6, 8), torch.arange(6) % 3) for _ in range(2)]
        clients = [make_client(copy.deepcopy(model), device, flag, batches) for flag in (False, True)]
        with patch.dict(clientbase._ITEM_STORE, {}, clear=True), \
                patch.object(clientbase, '_MEMORY_ONLY_ITEMS', True), \
                patch('torch.save', side_effect=AssertionError('unexpected disk write')):
            for _ in range(2):
                results = []
                for c in clients:
                    c.train()
                    evaluation, loss = c.test_metrics(), c.train_metrics()
                    expected = 'cuda' if c.keep_client_models_on_gpu and device == 'cuda' else 'cpu'
                    self.assertEqual(next(c.model.parameters()).device.type, expected)
                    if expected == 'cuda':
                        self.assertTrue(all(p.grad is None for p in c.model.parameters()))
                    self.assertIs(clientbase.load_item(c.role, 'model', c.save_folder_name), c.model)
                    results.append((evaluation, loss, c.protos, c.model.state_dict()))
                self.assertEqual(results[0][:2], results[1][:2])
                for key in results[0][2]:
                    torch.testing.assert_close(results[0][2][key], results[1][2][key], rtol=0, atol=0)
                for key in results[0][3]:
                    torch.testing.assert_close(results[0][3][key].cpu(), results[1][3][key].cpu(), rtol=0, atol=0)

    def test_cpu_train_and_evaluation_match(self):
        self.check_phases('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA device required')
    def test_cuda_train_and_evaluation_match(self):
        self.check_phases('cuda')


if __name__ == '__main__':
    unittest.main()
