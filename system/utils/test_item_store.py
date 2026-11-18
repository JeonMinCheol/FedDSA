"""In-memory item store must behave like the disk store for server items: each load is an independent copy."""
import torch.nn as nn

from flcore.clients.clientbase import load_item, save_item


def test_server_items_are_copies():
    path = "/tmp/temp_item_store_test"
    save_item(nn.Linear(2, 2), "Server", "global_model", path)
    a, b = load_item("Server", "global_model", path), load_item("Server", "global_model", path)
    assert a is not b
    a.weight.data.add_(1.0)  # one client training its copy must not change what the next client receives
    assert not (load_item("Server", "global_model", path).weight == a.weight).any()
    own = nn.Linear(2, 2)
    save_item(own, "Client_0", "model", path)
    assert load_item("Client_0", "model", path) is own  # a client's own items keep the fast in-place path


if __name__ == "__main__":
    test_server_items_are_copies()
    print("ok")
