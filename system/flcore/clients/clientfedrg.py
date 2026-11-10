"""FedRG local CE + personalized representation alignment."""

import time

import torch

from flcore.clients.clientbase import save_item
from flcore.clients.clientprototype import (
    PrototypeClient, optional_item, move_input, prototype_mse, singleton_batchnorm,
)


class clientRG(PrototypeClient):
    def train(self):
        start = time.time()
        model = self.load_model().train()
        reps = optional_item(self.role, 'reps', self.save_folder_name, self.device) or {}
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        for _ in range(self.local_epochs):
            for x, y in self.load_train_data():
                x, y = move_input(x, self.device), y.to(self.device)
                with singleton_batchnorm(model, len(y)):
                    features = model.base(x)
                    loss = self.loss(model.head(features), y)
                if reps:
                    loss = loss + prototype_mse(features, y, reps)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        protos = self.collect_prototypes(model)
        save_item(model, self.role, 'model', self.save_folder_name)
        save_item(protos, self.role, 'protos', self.save_folder_name)
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start

    def set_rep(self, reps):
        save_item(reps, self.role, 'reps', self.save_folder_name)
