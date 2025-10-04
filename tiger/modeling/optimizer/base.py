import torch

OPTIMIZERS = {
    'sgd': torch.optim.SGD,
    'adam': torch.optim.Adam,
    'adamw': torch.optim.AdamW
}

class BaseOptimizer:
    pass

class BasicOptimizer(BaseOptimizer):
    def __init__(self, model, optimizer, clip_grad_threshold=None):
        self._model = model
        self._optimizer = optimizer
        self._clip_grad_threshold = clip_grad_threshold

    def step(self, loss):
        self._optimizer.zero_grad()
        loss.backward()

        if self._clip_grad_threshold is not None:
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._clip_grad_threshold)

        self._optimizer.step()

    def state_dict(self):
        return {'optimizer': self._optimizer.state_dict()}
