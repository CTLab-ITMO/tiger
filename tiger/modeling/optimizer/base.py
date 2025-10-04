import torch

_OPTIMIZERS = {
    'sgd': torch.optim.SGD,
    'adam': torch.optim.Adam,
    'adamw': torch.optim.AdamW
}


class BasicOptimizer:
    def __init__(self, model, optimizer_config, clip_grad_threshold=None):
        self._model = model
        self._optimizer = _OPTIMIZERS[optimizer_config.pop('type')](
            model.parameters(),
            **optimizer_config
        )
        self._clip_grad_threshold = clip_grad_threshold

    def step(self, loss):
        self._optimizer.zero_grad()
        loss.backward()

        if self._clip_grad_threshold is not None:
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._clip_grad_threshold)

        self._optimizer.step()

    def state_dict(self):
        return {'optimizer': self._optimizer.state_dict()}
