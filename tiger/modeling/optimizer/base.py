import torch

OPTIMIZERS = {
    'sgd': torch.optim.SGD,
    'adam': torch.optim.Adam,
    'adamw': torch.optim.AdamW
}

SCHEDULERS = {
    'step': torch.optim.lr_scheduler.StepLR,
    'cyclic': torch.optim.lr_scheduler.CyclicLR
}

class BaseOptimizer:
    pass

class BasicOptimizer(BaseOptimizer, config_name='basic'):
    def __init__(self, model, optimizer, scheduler=None, clip_grad_threshold=None):
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._clip_grad_threshold = clip_grad_threshold

    def step(self, loss):
        self._optimizer.zero_grad()
        loss.backward()

        if self._clip_grad_threshold is not None:
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._clip_grad_threshold)

        self._optimizer.step()
        if self._scheduler is not None:
            self._scheduler.step()

    def state_dict(self):
        state_dict = {'optimizer': self._optimizer.state_dict()}
        if self._scheduler is not None:
            state_dict.update({'scheduler': self._scheduler.state_dict()})
        return state_dict
