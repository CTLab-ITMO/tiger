import logging

logger = logging.getLogger(__name__)


class TorchDataloader(config_name='torch'):

    def __init__(self, dataloader):
        self._dataloader = dataloader

    def __iter__(self):
        return iter(self._dataloader)

    def __len__(self):
        return len(self._dataloader)
