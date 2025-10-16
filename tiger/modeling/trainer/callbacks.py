import numpy as np
import torch

from modeling.utils import create_logger, DEVICE

LOGGER = create_logger(name=__name__)


class MetricCallback:
    def __init__(self, tensorboard_writer, on_step, loss_prefix):
        self._tensorboard_writer = tensorboard_writer
        self._on_step = on_step
        self._loss_prefix = loss_prefix

    def __call__(self, inputs, step_num):
        if step_num % self._on_step == 0:
            self._tensorboard_writer.add_scalar(
                'train/{}'.format(self._loss_prefix),
                inputs[self._loss_prefix],
                step_num
            )
            self._tensorboard_writer.flush()


class InferenceCallback:
    def __init__(
            self,
            tensorboard_writer,
            step_name,
            model,
            dataloader,
            on_step,
            pred_prefix,
            labels_prefix,
            metrics=None,
    ):
        self._tensorboard_writer = tensorboard_writer
        self._step_name = step_name
        self._model = model
        self._dataloader = dataloader
        self._on_step = on_step
        self._metrics = metrics if metrics is not None else {}
        self._pred_prefix = pred_prefix
        self._labels_prefix = labels_prefix

    def __call__(self, inputs, step_num):
        if step_num % self._on_step == 0:
            LOGGER.debug(f'Running {self._step_name} on step {step_num}...')
            running_params = {}
            for metric_name, metric_function in self._metrics.items():
                running_params[metric_name] = []

            self._model.eval()
            with torch.inference_mode():
                for batch in self._dataloader:
                    for key, value in batch.items():
                        batch[key] = value.to(DEVICE)

                    batch.update(self._model(batch))

                    for metric_name, metric_function in self._metrics.items():
                        running_params[metric_name].extend(metric_function(
                            inputs=batch,
                            pred_prefix=self._pred_prefix,
                            labels_prefix=self._labels_prefix,
                        ))

            for label, value in running_params.items():
                inputs[f'{self._step_name}/{label}'] = np.mean(value)
                self._tensorboard_writer.add_scalar(
                    f'{self._step_name}/{label}',
                    np.mean(value),
                    step_num
                )
            self._tensorboard_writer.flush()

            LOGGER.debug(f'Running {self._step_name} on step {step_num} is done!')
