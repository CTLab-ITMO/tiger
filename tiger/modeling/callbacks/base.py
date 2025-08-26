from metric import BaseMetric, StatefullMetric

import utils
from utils import MetaParent, create_logger

import numpy as np
import os
import torch
from pathlib import Path

logger = create_logger(name=__name__)


class BaseCallback(metaclass=MetaParent):

    def __init__(
            self,
            model,
            train_dataloader,
            validation_dataloader,
            eval_dataloader,
            optimizer
    ):
        self._model = model
        self._train_dataloader = train_dataloader
        self._validation_dataloader = validation_dataloader
        self._eval_dataloader = eval_dataloader
        self._optimizer = optimizer

    def __call__(self, inputs, step_num):
        raise NotImplementedError


class MetricCallback(BaseCallback, config_name='metric'):

    def __init__(
            self,
            model,
            train_dataloader,
            validation_dataloader,
            eval_dataloader,
            optimizer,
            on_step,
            metrics,
            loss_prefix
    ):
        super().__init__(
            model,
            train_dataloader,
            validation_dataloader,
            eval_dataloader,
            optimizer
        )
        self._on_step = on_step
        self._loss_prefix = loss_prefix
        self._metrics = metrics if metrics is not None else {}

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            model=kwargs['model'],
            train_dataloader=kwargs['train_dataloader'],
            validation_dataloader=kwargs['validation_dataloader'],
            eval_dataloader=kwargs['eval_dataloader'],
            optimizer=kwargs['optimizer'],
            on_step=config['on_step'],
            metrics=config.get('metrics', None),
            loss_prefix=config['loss_prefix']
        )

    def __call__(self, inputs, step_num):
        if step_num % self._on_step == 0:
            for metric_name, metric_function in self._metrics.items():
                metric_value = metric_function(
                    ground_truth=inputs[self._model.schema['ground_truth_prefix']],
                    predictions=inputs[self._model.schema['predictions_prefix']]
                )

                utils.tensorboards.GLOBAL_TENSORBOARD_WRITER.add_scalar(
                    'train/{}'.format(metric_name),
                    metric_value,
                    step_num
                )

            utils.tensorboards.GLOBAL_TENSORBOARD_WRITER.add_scalar(
                'train/{}'.format(self._loss_prefix),
                inputs[self._loss_prefix],
                step_num
            )
            utils.tensorboards.GLOBAL_TENSORBOARD_WRITER.flush()


class CheckpointCallback(BaseCallback, config_name='checkpoint'):

    def __init__(
            self,
            model,
            train_dataloader,
            validation_dataloader,
            eval_dataloader,
            optimizer,
            on_step,
            save_path,
            model_name
    ):
        super().__init__(
            model,
            train_dataloader,
            validation_dataloader,
            eval_dataloader,
            optimizer
        )
        self._on_step = on_step
        self._save_path = Path(os.path.join(save_path, model_name))
        if self._save_path.exists():
            logger.warning('Checkpoint path `{}` is already exists!'.format(self._save_path))
        else:
            self._save_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            model=kwargs['model'],
            train_dataloader=kwargs['train_dataloader'],
            validation_dataloader=kwargs['validation_dataloader'],
            eval_dataloader=kwargs['eval_dataloader'],
            optimizer=kwargs['optimizer'],
            on_step=config['on_step'],
            save_path=config['save_path'],
            model_name=config['model_name']
        )

    def __call__(self, inputs, step_num):
        if step_num % self._on_step == 0:
            logger.debug('Saving model state on step {}...'.format(step_num))
            torch.save(
                {
                    'step_num': step_num,
                    'model_state_dict': self._model.state_dict(),
                    'optimizer_state_dict': self._optimizer.state_dict(),
                },
                os.path.join(self._save_path, 'checkpoint_{}.pth'.format(step_num))
            )
            logger.debug('Saving done!')


class InferenceCallback(BaseCallback):

    def __init__(
            self,
            model,
            train_dataloader,
            validation_dataloader,
            eval_dataloader,
            optimizer,
            on_step,
            pred_prefix,
            labels_prefix,
            metrics=None,
            loss_prefix=None,
    ):
        super().__init__(
            model,
            train_dataloader,
            validation_dataloader,
            eval_dataloader,
            optimizer
        )
        self._on_step = on_step
        self._metrics = metrics if metrics is not None else {}
        self._pred_prefix = pred_prefix
        self._labels_prefix = labels_prefix
        self._loss_prefix = loss_prefix

    @classmethod
    def create_from_config(cls, config, **kwargs):
        metrics = {
            metric_name: BaseMetric.create_from_config(metric_cfg, **kwargs)
            for metric_name, metric_cfg in config['metrics'].items()
        }

        return cls(
            model=kwargs['model'],
            train_dataloader=kwargs['train_dataloader'],
            validation_dataloader=kwargs['validation_dataloader'],
            eval_dataloader=kwargs['eval_dataloader'],
            optimizer=kwargs['optimizer'],
            on_step=config['on_step'],
            metrics=metrics,
            pred_prefix=config['pred_prefix'],
            labels_prefix=config['labels_prefix']
        )

    def __call__(self, inputs, step_num):
        if step_num % self._on_step == 0:  # TODO Add time monitoring
            logger.debug(f'Running {self._get_name()} on step {step_num}...')
            running_params = {}
            for metric_name, metric_function in self._metrics.items():
                running_params[metric_name] = []
            if self._loss_prefix is not None:
                running_params[self._loss_prefix] = []

            self._model.eval()
            with torch.no_grad():
                for batch in self._get_dataloader():

                    for key, value in batch.items():
                        batch[key] = value.to(utils.DEVICE)

                    batch.update(self._model(batch))

                    # for key, values in batch.items():
                    #     batch[key] = values.cpu()

                    for metric_name, metric_function in self._metrics.items():
                        running_params[metric_name].extend(metric_function(
                            inputs=batch,
                            pred_prefix=self._pred_prefix,
                            labels_prefix=self._labels_prefix,
                        ))

                    if self._loss_prefix is not None:
                        running_params[self._loss_prefix] += batch[self._loss_prefix].item()
            
            for metric_name, metric_function in self._metrics.items():
                if isinstance(metric_function, StatefullMetric):
                    running_params[metric_name] = metric_function.reduce(running_params[metric_name])

            for label, value in running_params.items():
                inputs[f'{self._get_name()}/{label}'] = np.mean(value)
                utils.tensorboards.GLOBAL_TENSORBOARD_WRITER.add_scalar(
                    f'{self._get_name()}/{label}',
                    np.mean(value),
                    step_num
                )
            utils.tensorboards.GLOBAL_TENSORBOARD_WRITER.flush()

            logger.debug(f'Running {self._get_name()} on step {step_num} is done!')
        
    def _get_name(self):
        return self.config_name

    def _get_dataloader(self):
        raise NotImplementedError


class ValidationCallback(InferenceCallback, config_name='validation'):

    @classmethod
    def create_from_config(cls, config, **kwargs):
        metrics = {
            metric_name: BaseMetric.create_from_config(metric_cfg, **kwargs)
            for metric_name, metric_cfg in config['metrics'].items()
        }

        return cls(
            model=kwargs['model'],
            train_dataloader=kwargs['train_dataloader'],
            validation_dataloader=kwargs['validation_dataloader'],
            eval_dataloader=kwargs['eval_dataloader'],
            optimizer=kwargs['optimizer'],
            on_step=config['on_step'],
            metrics=metrics,
            pred_prefix=config['pred_prefix'],
            labels_prefix=config['labels_prefix']
        )

    def _get_dataloader(self):
        return self._validation_dataloader


class EvalCallback(InferenceCallback, config_name='eval'):

    @classmethod
    def create_from_config(cls, config, **kwargs):
        metrics = {
            metric_name: BaseMetric.create_from_config(metric_cfg, **kwargs)
            for metric_name, metric_cfg in config['metrics'].items()
        }

        return cls(
            model=kwargs['model'],
            train_dataloader=kwargs['train_dataloader'],
            validation_dataloader=kwargs['validation_dataloader'],
            eval_dataloader=kwargs['eval_dataloader'],
            optimizer=kwargs['optimizer'],
            on_step=config['on_step'],
            metrics=metrics,
            pred_prefix=config['pred_prefix'],
            labels_prefix=config['labels_prefix']
        )

    def _get_dataloader(self):
        return self._eval_dataloader


class CompositeCallback(BaseCallback, config_name='composite'):
    def __init__(
            self,
            model,
            train_dataloader,
            validation_dataloader,
            eval_dataloader,
            optimizer,
            callbacks
    ):
        super().__init__(
            model,
            train_dataloader,
            validation_dataloader,
            eval_dataloader,
            optimizer
        )
        self._callbacks = callbacks

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            model=kwargs['model'],
            train_dataloader=kwargs['train_dataloader'],
            validation_dataloader=kwargs['validation_dataloader'],
            eval_dataloader=kwargs['eval_dataloader'],
            optimizer=kwargs['optimizer'],
            callbacks=[
                BaseCallback.create_from_config(cfg, **kwargs)
                for cfg in config['callbacks']
            ]
        )

    def __call__(self, inputs, step_num):
        for callback in self._callbacks:
            callback(inputs, step_num)
