import copy
import os

import torch

from modeling.trainer import MetricCallback, InferenceCallback
from modeling.utils import create_logger, TensorboardWriter, DEVICE

LOGGER = create_logger(name=__name__)


class Trainer:
    def __init__(
            self,
            experiment_name,
            train_dataloader,
            validation_dataloader,
            eval_dataloader,
            model,
            optimizer,
            loss_function,
            ranking_metrics,
            epoch_cnt=None,
            step_cnt=None,
            best_metric=None,
            epochs_threshold=40,
            valid_step=256,
            eval_step=256,
            checkpoint_dir='../checkpoints',
            checkpoint=None
    ):
        self._experiment_name = experiment_name
        self._train_dataloader = train_dataloader
        self._validation_dataloader = validation_dataloader
        self._eval_dataloader = eval_dataloader
        self._model = model
        self._optimizer = optimizer
        self._loss_function = loss_function
        self._epoch_cnt = epoch_cnt
        self._step_cnt = step_cnt
        self._best_metric = best_metric
        self._epochs_threshold = epochs_threshold
        self._ranking_metrics = ranking_metrics
        self._checkpoint_dir = checkpoint_dir
        os.makedirs(self._checkpoint_dir, exist_ok=True)

        tensorboard_writer = TensorboardWriter(self._experiment_name)

        self._metric_callback = MetricCallback(tensorboard_writer=tensorboard_writer, on_step=1, loss_prefix="loss")

        self._validation_callback = InferenceCallback(
            tensorboard_writer=tensorboard_writer,
            step_name="validation",
            model=model,
            dataloader=validation_dataloader,
            on_step=valid_step,
            metrics=ranking_metrics,
            pred_prefix="predictions",
            labels_prefix="labels"
        )

        self._eval_callback = InferenceCallback(
            tensorboard_writer=tensorboard_writer,
            step_name="eval",
            model=model,
            dataloader=eval_dataloader,
            on_step=eval_step,
            metrics=ranking_metrics,
            pred_prefix="predictions",
            labels_prefix="labels"
        )

        if checkpoint is not None:
            checkpoint_path = os.path.join(checkpoint_dir, f'{checkpoint}.pth')
            model.load_state_dict(torch.load(checkpoint_path))

    def train(self):
        step_num = 0
        epoch_num = 0
        current_metric = 0
        best_epoch = 0
        best_checkpoint = None

        LOGGER.debug('Start training...')

        while (self._epoch_cnt is None or epoch_num < self._epoch_cnt) and (
                self._step_cnt is None or step_num < self._step_cnt):
            if best_epoch + self._epochs_threshold < epoch_num:
                LOGGER.debug(
                    'There is no progress during {} epochs. Finish training'.format(self._epochs_threshold))
                break

            LOGGER.debug(f'Start epoch {epoch_num}')
            for batch in self._train_dataloader:
                self._model.train()

                for key, values in batch.items():
                    batch[key] = values.to(DEVICE)

                batch.update(self._model(batch))
                loss = self._loss_function(batch)

                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                self._metric_callback(batch, step_num)
                self._validation_callback(batch, step_num)
                self._eval_callback(batch, step_num)

                if self._best_metric is None:  # If no best metric is provided last checkpoint is taken
                    best_checkpoint = copy.deepcopy(self._model.state_dict())
                    best_epoch = epoch_num
                elif (
                    best_checkpoint is None  # If no best checkpoint exists this one is taken
                    or self._best_metric in batch and current_metric <= batch[self._best_metric]  # or if metrics improved compared to previous one
                ):
                    current_metric = batch[self._best_metric]
                    best_checkpoint = copy.deepcopy(self._model.state_dict())
                    best_epoch = epoch_num

                step_num += 1

            epoch_num += 1
        LOGGER.debug('Training procedure has been finished!')
        return best_checkpoint

    def save(self):
        LOGGER.debug('Saving model...')
        checkpoint_path = f'{self._checkpoint_dir}/{self._experiment_name}_final_state.pth'
        torch.save(self._model.state_dict(), checkpoint_path)
        LOGGER.debug('Saved model as {}'.format(checkpoint_path))
