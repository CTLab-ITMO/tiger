import copy
import os

import torch

from ..callbacks import MetricCallback, InferenceCallback
from ..utils import create_logger, TensorboardWriter, DEVICE


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
            valid_step=64,
            eval_step=256,
            checkpoint=None
    ):
        self._experiment_name = experiment_name
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.eval_dataloader = eval_dataloader
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.epoch_cnt = epoch_cnt
        self.step_cnt = step_cnt
        self.best_metric = best_metric
        self.epochs_threshold = epochs_threshold

        self.ranking_metrics = ranking_metrics

        tensorboard_writer = TensorboardWriter(self._experiment_name)

        self.metric_callback = MetricCallback(tensorboard_writer=tensorboard_writer, on_step=1, loss_prefix="loss")

        self.validation_callback = InferenceCallback(
            tensorboard_writer=tensorboard_writer,
            step_name="validation",
            model=model,
            dataloader=validation_dataloader,
            on_step=valid_step,
            metrics=ranking_metrics,
            pred_prefix="predictions",
            labels_prefix="labels"
        )

        self.eval_callback = InferenceCallback(
            tensorboard_writer=tensorboard_writer,
            step_name="eval",
            model=model,
            dataloader=eval_dataloader,
            on_step=eval_step,
            metrics=ranking_metrics,
            pred_prefix="predictions",
            labels_prefix="labels"
        )

        self.logger = create_logger(name=__name__)

        if checkpoint is not None:
            checkpoint_path = os.path.join('../checkpoints', f'{checkpoint}.pth')
            model.load_state_dict(torch.load(checkpoint_path))

    def train(self):
        step_num = 0
        epoch_num = 0
        current_metric = 0
        best_epoch = 0
        best_checkpoint = None

        self.logger.debug('Start training...')

        while (self.epoch_cnt is None or epoch_num < self.epoch_cnt) and (
                self.step_cnt is None or step_num < self.step_cnt):
            if best_epoch + self.epochs_threshold < epoch_num:
                self.logger.debug(
                    'There is no progress during {} epochs. Finish training'.format(self.epochs_threshold))
                break

            self.logger.debug(f'Start epoch {epoch_num}')
            for step, batch in enumerate(self.train_dataloader):
                batch_ = batch

                self.model.train()

                for key, values in batch_.items():
                    batch_[key] = batch_[key].to(DEVICE)

                batch_.update(self.model(batch_))
                loss = self.loss_function(batch_)

                self.optimizer.step(loss)

                self.metric_callback(batch_, step_num)
                self.validation_callback(batch_, step_num)
                self.eval_callback(batch_, step_num)

                step_num += 1

                if self.best_metric is None:
                    best_checkpoint = copy.deepcopy(self.model.state_dict())
                    best_epoch = epoch_num
                elif (best_checkpoint is None
                      or self.best_metric in batch_ and current_metric <= batch_[self.best_metric]):
                    current_metric = batch_[self.best_metric]
                    best_checkpoint = copy.deepcopy(self.model.state_dict())
                    best_epoch = epoch_num

            epoch_num += 1
        self.logger.debug('Training procedure has been finished!')
        return best_checkpoint

    def save(self):
        self.logger.debug('Saving model...')
        checkpoint_path = f'../checkpoints/{self._experiment_name}_final_state.pth'
        torch.save(self.model.state_dict(), checkpoint_path)
        self.logger.debug('Saved model as {}'.format(checkpoint_path))
