import copy
import json
import os

import torch
from torch.utils.data import DataLoader

from modeling import utils
from modeling.callbacks import CompositeCallback
from modeling.callbacks.base import MetricCallback, ValidationCallback, EvalCallback
from modeling.dataloader.base import TorchDataloader
from modeling.dataloader.batch_processors import BasicBatchProcessor
from modeling.dataset.base import ScientificDataset
from modeling.loss import SASRecLoss, CompositeLoss
from modeling.metric.base import NDCGMetric, RecallMetric, CoverageMetric
from modeling.models import SasRecModel
from modeling.optimizer.base import BasicOptimizer, OPTIMIZERS
from modeling.utils import parse_args, create_logger, fix_random_seed, tensorboards

logger = create_logger(name=__name__)
seed_val = 42


def create_ranking_metrics(dataset):
    return {
        "ndcg@5": NDCGMetric(5),
        "ndcg@10": NDCGMetric(10),
        "ndcg@20": NDCGMetric(20),
        "recall@5": RecallMetric(5),
        "recall@10": RecallMetric(10),
        "recall@20": RecallMetric(20),
        "coverage@5": CoverageMetric(5, dataset.meta['num_items']),
        "coverage@10": CoverageMetric(10, dataset.meta['num_items']),
        "coverage@20": CoverageMetric(20, dataset.meta['num_items'])
    }


def train(dataloader, model, optimizer, loss_function, callback, epoch_cnt=None, step_cnt=None, best_metric=None,
          epochs_threshold=None):
    step_num = 0
    epoch_num = 0
    current_metric = 0

    best_epoch = 0
    best_checkpoint = None

    logger.debug('Start training...')

    while (epoch_cnt is None or epoch_num < epoch_cnt) and (step_cnt is None or step_num < step_cnt):
        if best_epoch + epochs_threshold < epoch_num:
            logger.debug('There is no progress during {} epochs. Finish training'.format(epochs_threshold))
            break

        logger.debug(f'Start epoch {epoch_num}')
        for step, batch in enumerate(dataloader):
            batch_ = batch

            model.train()

            for key, values in batch_.items():
                batch_[key] = batch_[key].to(utils.DEVICE)

            batch_.update(model(batch_))
            loss = loss_function(batch_)

            optimizer.step(loss)
            callback(batch_, step_num)
            step_num += 1

            if best_metric is None:
                # Take the last model
                best_checkpoint = copy.deepcopy(model.state_dict())
                best_epoch = epoch_num
            elif best_checkpoint is None or best_metric in batch_ and current_metric <= batch_[best_metric]:
                # If it is the first checkpoint, or it is the best checkpoint
                current_metric = batch_[best_metric]
                best_checkpoint = copy.deepcopy(model.state_dict())
                best_epoch = epoch_num

        epoch_num += 1
    logger.debug('Training procedure has been finished!')
    return best_checkpoint


def main():
    fix_random_seed(seed_val)
    config = parse_args()

    utils.GLOBAL_TENSORBOARD_WRITER = tensorboards.TensorboardWriter(config['experiment_name'])

    logger.debug('Training config: \n{}'.format(json.dumps(config, indent=2)))
    logger.debug('Current DEVICE: {}'.format(utils.DEVICE))

    dataset = ScientificDataset.create_from_config(config['dataset'])

    train_sampler, validation_sampler, test_sampler = dataset.get_samplers()

    train_dataloader = TorchDataloader(
        dataloader=DataLoader(
            dataset=train_sampler,
            batch_size=config["dataloader_batch_size"]["train"],
            drop_last=True,
            shuffle=True,
            collate_fn=BasicBatchProcessor()
        )
    )

    validation_dataloader = TorchDataloader(
        dataloader=DataLoader(
            dataset=validation_sampler,
            batch_size=config["dataloader_batch_size"]["validation"],
            drop_last=False,
            shuffle=False,
            collate_fn=BasicBatchProcessor()
        )
    )

    eval_dataloader = TorchDataloader(
        dataloader=DataLoader(
            dataset=test_sampler,
            batch_size=256,
            drop_last=False,
            shuffle=False,
            collate_fn=BasicBatchProcessor()
        )
    )

    model = SasRecModel.create_from_config(config['model'], **dataset.meta).to(utils.DEVICE)

    if 'checkpoint' in config:
        checkpoint_path = os.path.join('../checkpoints', f'{config["checkpoint"]}.pth')
        logger.debug('Loading checkpoint from {}'.format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        print(checkpoint.keys())
        model.load_state_dict(checkpoint)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params}")

    loss_function = CompositeLoss(
        losses=[
            SASRecLoss(
                positive_prefix="positive_scores",
                negative_prefix="negative_scores",
                output_prefix="downstream_loss"
            )
        ],
        weights=[1.0],
        output_prefix="loss"
    )

    optimizer_cfg = copy.deepcopy(config['optimizer'])
    optimizer = BasicOptimizer(
        model=model,
        optimizer=OPTIMIZERS[optimizer_cfg.pop('type')](
            model.parameters(),
            **optimizer_cfg
        ),
        scheduler=None,
        clip_grad_threshold=config.get('clip_grad_threshold', None)
    )

    callback = CompositeCallback(
        model=model,
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        eval_dataloader=eval_dataloader,
        optimizer=optimizer,
        callbacks=[
            # Метрики для тренировки (логирование каждый шаг)
            MetricCallback(
                model=model,
                train_dataloader=train_dataloader,
                validation_dataloader=validation_dataloader,
                eval_dataloader=eval_dataloader,
                optimizer=optimizer,
                on_step=1,
                metrics=None,  # Только loss
                loss_prefix="loss"
            ),

            # Валидация каждые 64 шага
            ValidationCallback(
                model=model,
                train_dataloader=train_dataloader,
                validation_dataloader=validation_dataloader,
                eval_dataloader=eval_dataloader,
                optimizer=optimizer,
                on_step=64,
                metrics=create_ranking_metrics(dataset),
                pred_prefix="predictions",
                labels_prefix="labels",
                loss_prefix=None
            ),

            # Финальная оценка каждые 256 шагов
            EvalCallback(
                model=model,
                train_dataloader=train_dataloader,
                validation_dataloader=validation_dataloader,
                eval_dataloader=eval_dataloader,
                optimizer=optimizer,
                on_step=256,
                metrics=create_ranking_metrics(dataset),
                pred_prefix="predictions",
                labels_prefix="labels",
                loss_prefix=None
            )
        ]
    )

    # TODO add verbose option for all callbacks, multiple optimizer options (???)
    # TODO create pre/post callbacks
    logger.debug('Everything is ready for training process!')

    # Train process
    _ = train(
        dataloader=train_dataloader,
        model=model,
        optimizer=optimizer,
        loss_function=loss_function,
        callback=callback,
        epoch_cnt=config.get('train_epochs_num'),
        step_cnt=config.get('train_steps_num'),
        best_metric=config.get('best_metric'),
        epochs_threshold=config.get('early_stopping_threshold', 40)
    )

    logger.debug('Saving model...')
    ckpt_dir = f"../checkpoints/{config['experiment_name']}"
    os.makedirs(ckpt_dir, exist_ok=True)

    checkpoint_path = '{}/{}_final_state.pth'.format(
        ckpt_dir,
        config['experiment_name']
    )
    torch.save(model.state_dict(), checkpoint_path)
    logger.debug('Saved model as {}'.format(checkpoint_path))

    item_embeds_path = '{}/{}_item_embeddings.pt'.format(
        ckpt_dir,
        config['experiment_name']
    )
    utils.save_sasrec_embeds(model, item_embeds_path)
    logger.debug('Saved item embeddings to {}'.format(item_embeds_path))


if __name__ == '__main__':
    main()