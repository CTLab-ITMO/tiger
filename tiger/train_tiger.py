import copy
import json
import os

import torch
from torch.utils.data import DataLoader

from modeling import utils
from modeling.callbacks.base import MetricCallback, InferenceCallback
from modeling.dataloader import LetterBatchProcessor
from modeling.dataset import LetterFullDataset
from modeling.loss import IdentityMapLoss, CompositeLoss
from modeling.metric.base import NDCGSemanticMetric, RecallSemanticMetric, CoverageSemanticMetric
from modeling.models import TigerModelT5
from modeling.optimizer import BasicOptimizer
from modeling.utils import parse_args, create_logger, fix_random_seed, tensorboards, DEVICE

logger = create_logger(name=__name__)
seed_val = 42


def create_ranking_metrics(dataset, codebook_size, num_codebooks):
    print("Logs codebook size: {}, num codebooks: {}".format(codebook_size, num_codebooks))
    return {
        "ndcg@5": NDCGSemanticMetric(5, codebook_size, num_codebooks),
        "ndcg@10": NDCGSemanticMetric(10, codebook_size, num_codebooks),
        "ndcg@20": NDCGSemanticMetric(20, codebook_size, num_codebooks),
        "recall@5": RecallSemanticMetric(5, codebook_size, num_codebooks),
        "recall@10": RecallSemanticMetric(10, codebook_size, num_codebooks),
        "recall@20": RecallSemanticMetric(20, codebook_size, num_codebooks),
        "coverage@5": CoverageSemanticMetric(5, codebook_size, num_codebooks, dataset.meta['num_items']),
        "coverage@10": CoverageSemanticMetric(10, codebook_size, num_codebooks, dataset.meta['num_items']),
        "coverage@20": CoverageSemanticMetric(20, codebook_size, num_codebooks, dataset.meta['num_items'])
    }


def train(dataloader, model, metric_callback, validation_callback, eval_callback,
          optimizer, loss_function, epoch_cnt=None, step_cnt=None, best_metric=None):
    step_num = 0
    epoch_num = 0
    current_metric = 0

    epochs_threshold = 40

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

            metric_callback(batch_, step_num)
            validation_callback(batch_, step_num)
            eval_callback(batch_, step_num)

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
    logger.debug('Current DEVICE: {}'.format(DEVICE))

    dataset = LetterFullDataset.create_from_config(config['dataset'])

    train_sampler, validation_sampler, test_sampler = dataset.get_samplers()

    train_dataloader = DataLoader(
        dataset=train_sampler,
        batch_size=config['dataloader']['train']["batch_size"],
        drop_last=True,
        shuffle=True,
        collate_fn=LetterBatchProcessor.create_from_config(config['dataloader']['train']["batch_processor"])
    )

    validation_dataloader = DataLoader(
        dataset=validation_sampler,
        batch_size=config['dataloader']['validation']["batch_size"],
        drop_last=False,
        shuffle=False,
        collate_fn=LetterBatchProcessor.create_from_config(config['dataloader']['validation']["batch_processor"])
    )

    eval_dataloader = DataLoader(
        dataset=test_sampler,
        batch_size=config['dataloader']['validation']["batch_size"],
        drop_last=False,
        shuffle=False,
        collate_fn=LetterBatchProcessor.create_from_config(config['dataloader']['validation']["batch_processor"])
    )

    model = TigerModelT5.create_from_config(config['model'], **dataset.meta).to(DEVICE)

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
            IdentityMapLoss(
                predictions_prefix="loss",
                output_prefix="loss"
            )
        ],
        weights=[1.0],
        output_prefix="loss"
    )

    optimizer = BasicOptimizer(
        model=model,
        optimizer_config=copy.deepcopy(config['optimizer']),
        clip_grad_threshold=config.get('clip_grad_threshold', None)
    )

    # Метрики для тренировки (логирование каждый шаг)
    metric_callback = MetricCallback(
        model=model,
        optimizer=optimizer,
        on_step=1,
        loss_prefix="loss"
    )

    # Валидация каждые 64 шага
    validation_callback = InferenceCallback(
        config_name="validation",
        model=model,
        dataloader=validation_dataloader,
        optimizer=optimizer,
        on_step=64,
        metrics=create_ranking_metrics(dataset, codebook_size=config['model']['codebook_size'],
                                       num_codebooks=4),
        pred_prefix="predictions",
        labels_prefix="labels"
    )

    # Финальная оценка каждые 256 шагов
    eval_callback = InferenceCallback(
        config_name="eval",
        model=model,
        dataloader=eval_dataloader,
        optimizer=optimizer,
        on_step=256,
        metrics=create_ranking_metrics(dataset, codebook_size=config['model']['codebook_size'],
                                       num_codebooks=4),
        pred_prefix="predictions",
        labels_prefix="labels"
    )
    # TODO add verbose option for all callbacks, multiple optimizer options (???)
    # TODO create pre/post callbacks
    logger.debug('Everything is ready for training process!')

    # Train process
    _ = train(
        dataloader=train_dataloader,
        metric_callback=metric_callback,
        validation_callback=validation_callback,
        eval_callback=eval_callback,
        model=model,
        optimizer=optimizer,
        loss_function=loss_function,
        epoch_cnt=config.get('train_epochs_num'),
        step_cnt=config.get('train_steps_num'),
        best_metric=config.get('best_metric')
    )

    logger.debug('Saving model...')
    checkpoint_path = '../checkpoints/{}_final_state.pth'.format(config['experiment_name'])
    torch.save(model.state_dict(), checkpoint_path)
    logger.debug('Saved model as {}'.format(checkpoint_path))


if __name__ == '__main__':
    main()
