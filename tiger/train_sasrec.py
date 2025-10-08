import copy
import json
import os

import torch
from torch.utils.data import DataLoader

from modeling import utils
from modeling.callbacks import MetricCallback, InferenceCallback
from modeling.dataloader import BasicBatchProcessor
from modeling.dataset import ScientificDataset
from modeling.loss import SASRecLoss
from modeling.metric import NDCGMetric, RecallMetric, CoverageMetric
from modeling.models import SasRecModel
from modeling.optimizer import BasicOptimizer
from modeling.utils import parse_args, create_logger, fix_random_seed, TensorboardWriter

logger = create_logger(name=__name__)
seed_val = 42


def create_ranking_metrics(num_items):
    return {
        "ndcg@5": NDCGMetric(5),
        "ndcg@10": NDCGMetric(10),
        "ndcg@20": NDCGMetric(20),
        "recall@5": RecallMetric(5),
        "recall@10": RecallMetric(10),
        "recall@20": RecallMetric(20),
        "coverage@5": CoverageMetric(5, num_items),
        "coverage@10": CoverageMetric(10, num_items),
        "coverage@20": CoverageMetric(20, num_items)
    }


def train(dataloader, model, metric_callback, validation_callback, eval_callback, optimizer, loss_function,
          epoch_cnt=None, step_cnt=None, best_metric=None,
          epochs_threshold=40):
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

            metric_callback(batch_, step_num)
            validation_callback(batch_, step_num)
            eval_callback(batch_, step_num)

            step_num += 1

            if best_metric is None:
                best_checkpoint = copy.deepcopy(model.state_dict())
                best_epoch = epoch_num
            elif best_checkpoint is None or best_metric in batch_ and current_metric <= batch_[best_metric]:
                current_metric = batch_[best_metric]
                best_checkpoint = copy.deepcopy(model.state_dict())
                best_epoch = epoch_num

        epoch_num += 1
    logger.debug('Training procedure has been finished!')
    return best_checkpoint


def main():
    fix_random_seed(seed_val)
    config = parse_args()

    utils.GLOBAL_TENSORBOARD_WRITER = TensorboardWriter(config['experiment_name'])

    logger.debug('Training config: \n{}'.format(json.dumps(config, indent=2)))
    logger.debug('Current DEVICE: {}'.format(utils.DEVICE))

    dataset = ScientificDataset.create_from_config(config['dataset'])

    dataset_num_items = dataset.num_items
    dataset_max_sequence_length = dataset.max_sequence_length

    train_sampler, validation_sampler, test_sampler = dataset.get_samplers()

    batch_processor = BasicBatchProcessor()

    train_dataloader = DataLoader(
        dataset=train_sampler,
        batch_size=config["dataloader_batch_size"]["train"],
        drop_last=True,
        shuffle=True,
        collate_fn=batch_processor
    )

    validation_dataloader = DataLoader(
        dataset=validation_sampler,
        batch_size=config["dataloader_batch_size"]["validation"],
        drop_last=False,
        shuffle=False,
        collate_fn=batch_processor
    )

    eval_dataloader = DataLoader(
        dataset=test_sampler,
        batch_size=config["dataloader_batch_size"]["validation"],
        drop_last=False,
        shuffle=False,
        collate_fn=batch_processor
    )

    model = SasRecModel(
        num_items=dataset_num_items,
        max_sequence_length=dataset_max_sequence_length,
        embedding_dim=config['model']['embedding_dim'],
        num_heads=config['model']['num_heads'],
        num_layers=config['model']['num_layers'],
        dim_feedforward=config['model']['dim_feedforward'],
        dropout=config['model']['dropout'],
        activation=config['model']['activation'],
        layer_norm_eps=config['model']['layer_norm_eps'],
        initializer_range=config['model']['initializer_range'],
    ).to(utils.DEVICE)

    if 'checkpoint' in config:
        checkpoint_path = os.path.join('../checkpoints', f'{config["checkpoint"]}.pth')
        logger.debug('Loading checkpoint from {}'.format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        print(checkpoint.keys())
        model.load_state_dict(checkpoint)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params}")

    loss_function = SASRecLoss(
        positive_prefix="positive_scores",
        negative_prefix="negative_scores",
        output_prefix="loss"
    )

    optimizer = BasicOptimizer(
        model=model,
        optimizer_config=copy.deepcopy(config['optimizer']),
        clip_grad_threshold=config.get('clip_grad_threshold', None)
    )

    metric_callback = MetricCallback(
        on_step=1,
        loss_prefix="loss"
    )

    ranking_metrics = create_ranking_metrics(dataset_num_items)

    validation_callback = InferenceCallback(
        config_name="validation",
        model=model,
        dataloader=validation_dataloader,
        on_step=64,
        metrics=ranking_metrics,
        pred_prefix="predictions",
        labels_prefix="labels"
    )

    eval_callback = InferenceCallback(
        config_name="eval",
        model=model,
        dataloader=eval_dataloader,
        on_step=256,
        metrics=ranking_metrics,
        pred_prefix="predictions",
        labels_prefix="labels"
    )
    logger.debug('Everything is ready for training process!')

    _ = train(
        dataloader=train_dataloader,
        model=model,
        metric_callback=metric_callback,
        validation_callback=validation_callback,
        eval_callback=eval_callback,
        optimizer=optimizer,
        loss_function=loss_function,
        epoch_cnt=config.get('train_epochs_num'),
        step_cnt=config.get('train_steps_num'),
        best_metric="validation/ndcg@20",
        epochs_threshold=config.get('early_stopping_threshold', 40),
    )

    logger.debug('Saving model...')
    checkpoint_path = '../checkpoints/{}_final_state.pth'.format(config['experiment_name'])
    torch.save(model.state_dict(), checkpoint_path)
    logger.debug('Saved model as {}'.format(checkpoint_path))


if __name__ == '__main__':
    main()
