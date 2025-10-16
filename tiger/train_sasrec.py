import json

import torch
from torch.utils.data import DataLoader

from modeling import utils
from modeling.dataloader import BatchProcessor
from modeling.dataset import Dataset
from modeling.loss import BCELoss
from modeling.metric import NDCGMetric, RecallMetric
from modeling.models import SasRecModel
from modeling.trainer import Trainer
from modeling.utils import parse_args, create_logger, fix_random_seed

LOGGER = create_logger(name=__name__)
SEED_VALUE = 42


def main():
    fix_random_seed(SEED_VALUE)
    config = parse_args()

    LOGGER.debug('Training config: \n{}'.format(json.dumps(config, indent=2)))
    LOGGER.debug('Current DEVICE: {}'.format(utils.DEVICE))

    dataset = Dataset.create(
        inter_json_path=config['dataset']['inter_json_path'],
        max_sequence_length=config['dataset']['max_sequence_length'],
        sampler_type=config['dataset']['sampler_type'],
        is_extended=False
    )
    dataset_num_items = dataset.num_items
    dataset_max_sequence_length = dataset.max_sequence_length

    train_sampler, validation_sampler, test_sampler = dataset.get_samplers()

    batch_processor = BatchProcessor()

    train_dataloader = DataLoader(
        dataset=train_sampler,
        batch_size=config['dataloader']['train_batch_size'],
        drop_last=True,
        shuffle=True,
        collate_fn=batch_processor
    )

    validation_dataloader = DataLoader(
        dataset=validation_sampler,
        batch_size=config['dataloader']['validation_batch_size'],
        drop_last=False,
        shuffle=False,
        collate_fn=batch_processor
    )

    eval_dataloader = DataLoader(
        dataset=test_sampler,
        batch_size=config['dataloader']['validation_batch_size'],
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
        activation=utils.get_activation_function(config['model']['activation']),
        topk_k=config['model']['top_k'],
        dropout=config['model']['dropout'],
        layer_norm_eps=config['model']['layer_norm_eps'],
        initializer_range=config['model']['initializer_range']
    ).to(utils.DEVICE)

    loss_function = BCELoss(
        positive_prefix='positive_scores',
        negative_prefix='negative_scores',
        output_prefix='loss'
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['optimizer']['lr'],
    )

    ranking_metrics = {
        'ndcg@5': NDCGMetric(5),
        'ndcg@10': NDCGMetric(10),
        'ndcg@20': NDCGMetric(20),
        'recall@5': RecallMetric(5),
        'recall@10': RecallMetric(10),
        'recall@20': RecallMetric(20)
    }

    LOGGER.debug('Everything is ready for training process!')

    trainer = Trainer(
        experiment_name=config['experiment_name'],
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        eval_dataloader=eval_dataloader,
        model=model,
        optimizer=optimizer,
        loss_function=loss_function,
        ranking_metrics=ranking_metrics,
        epoch_cnt=config.get('train_epochs_num'),
        step_cnt=config.get('train_steps_num'),
        best_metric='validation/ndcg@20',
        epochs_threshold=config.get('early_stopping_threshold', 40),
        valid_step=256,
        eval_step=256,
        checkpoint=config.get('checkpoint', None),
    )

    trainer.train()
    trainer.save()

    LOGGER.debug('Training finished!')


if __name__ == '__main__':
    main()
