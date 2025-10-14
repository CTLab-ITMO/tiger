import json

import torch
from torch.utils.data import DataLoader

from modeling import utils
from modeling.dataloader import BatchProcessor
from modeling.dataset import Dataset
from modeling.loss import IdentityLoss
from modeling.metric import NDCGSemanticMetric, RecallSemanticMetric
from modeling.models import TigerModel
from modeling.utils import parse_args, create_logger, fix_random_seed
from modeling.trainer import Trainer


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
        is_extended=True
    )

    train_sampler, validation_sampler, test_sampler = dataset.get_samplers()

    num_codebooks = config['dataset']['num_codebooks']
    user_ids_count = config['model']['user_ids_count']
    batch_processor = BatchProcessor.create(
        config['dataset']['index_json_path'], num_codebooks, user_ids_count
    )

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

    model = TigerModel(
        embedding_dim=config['model']['embedding_dim'],
        codebook_size=config['model']['codebook_size'],
        sem_id_len=num_codebooks,
        user_ids_count=user_ids_count,
        num_positions=config['model']['num_positions'],
        num_heads=config['model']['num_heads'],
        num_encoder_layers=config['model']['num_encoder_layers'],
        num_decoder_layers=config['model']['num_decoder_layers'],
        dim_feedforward=config['model']['dim_feedforward'],
        num_beams=config['model']['num_beams'],
        num_return_sequences=config['model']['top_k'],
        activation=config['model']['activation'],
        d_kv=config['model']['d_kv'],
        dropout=config['model']['dropout'],
        layer_norm_eps=config['model']['layer_norm_eps'],
        initializer_range=config['model']['initializer_range'],
    ).to(utils.DEVICE)

    loss_function = IdentityLoss(
        predictions_prefix='loss',
        output_prefix='loss'
    )  # Passes through the loss computed inside the model without modification

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['optimizer']['lr'],
    )

    codebook_size = config['model']['codebook_size']
    ranking_metrics = {
        'ndcg@5': NDCGSemanticMetric(5, codebook_size, num_codebooks),
        'ndcg@10': NDCGSemanticMetric(10, codebook_size, num_codebooks),
        'ndcg@20': NDCGSemanticMetric(20, codebook_size, num_codebooks),
        'recall@5': RecallSemanticMetric(5, codebook_size, num_codebooks),
        'recall@10': RecallSemanticMetric(10, codebook_size, num_codebooks),
        'recall@20': RecallSemanticMetric(20, codebook_size, num_codebooks)
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
