import copy
import json

from torch.utils.data import DataLoader

from modeling import utils
from modeling.dataloader import SemanticIdsBatchProcessor
from modeling.dataset import ScientificDataset
from modeling.loss import IdentityLoss
from modeling.metric import NDCGSemanticMetric, RecallSemanticMetric, CoverageSemanticMetric
from modeling.models import TigerModel
from modeling.optimizer import BasicOptimizer
from modeling.utils import parse_args, create_logger, fix_random_seed
from modeling.trainer import Trainer

logger = create_logger(name=__name__)
seed_val = 42


def main():
    fix_random_seed(seed_val)
    config = parse_args()

    logger.debug('Training config: \n{}'.format(json.dumps(config, indent=2)))
    logger.debug('Current DEVICE: {}'.format(utils.DEVICE))

    dataset = ScientificDataset.create(inter_json_path=config['dataset']['inter_json_path'],
                                       max_sequence_length=config['dataset']['max_sequence_length'],
                                       sampler_type=config['dataset']['sampler_type'],
                                       mode='full')

    dataset_num_items = dataset.num_items

    train_sampler, validation_sampler, test_sampler = dataset.get_samplers()

    num_codebooks = config['dataset']['num_codebooks']
    user_ids_count = config['model']['user_ids_count']
    batch_processor = SemanticIdsBatchProcessor.create(
        config['dataset']["index_json_path"], num_codebooks, user_ids_count
    )

    train_dataloader = DataLoader(
        dataset=train_sampler,
        batch_size=config["dataloader"]["train_batch_size"],
        drop_last=True,
        shuffle=True,
        collate_fn=batch_processor
    )

    validation_dataloader = DataLoader(
        dataset=validation_sampler,
        batch_size=config["dataloader"]["validation_batch_size"],
        drop_last=False,
        shuffle=False,
        collate_fn=batch_processor
    )

    eval_dataloader = DataLoader(
        dataset=test_sampler,
        batch_size=config["dataloader"]["validation_batch_size"],
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
        dropout=config['model']['dropout'],
        initializer_range=config['model']['initializer_range']
    ).to(utils.DEVICE)

    loss_function = IdentityLoss(
        predictions_prefix="loss",
        output_prefix="loss"
    )

    optimizer = BasicOptimizer(
        model=model,
        optimizer_config=copy.deepcopy(config['optimizer']),
        clip_grad_threshold=config.get('clip_grad_threshold', None)
    )

    codebook_size = config['model']['codebook_size']
    ranking_metrics = {
        "ndcg@5": NDCGSemanticMetric(5, codebook_size, num_codebooks),
        "ndcg@10": NDCGSemanticMetric(10, codebook_size, num_codebooks),
        "ndcg@20": NDCGSemanticMetric(20, codebook_size, num_codebooks),
        "recall@5": RecallSemanticMetric(5, codebook_size, num_codebooks),
        "recall@10": RecallSemanticMetric(10, codebook_size, num_codebooks),
        "recall@20": RecallSemanticMetric(20, codebook_size, num_codebooks),
        "coverage@5": CoverageSemanticMetric(5, codebook_size, dataset_num_items, num_codebooks),
        "coverage@10": CoverageSemanticMetric(10, codebook_size, dataset_num_items, num_codebooks),
        "coverage@20": CoverageSemanticMetric(20, codebook_size, dataset_num_items, num_codebooks)
    }

    logger.debug('Everything is ready for training process!')

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
        best_metric="validation/ndcg@20",
        epochs_threshold=config.get('early_stopping_threshold', 40),
        valid_step=64,
        eval_step=256,
        checkpoint=config.get('checkpoint', None),
    )

    trainer.train()
    trainer.save()

    logger.debug('Training finished!')


if __name__ == '__main__':
    main()
