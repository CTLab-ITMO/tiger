import json
import logging
from torch.utils.data import DataLoader
import torch

from datasets import EmbDataset
from models.rqvae import RQVAE
from trainer import Trainer
from utils import create_logger, parse_args, fix_random_seed, create_checkpoint_dir


logger = create_logger(name=__name__)


def main():
    config = parse_args()

    fix_random_seed(config['seed'] if 'seed' in config else 42)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    logger.info(f'Training config: \n{json.dumps(config, indent=2)}')
    logger.info(f'Current device: {device}')
    checkpoint_dir = f"{config['training']['ckpt_dir']}/{config['experiment_name']}"
    create_checkpoint_dir(checkpoint_dir)

    dataset = EmbDataset(config['dataset']['data_path'])
    logger.info(f'Dataset created with dimension: {dataset.dim}')

    dataloader = DataLoader(
        dataset,
        num_workers=config['dataloader']['num_workers'],
        batch_size=config['dataloader']['batch_size'],
        shuffle=config['dataloader']['shuffle'],
        pin_memory=config['dataloader']['pin_memory']
    )
    logger.info(f'DataLoader created with batch size: {config["dataloader"]["batch_size"]}')

    cf_emb = torch.load(config['model']['cf_emb_path']).squeeze().detach().numpy()

    model = RQVAE(
        in_dim=dataset.dim,
        num_emb_list=config['model']['num_emb_list'],
        e_dim=config['model']['e_dim'],
        layers=config['model']['layers'],
        dropout_prob=config['model']['dropout_prob'],
        bn=config['model']['bn'],
        loss_type=config['model']['loss_type'],
        quant_loss_weight=config['model']['quant_loss_weight'],
        kmeans_init=config['model']['kmeans_init'],
        kmeans_iters=config['model']['kmeans_iters'],
        sk_epsilons=config['model']['sk_epsilons'],
        sk_iters=config['model']['sk_iters'],
        beta=config['model']['beta'],
        alpha=config['model']['alpha'],
        n_clusters=config['model']['n_clusters'],
        sample_strategy=config['model']['sample_strategy'],
        cf_embedding=cf_emb
    )
    logger.info(f'Model created: {model}')

    class Args:
        def __init__(self):
            self.learner = config['optimizer']['type']
            self.lr = config['optimizer']['lr']
            self.weight_decay = config['optimizer']['weight_decay']
            self.epochs = config['training']['epochs']
            self.eval_step = config['training']['eval_step']
            self.ckpt_dir = checkpoint_dir
            self.device = device
            self.experiment_name = config['experiment_name']
            self.num_workers = config['dataloader']['num_workers']
            self.data_path = config['dataset']['data_path']

    args = Args()

    trainer = Trainer(args, model)
    logger.info('Trainer created')

    logger.info('Starting training process...')
    best_loss, best_collision_rate = trainer.fit(dataloader)

    logger.info(f"Training completed!")
    logger.info(f"Best Loss: {best_loss}")
    logger.info(f"Best Collision Rate: {best_collision_rate}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()