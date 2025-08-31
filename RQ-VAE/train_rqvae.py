import json
import logging
from torch.utils.data import DataLoader
import torch

from datasets import EmbDataset
from models.rqvae import RQVAE
from trainer import Trainer
from utils import create_logger, parse_args, fix_random_seed, create_checkpoint_dir
import argparse

logger = create_logger(name=__name__)


def create_args_from_config(config, checkpoint_dir, device):
    args = argparse.Namespace()

    args.num_emb_list = config['model']['num_emb_list']
    args.e_dim = config['model']['e_dim']
    args.layers = config['model']['layers']
    args.dropout_prob = config['model']['dropout_prob']
    args.bn = config['model']['bn']
    args.loss_type = config['model']['loss_type']
    args.quant_loss_weight = config['model']['quant_loss_weight']
    args.kmeans_init = config['model']['kmeans_init']
    args.kmeans_iters = config['model']['kmeans_iters']
    args.sk_epsilons = config['model']['sk_epsilons']
    args.sk_iters = config['model']['sk_iters']
    args.beta = config['model']['beta']
    args.alpha = config['model']['alpha']
    args.n_clusters = config['model']['n_clusters']
    args.sample_strategy = config['model']['sample_strategy']

    args.learner = config['optimizer']['type']
    args.lr = config['optimizer']['lr']
    args.weight_decay = config['optimizer']['weight_decay']
    args.epochs = config['training']['epochs']
    args.eval_step = config['training']['eval_step']
    args.ckpt_dir = checkpoint_dir
    args.device = device
    args.experiment_name = config['experiment_name']
    args.num_workers = config['dataloader']['num_workers']
    args.data_path = config['dataset']['data_path']

    args.dataloader_num_workers = config['dataloader']['num_workers']
    args.dataloader_batch_size = config['dataloader']['batch_size']
    args.dataloader_shuffle = config['dataloader']['shuffle']
    args.dataloader_pin_memory = config['dataloader']['pin_memory']

    return args


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

    cf_emb = torch.load(config['model']['cf_emb_path']).squeeze().detach().numpy()

    args = create_args_from_config(config, checkpoint_dir, device)

    dataloader = DataLoader(
        dataset,
        num_workers=args.dataloader_num_workers,
        batch_size=args.dataloader_batch_size,
        shuffle=args.dataloader_shuffle,
        pin_memory=args.dataloader_pin_memory
    )
    logger.info(f'DataLoader created with batch size: {args.dataloader_batch_size}')

    model = RQVAE(
        in_dim=dataset.dim,
        num_emb_list=args.num_emb_list,
        e_dim=args.e_dim,
        layers=args.layers,
        dropout_prob=args.dropout_prob,
        bn=args.bn,
        loss_type=args.loss_type,
        quant_loss_weight=args.quant_loss_weight,
        kmeans_init=args.kmeans_init,
        kmeans_iters=args.kmeans_iters,
        sk_epsilons=args.sk_epsilons,
        sk_iters=args.sk_iters,
        beta=args.beta,
        alpha=args.alpha,
        n_clusters=args.n_clusters,
        sample_strategy=args.sample_strategy,
        cf_embedding=cf_emb
    ).to(device)
    logger.info(f'Model created: {model}')

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
