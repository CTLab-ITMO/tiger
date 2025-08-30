import collections
import json
import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import EmbDataset
from models.rqvae import RQVAE
from utils import create_logger, parse_args

logger = create_logger(name=__name__)


def main():
    config = parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logger.info(f'Using device: {device}')
    logger.info(f'Config: \n{json.dumps(config, indent=2)}')

    checkpoint_config = config['checkpoint']
    ckpt_path = os.path.join(
        checkpoint_config['root_path'],
        f"alpha{checkpoint_config['alpha']}-beta{checkpoint_config['beta']}",
        checkpoint_config['checkpoint_name']
    )

    logger.info(f'Loading checkpoint from: {ckpt_path}')

    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    args = ckpt["args"]
    state_dict = ckpt["state_dict"]

    data = EmbDataset(args.data_path)
    logger.info(f'Dataset loaded with dimension: {data.dim}')

    model = RQVAE(
        in_dim=data.dim,
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
    )

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    logger.info('Model loaded and set to evaluation mode')

    inference_config = config['inference']
    data_loader = DataLoader(
        data,
        num_workers=inference_config['num_workers'],
        batch_size=inference_config['batch_size'],
        shuffle=inference_config['shuffle'],
        pin_memory=inference_config['pin_memory']
    )

    from k_means_constrained import KMeansConstrained

    def constrained_km(data, n_clusters=10):
        x = data
        size_min = min(len(data) // (n_clusters * 2), 10)
        clf = KMeansConstrained(
            n_clusters=n_clusters, size_min=size_min, size_max=n_clusters * 6,
            max_iter=10, n_init=10, n_jobs=10, verbose=False
        )
        clf.fit(x)
        return torch.from_numpy(clf.cluster_centers_), torch.from_numpy(clf.labels_).tolist()

    labels = {"0": [], "1": [], "2": [], "3": []}
    embs = [layer.embedding.weight.cpu().detach().numpy() for layer in model.rq.vq_layers]

    n_clusters = inference_config['n_clusters']
    for idx, emb in enumerate(embs):
        centers, label = constrained_km(emb, n_clusters)
        labels[str(idx)] = label

    logger.info(f'Prepared labels for {len(embs)} VQ layers with {n_clusters} clusters each')

    all_indices = []
    all_indices_str = []
    prefix = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>", "<f_{}>"]

    logger.info('Starting initial index generation...')

    for d in tqdm(data_loader, desc="Generating indices"):
        d, emb_idx = d[0], d[1]
        d = d.to(device)

        indices = model.get_indices(d, labels, use_sk=False)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()

        for index in indices:
            code = []
            for i, ind in enumerate(index):
                code.append(prefix[i].format(int(ind)))

            all_indices.append(code)
            all_indices_str.append(str(code))

    all_indices = np.array(all_indices)
    all_indices_str = np.array(all_indices_str)

    logger.info(f'Generated {len(all_indices)} initial indices')

    def check_collision(all_indices_str):
        tot_item = len(all_indices_str)
        tot_indice = len(set(all_indices_str.tolist()))
        return tot_item == tot_indice

    def get_indices_count(all_indices_str):
        indices_count = collections.defaultdict(int)
        for index in all_indices_str:
            indices_count[index] += 1
        return indices_count

    def get_collision_item(all_indices_str):
        index2id = {}
        for i, index in enumerate(all_indices_str):
            if index not in index2id:
                index2id[index] = []
            index2id[index].append(i)

        collision_item_groups = []
        for index in index2id:
            if len(index2id[index]) > 1:
                collision_item_groups.append(index2id[index])

        return collision_item_groups

    collision_config = config['collision_handling']
    for vq in model.rq.vq_layers[:-1]:
        vq.sk_epsilon = collision_config['sk_epsilon_override']

    if model.rq.vq_layers[-1].sk_epsilon == 0.0:
        model.rq.vq_layers[-1].sk_epsilon = collision_config['sk_epsilon_default']

    max_iterations = inference_config['max_collision_iterations']
    tt = 0

    logger.info('Starting collision handling...')

    while True:
        if tt >= max_iterations or check_collision(all_indices_str):
            break

        collision_item_groups = get_collision_item(all_indices_str)
        logger.info(f'Iteration {tt}: Found {len(collision_item_groups)} collision groups')

        for collision_items in tqdm(collision_item_groups, desc=f"Handling collisions (iter {tt})"):
            d = data[collision_items]
            d = d[0].to(device)
            indices = model.get_indices(d, labels, use_sk=True)
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()

            for item, index in zip(collision_items, indices):
                code = []
                for i, ind in enumerate(index):
                    code.append(prefix[i].format(int(ind)))

                all_indices[item] = code
                all_indices_str[item] = str(code)

        tt += 1

    logger.info(f'Collision handling completed after {tt} iterations')

    output_config = config['output']
    output_dir = output_config['output_dir'].format(dataset=config['dataset'])
    output_filename = output_config['output_filename'].format(
        dataset=config['dataset'],
        epoch=checkpoint_config['epoch'],
        alpha=checkpoint_config['alpha'],
        beta=checkpoint_config['beta']
    )
    output_file = os.path.join(output_dir, output_filename)

    os.makedirs(output_dir, exist_ok=True)

    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    collision_rate = (tot_item - tot_indice) / tot_item
    max_conflicts = max(get_indices_count(all_indices_str).values())

    logger.info(f"All indices number: {len(all_indices)}")
    logger.info(f"Max number of conflicts: {max_conflicts}")
    logger.info(f"Collision Rate: {collision_rate}")

    all_indices_dict = {}
    for item, indices in enumerate(all_indices.tolist()):
        all_indices_dict[item] = list(indices)

    with open(output_file, 'w') as fp:
        json.dump(all_indices_dict, fp)

    logger.info(f'Results saved to: {output_file}')
    logger.info('Index generation completed successfully!')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
