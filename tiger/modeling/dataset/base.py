from collections import defaultdict
import json

from tqdm import tqdm

from dataset.samplers import TrainSampler, EvalSampler

from utils import MetaParent, DEVICE

import pickle
import torch
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix

import os
import logging

logger = logging.getLogger(__name__)


class BaseDataset(metaclass=MetaParent):

    def get_samplers(self):
        raise NotImplementedError


class SequenceDataset(BaseDataset, config_name='sequence'):

    def __init__(
            self,
            train_sampler,
            validation_sampler,
            test_sampler,
            num_users,
            num_items,
            max_sequence_length
    ):
        self._train_sampler = train_sampler
        self._validation_sampler = validation_sampler
        self._test_sampler = test_sampler
        self._num_users = num_users
        self._num_items = num_items
        self._max_sequence_length = max_sequence_length

    @classmethod
    def create_from_config(cls, config, **kwargs):
        data_dir_path = os.path.join(config['path_to_data_dir'], config['name'])

        train_dataset, train_max_user_id, train_max_item_id, train_seq_len = cls._create_dataset(
            dir_path=data_dir_path,
            part='train',
            max_sequence_length=config['max_sequence_length'],
            use_cached=config.get('use_cached', False)
        )
        validation_dataset, valid_max_user_id, valid_max_item_id, valid_seq_len = cls._create_dataset(
            dir_path=data_dir_path,
            part='valid',
            max_sequence_length=config['max_sequence_length'],
            use_cached=config.get('use_cached', False)
        )
        test_dataset, test_max_user_id, test_max_item_id, test_seq_len = cls._create_dataset(
            dir_path=data_dir_path,
            part='test',
            max_sequence_length=config['max_sequence_length'],
            use_cached=config.get('use_cached', False)
        )

        max_user_id = max([train_max_user_id, valid_max_user_id, test_max_user_id])
        max_item_id = max([train_max_item_id, valid_max_item_id, test_max_item_id])
        max_seq_len = max([train_seq_len, valid_seq_len, test_seq_len])

        logger.info('Train dataset size: {}'.format(len(train_dataset)))
        logger.info('Test dataset size: {}'.format(len(test_dataset)))
        logger.info('Max user id: {}'.format(max_user_id))
        logger.info('Max item id: {}'.format(max_item_id))
        logger.info('Max sequence length: {}'.format(max_seq_len))

        train_interactions = sum(list(map(lambda x: len(x), train_dataset)))  # whole user history as a sample
        valid_interactions = len(validation_dataset)  # each new interaction as a sample
        test_interactions = len(test_dataset) # each new interaction as a sample
        logger.info('{} dataset sparsity: {}'.format(
            config['name'], (train_interactions + valid_interactions + test_interactions) / max_user_id / max_item_id
        ))

        train_sampler = TrainSampler.create_from_config(
            config['samplers'],
            dataset=train_dataset,
            num_users=max_user_id,
            num_items=max_item_id
        )
        validation_sampler = EvalSampler.create_from_config(
            config['samplers'],
            dataset=validation_dataset,
            num_users=max_user_id,
            num_items=max_item_id
        )
        test_sampler = EvalSampler.create_from_config(
            config['samplers'],
            dataset=test_dataset,
            num_users=max_user_id,
            num_items=max_item_id
        )

        return cls(
            train_sampler=train_sampler,
            validation_sampler=validation_sampler,
            test_sampler=test_sampler,
            num_users=max_user_id,
            num_items=max_item_id,
            max_sequence_length=max_seq_len
        )

    @classmethod
    def _create_dataset(cls, dir_path, part, max_sequence_length=None, use_cached=False):
        max_user_id = 0
        max_item_id = 0
        max_sequence_len = 0

        if use_cached and os.path.exists(os.path.join(dir_path, '{}.pkl'.format(part))):
            logger.info(f'Take cached dataset from {os.path.join(dir_path, "{}.pkl".format(part))}')

            with open(os.path.join(dir_path, '{}.pkl'.format(part)), 'rb') as dataset_file:
                dataset, max_user_id, max_item_id, max_sequence_len = pickle.load(dataset_file)
        else:
            logger.info('Cache is forecefully ignored.' if not use_cached else 'No cached dataset has been found.')
            logger.info(f'Creating a dataset {os.path.join(dir_path, "{}.txt".format(part))}...')

            dataset_path = os.path.join(dir_path, '{}.txt'.format(part))
            with open(dataset_path, 'r') as f:
                data = f.readlines()

            sequence_info = cls._create_sequences(data, max_sequence_length)
            user_sequences, item_sequences, max_user_id, max_item_id, max_sequence_len = sequence_info

            dataset = []
            for user_id, item_ids in zip(user_sequences, item_sequences):
                dataset.append({
                    'user.ids': [user_id], 'user.length': 1,
                    'item.ids': item_ids, 'item.length': len(item_ids)
                })

            logger.info('{} dataset size: {}'.format(part, len(dataset)))
            logger.info('{} dataset max sequence length: {}'.format(part, max_sequence_len))

            with open(os.path.join(dir_path, '{}.pkl'.format(part)), 'wb') as dataset_file:
                pickle.dump(
                    (dataset, max_user_id, max_item_id, max_sequence_len),
                    dataset_file
                )

        return dataset, max_user_id, max_item_id, max_sequence_len

    @staticmethod
    def _create_sequences(data, max_sample_len):
        user_sequences = []
        item_sequences = []

        max_user_id = 0
        max_item_id = 0
        max_sequence_length = 0

        for sample in data:
            sample = sample.strip('\n').split(' ')
            item_ids = [int(item_id) for item_id in sample[1:]][-max_sample_len:]
            user_id = int(sample[0])

            max_user_id = max(max_user_id, user_id)
            max_item_id = max(max_item_id, max(item_ids))
            max_sequence_length = max(max_sequence_length, len(item_ids))

            user_sequences.append(user_id)
            item_sequences.append(item_ids)

        return user_sequences, item_sequences, max_user_id, max_item_id, max_sequence_length

    def get_samplers(self):
        return self._train_sampler, self._validation_sampler, self._test_sampler

    @property
    def num_users(self):
        return self._num_users

    @property
    def num_items(self):
        return self._num_items

    @property
    def max_sequence_length(self):
        return self._max_sequence_length

    @property
    def meta(self):
        return {
            'num_users': self.num_users,
            'num_items': self.num_items,
            'max_sequence_length': self.max_sequence_length
        }


class GraphDataset(BaseDataset, config_name='graph'):

    def __init__(
            self,
            dataset,
            graph_dir_path,
            use_train_data_only=True,
            use_user_graph=False,
            use_item_graph=False
    ):
        self._dataset = dataset
        self._graph_dir_path = graph_dir_path
        self._use_train_data_only = use_train_data_only
        self._use_user_graph = use_user_graph
        self._use_item_graph = use_item_graph

        self._num_users = dataset.num_users
        self._num_items = dataset.num_items

        train_sampler, validation_sampler, test_sampler = dataset.get_samplers()

        train_interactions, train_user_interactions, train_item_interactions = [], [], []

        train_user_2_items = defaultdict(set)
        train_item_2_users = defaultdict(set)
        visited_user_item_pairs = set()

        for sample in train_sampler.dataset:
            user_id = sample['user.ids'][0]
            item_ids = sample['item.ids']

            for item_id in item_ids:
                if (user_id, item_id) not in visited_user_item_pairs:
                    train_interactions.append((user_id, item_id))
                    train_user_interactions.append(user_id)
                    train_item_interactions.append(item_id)

                    train_user_2_items[user_id].add(item_id)
                    train_item_2_users[item_id].add(user_id)

                    visited_user_item_pairs.add((user_id, item_id))

        # TODO create separated function
        if not self._use_train_data_only:
            for sample in validation_sampler.dataset:
                user_id = sample['user.ids'][0]
                item_ids = sample['item.ids']

                for item_id in item_ids:
                    if (user_id, item_id) not in visited_user_item_pairs:
                        train_interactions.append((user_id, item_id))
                        train_user_interactions.append(user_id)
                        train_item_interactions.append(item_id)

                        train_user_2_items[user_id].add(item_id)
                        train_item_2_users[item_id].add(user_id)

                        visited_user_item_pairs.add((user_id, item_id))

            for sample in test_sampler.dataset:
                user_id = sample['user.ids'][0]
                item_ids = sample['item.ids']

                for item_id in item_ids:
                    if (user_id, item_id) not in visited_user_item_pairs:
                        train_interactions.append((user_id, item_id))
                        train_user_interactions.append(user_id)
                        train_item_interactions.append(item_id)

                        train_user_2_items[user_id].add(item_id)
                        train_item_2_users[item_id].add(user_id)

                        visited_user_item_pairs.add((user_id, item_id))

        self._train_interactions = np.array(train_interactions)
        self._train_user_interactions = np.array(train_user_interactions)
        self._train_item_interactions = np.array(train_item_interactions)

        path_to_graph = os.path.join(graph_dir_path, 'general_graph.npz')
        if os.path.exists(path_to_graph):
            self._graph = sp.load_npz(path_to_graph)
        else:
            # place ones only when co-occurrence happens
            user2item_connections = csr_matrix(
                (np.ones(len(train_user_interactions)), (train_user_interactions, train_item_interactions)),
                shape=(self._num_users + 2, self._num_items + 2)
            )  # (num_users + 2, num_items + 2), bipartite graph
            self._graph = self.get_sparse_graph_layer(
                user2item_connections,
                self._num_users + 2,
                self._num_items + 2,
                biparite=True
            )
            sp.save_npz(path_to_graph, self._graph)

        self._graph = self._convert_sp_mat_to_sp_tensor(self._graph).coalesce().to(DEVICE)

        if self._use_user_graph:
            path_to_user_graph = os.path.join(graph_dir_path, 'user_graph.npz')
            if os.path.exists(path_to_user_graph):
                self._user_graph = sp.load_npz(path_to_user_graph)
            else:
                user2user_interactions_fst = []
                user2user_interactions_snd = []
                visited_user_item_pairs = set()
                visited_user_user_pairs = set()

                for user_id, item_id in tqdm(zip(self._train_user_interactions, self._train_item_interactions)):
                    if (user_id, item_id) in visited_user_item_pairs:
                        continue  # process (user, item) pair only once
                    visited_user_item_pairs.add((user_id, item_id))

                    for connected_user_id in train_item_2_users[item_id]:
                        if (user_id, connected_user_id) in visited_user_user_pairs or user_id == connected_user_id:
                            continue  # add (user, user) to graph connections pair only once
                        visited_user_user_pairs.add((user_id, connected_user_id))

                        user2user_interactions_fst.append(user_id)
                        user2user_interactions_snd.append(connected_user_id)

                # (user, user) graph
                user2user_connections = csr_matrix(
                    (
                    np.ones(len(user2user_interactions_fst)), (user2user_interactions_fst, user2user_interactions_snd)),
                    shape=(self._num_users + 2, self._num_users + 2)
                )

                self._user_graph = self.get_sparse_graph_layer(
                    user2user_connections,
                    self._num_users + 2,
                    self._num_users + 2,
                    biparite=False
                )
                sp.save_npz(path_to_user_graph, self._user_graph)

            self._user_graph = self._convert_sp_mat_to_sp_tensor(self._user_graph).coalesce().to(DEVICE)
        else:
            self._user_graph = None

        if self._use_item_graph:
            path_to_item_graph = os.path.join(graph_dir_path, 'item_graph.npz')
            if os.path.exists(path_to_item_graph):
                self._item_graph = sp.load_npz(path_to_item_graph)
            else:
                item2item_interactions_fst = []
                item2item_interactions_snd = []
                visited_user_item_pairs = set()
                visited_item_item_pairs = set()

                for user_id, item_id in tqdm(zip(self._train_user_interactions, self._train_item_interactions)):
                    if (user_id, item_id) in visited_user_item_pairs:
                        continue  # process (user, item) pair only once
                    visited_user_item_pairs.add((user_id, item_id))

                    for connected_item_id in train_user_2_items[user_id]:
                        if (item_id, connected_item_id) in visited_item_item_pairs or item_id == connected_item_id:
                            continue  # add (item, item) to graph connections pair only once
                        visited_item_item_pairs.add((item_id, connected_item_id))

                        item2item_interactions_fst.append(item_id)
                        item2item_interactions_snd.append(connected_item_id)

                # (item, item) graph
                item2item_connections = csr_matrix(
                    (
                    np.ones(len(item2item_interactions_fst)), (item2item_interactions_fst, item2item_interactions_snd)),
                    shape=(self._num_items + 2, self._num_items + 2)
                )
                self._item_graph = self.get_sparse_graph_layer(
                    item2item_connections,
                    self._num_items + 2,
                    self._num_items + 2,
                    biparite=False
                )
                sp.save_npz(path_to_item_graph, self._item_graph)

            self._item_graph = self._convert_sp_mat_to_sp_tensor(self._item_graph).coalesce().to(DEVICE)
        else:
            self._item_graph = None

    @classmethod
    def create_from_config(cls, config):
        dataset = BaseDataset.create_from_config(config['dataset'])
        return cls(
            dataset=dataset,
            graph_dir_path=config['graph_dir_path'],
            use_user_graph=config.get('use_user_graph', False),
            use_item_graph=config.get('use_item_graph', False)
        )

    @staticmethod
    def get_sparse_graph_layer(sparse_matrix, fst_dim, snd_dim, biparite=False):
        mat_dim_size = fst_dim + snd_dim if biparite else fst_dim

        adj_mat = sp.dok_matrix(
            (mat_dim_size, mat_dim_size),
            dtype=np.float32
        )
        adj_mat = adj_mat.tolil()

        R = sparse_matrix.tolil()  # list of lists (fst_dim, snd_dim)

        if biparite:
            adj_mat[:fst_dim, fst_dim:] = R  # (num_users, num_items)
            adj_mat[fst_dim:, :fst_dim] = R.T  # (num_items, num_users)
        else:
            adj_mat = R

        adj_mat = adj_mat.todok()
        # adj_mat += sp.eye(adj_mat.shape[0])  # remove division by zero issue

        edges_degree = np.array(adj_mat.sum(axis=1))  # D

        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        d_inv = np.power(edges_degree, -0.5).flatten()  # D^(-0.5)
        d_inv[np.isinf(d_inv)] = 0.  # fix NaNs in case if row with zero connections
        d_mat = sp.diags(d_inv)  # make it square matrix

        # D^(-0.5) @ A @ D^(-0.5)
        norm_adj = d_mat.dot(adj_mat).dot(d_mat)

        return norm_adj.tocsr()

    @staticmethod
    def _convert_sp_mat_to_sp_tensor(X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    @property
    def num_users(self):
        return self._dataset.num_users

    @property
    def num_items(self):
        return self._dataset.num_items

    def get_samplers(self):
        return self._dataset.get_samplers()

    @property
    def meta(self):
        meta = {
            'user_graph': self._user_graph,
            'item_graph': self._item_graph,
            'graph': self._graph,
            **self._dataset.meta
        }
        return meta


class DuorecDataset(BaseDataset, config_name='duorec'):

    def __init__(self, dataset):
        self._dataset = dataset
        self._num_users = dataset.num_users
        self._num_items = dataset.num_items

        train_sampler, _, _ = self._dataset.get_samplers()

        target_2_sequences = defaultdict(list)
        for sample in train_sampler.dataset:
            item_ids = sample['item.ids']

            target_item = item_ids[-1]
            semantic_similar_item_ids = item_ids[:-1]

            target_2_sequences[target_item].append(semantic_similar_item_ids)

        train_sampler._target_2_sequences = target_2_sequences

    @classmethod
    def create_from_config(cls, config):
        dataset = BaseDataset.create_from_config(config['dataset'])
        return cls(dataset)

    @property
    def num_users(self):
        return self._dataset.num_users

    @property
    def num_items(self):
        return self._dataset.num_items

    def get_samplers(self):
        return self._dataset.get_samplers()

    @property
    def meta(self):
        return self._dataset.meta


class ScientificDataset(BaseDataset, config_name='scientific'):

    def __init__(
            self,
            train_sampler,
            validation_sampler,
            test_sampler,
            num_users,
            num_items,
            max_sequence_length
    ):
        self._train_sampler = train_sampler
        self._validation_sampler = validation_sampler
        self._test_sampler = test_sampler
        self._num_users = num_users
        self._num_items = num_items
        self._max_sequence_length = max_sequence_length

    @classmethod
    def create_from_config(cls, config, **kwargs):
        data_dir_path = os.path.join(config['path_to_data_dir'], config['name'])
        max_sequence_length = config['max_sequence_length']
        max_user_id, max_item_id = 0, 0
        train_dataset, validation_dataset, test_dataset = [], [], []

        dataset_path = os.path.join(data_dir_path, '{}.txt'.format('all_data'))
        with open(dataset_path, 'r') as f:
            data = f.readlines()

        for sample in data:
            sample = sample.strip('\n').split(' ')
            user_id = int(sample[0])
            item_ids = [int(item_id) for item_id in sample[1:]]

            max_user_id = max(max_user_id, user_id)
            max_item_id = max(max_item_id, max(item_ids))

            assert len(item_ids) >= 5

            train_dataset.append({
                'user.ids': [user_id],
                'user.length': 1,
                'item.ids': item_ids[:-2][-max_sequence_length:],
                'item.length': len(item_ids[:-2][-max_sequence_length:])
            })
            assert len(item_ids[:-2][-max_sequence_length:]) == len(set(item_ids[:-2][-max_sequence_length:]))
            validation_dataset.append({
                'user.ids': [user_id],
                'user.length': 1,
                'item.ids': item_ids[:-1][-max_sequence_length:],
                'item.length': len(item_ids[:-1][-max_sequence_length:])
            })
            assert len(item_ids[:-1][-max_sequence_length:]) == len(set(item_ids[:-1][-max_sequence_length:]))
            test_dataset.append({
                'user.ids': [user_id],
                'user.length': 1,
                'item.ids': item_ids[-max_sequence_length:],
                'item.length': len(item_ids[-max_sequence_length:])
            })
            assert len(item_ids[-max_sequence_length:]) == len(set(item_ids[-max_sequence_length:]))

        logger.info('Train dataset size: {}'.format(len(train_dataset)))
        logger.info('Test dataset size: {}'.format(len(test_dataset)))
        logger.info('Max user id: {}'.format(max_user_id))
        logger.info('Max item id: {}'.format(max_item_id))
        logger.info('Max sequence length: {}'.format(max_sequence_length))
        logger.info('{} dataset sparsity: {}'.format(
            config['name'], (len(train_dataset) + len(test_dataset)) / max_user_id / max_item_id
        ))

        train_sampler = TrainSampler.create_from_config(
            config['samplers'],
            dataset=train_dataset,
            num_users=max_user_id,
            num_items=max_item_id
        )
        validation_sampler = EvalSampler.create_from_config(
            config['samplers'],
            dataset=validation_dataset,
            num_users=max_user_id,
            num_items=max_item_id
        )
        test_sampler = EvalSampler.create_from_config(
            config['samplers'],
            dataset=test_dataset,
            num_users=max_user_id,
            num_items=max_item_id
        )

        return cls(
            train_sampler=train_sampler,
            validation_sampler=validation_sampler,
            test_sampler=test_sampler,
            num_users=max_user_id,
            num_items=max_item_id,
            max_sequence_length=max_sequence_length
        )

    def get_samplers(self):
        return self._train_sampler, self._validation_sampler, self._test_sampler

    @property
    def num_users(self):
        return self._num_users

    @property
    def num_items(self):
        return self._num_items

    @property
    def max_sequence_length(self):
        return self._max_sequence_length

    @property
    def meta(self):
        return {
            'num_users': self.num_users,
            'num_items': self.num_items,
            'max_sequence_length': self.max_sequence_length
        }


class ScientificFullDataset(ScientificDataset, config_name="scientific_full"):
    def __init__(
            self,
            train_sampler,
            validation_sampler,
            test_sampler,
            num_users,
            num_items,
            max_sequence_length,
     ):
        self._train_sampler = train_sampler
        self._validation_sampler = validation_sampler
        self._test_sampler = test_sampler
        self._num_users = num_users
        self._num_items = num_items
        self._max_sequence_length = max_sequence_length
    
    @classmethod
    def create_from_config(cls, config, **kwargs):
        data_dir_path = os.path.join(config["path_to_data_dir"], config["name"])
        max_sequence_length = config["max_sequence_length"]
        max_user_id, max_item_id = 0, 0
        train_dataset, validation_dataset, test_dataset = [], [], []

        dataset_path = os.path.join(data_dir_path, "{}.txt".format("all_data"))
        with open(dataset_path, "r") as f:
            data = f.readlines()

            for sample in data:
                sample = sample.strip("\n").split(" ")
                user_id = int(sample[0])
                item_ids = [int(item_id) for item_id in sample[1:]]

                max_user_id = max(max_user_id, user_id)
                max_item_id = max(max_item_id, max(item_ids))

                assert len(item_ids) >= 5, 'Core-5 dataset required'

                # item_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                # prefix_length: 5, 6, 7, 8
                for prefix_length in range(5 - 2 + 1, len(item_ids) - 2 + 1):
                    # prefix = [1, 2, 3, 4, 5]
                    # prefix = [1, 2, 3, 4, 5, 6]
                    # prefix = [1, 2, 3, 4, 5, 6, 7]
                    # prefix = [1, 2, 3, 4, 5, 6, 7, 8]
                    prefix = item_ids[
                        :prefix_length
                    ]  # TODO no sliding window, only incrementing sequence from last 50 items

                    train_dataset.append(
                        {
                            "user.ids": [user_id],
                            "user.length": 1,
                            "item.ids": prefix[-max_sequence_length:],
                            "item.length": len(prefix[-max_sequence_length:]),
                        }
                    )

                # item_ids[:-1] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
                validation_dataset.append(
                    {
                        "user.ids": [user_id],
                        "user.length": 1,
                        "item.ids": item_ids[:-1][-max_sequence_length:],
                        "item.length": len(item_ids[:-1][-max_sequence_length:]),
                    }
                )

                # item_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                test_dataset.append(
                    {
                        "user.ids": [user_id],
                        "user.length": 1,
                        "item.ids": item_ids[-max_sequence_length:],
                        "item.length": len(item_ids[-max_sequence_length:]),
                    }
                )

        logger.info("Train dataset size: {}".format(len(train_dataset)))
        logger.info("Validation dataset size: {}".format(len(validation_dataset)))
        logger.info("Test dataset size: {}".format(len(test_dataset)))
        logger.info("Max user id: {}".format(max_user_id))
        logger.info("Max item id: {}".format(max_item_id))
        logger.info("Max sequence length: {}".format(max_sequence_length))
        logger.info(
            "{} dataset sparsity: {}".format(
                config["name"],
                (len(train_dataset) + len(test_dataset)) / max_user_id / max_item_id,
            )
        )

        train_sampler = TrainSampler.create_from_config(
            config["samplers"],
            dataset=train_dataset,
            num_users=max_user_id,
            num_items=max_item_id,
        )
        validation_sampler = EvalSampler.create_from_config(
            config["samplers"],
            dataset=validation_dataset,
            num_users=max_user_id,
            num_items=max_item_id,
        )
        test_sampler = EvalSampler.create_from_config(
            config["samplers"],
            dataset=test_dataset,
            num_users=max_user_id,
            num_items=max_item_id,
        )

        return cls(
            train_sampler=train_sampler,
            validation_sampler=validation_sampler,
            test_sampler=test_sampler,
            num_users=max_user_id,
            num_items=max_item_id,
            max_sequence_length=max_sequence_length,
        )


class LetterDataset(ScientificDataset, config_name="letter"):
    @classmethod
    def create_from_config(cls, config, **kwargs):
        user_interactions_path = os.path.join(config["letter_inter_json"])
        with open(user_interactions_path, "r") as f:
            user_interactions = json.load(f)

        dir_path = os.path.join(config["path_to_data_dir"], config["name"])

        os.makedirs(dir_path, exist_ok=True)
        dataset_path = os.path.join(dir_path, "all_data.txt")

        logger.info(f"Saving data to {dataset_path}")

        # Map from LETTER format to Our format
        with open(dataset_path, "w") as f:
            for user_id, item_ids in user_interactions.items():
                items_repr = map(str, item_ids)
                f.write(f"{user_id} {' '.join(items_repr)}\n")

        dataset = ScientificDataset.create_from_config(config, **kwargs)

        return cls(
            train_sampler=dataset._train_sampler,
            validation_sampler=dataset._validation_sampler,
            test_sampler=dataset._test_sampler,
            num_users=dataset._num_users,
            num_items=dataset._num_items,
            max_sequence_length=dataset._max_sequence_length,
        )


class LetterFullDataset(ScientificFullDataset, config_name="letter_full"):
    @classmethod
    def create_from_config(cls, config, **kwargs):
        user_interactions_path = os.path.join(config["letter_inter_json"])
        with open(user_interactions_path, "r") as f:
            user_interactions = json.load(f)

        dir_path = os.path.join(config["path_to_data_dir"], config["name"])

        os.makedirs(dir_path, exist_ok=True)
        dataset_path = os.path.join(dir_path, "all_data.txt")

        logger.info(f"Saving data to {dataset_path}")

        # Map from LETTER format to Our format
        with open(dataset_path, "w") as f:
            for user_id, item_ids in user_interactions.items():
                items_repr = map(str, item_ids)
                f.write(f"{user_id} {' '.join(items_repr)}\n")

        dataset = ScientificFullDataset.create_from_config(config, **kwargs)

        return cls(
            train_sampler=dataset._train_sampler,
            validation_sampler=dataset._validation_sampler,
            test_sampler=dataset._test_sampler,
            num_users=dataset._num_users,
            num_items=dataset._num_items,
            max_sequence_length=dataset._max_sequence_length,
        )

class RqVaeDataset(BaseDataset, config_name="rqvae"):
    def __init__(self, train_sampler, validation_sampler, test_sampler, num_items):
        self._train_sampler = train_sampler
        self._validation_sampler = validation_sampler
        self._test_sampler = test_sampler
        self._num_items = num_items

    @classmethod
    def create_from_config(cls, config, **kwargs):
        data_dir_path = os.path.join(config["path_to_data_dir"], config["name"])
        train_dataset, validation_dataset, test_dataset = [], [], []

        dataset_path = os.path.join(data_dir_path, "{}.pkl".format("final_data_reduced"))

        with open(dataset_path, 'rb') as file:
            data_reduced = pickle.load(file)

        embeddings_np = data_reduced['embedding']  # Assuming this is a numpy array
        item_ids = data_reduced['item_id']  # Array or list of item IDs
        print(len(embeddings_np), len(embeddings_np[0]))
        embeddings_tensor = torch.tensor(embeddings_np)

        train_dataset = []
        for idx, item_id in enumerate(item_ids):
            train_dataset.append({
                "item.id": item_id,
                "item.embed": embeddings_tensor[idx]
            })
        print(train_dataset[0])
        logger.info("Train dataset size: {}".format(len(train_dataset)))
        logger.info("Test dataset size: {}".format(len(test_dataset)))

        train_sampler = TrainSampler.create_from_config(
            config["samplers"], dataset=train_dataset
        )
        validation_sampler = EvalSampler.create_from_config(
            config["samplers"], dataset=validation_dataset
        )
        test_sampler = EvalSampler.create_from_config(
            config["samplers"], dataset=test_dataset
        )

        return cls(
            train_sampler=train_sampler,
            validation_sampler=validation_sampler,
            test_sampler=test_sampler,
            num_items=len(test_dataset),
        )

    def get_samplers(self):
        return self._train_sampler, self._validation_sampler, self._test_sampler

    @property
    def num_items(self):
        return self._num_items

    @property
    def max_sequence_length(self):
        return self._max_sequence_length

    @property
    def meta(self):
        return {"num_items": self.num_items, "train_sampler": self._train_sampler}
