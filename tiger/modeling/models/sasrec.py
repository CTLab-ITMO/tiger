from models import SequentialTorchModel
from utils import create_masked_tensor

import torch


class SasRecModel(SequentialTorchModel, config_name='sasrec'):

    def __init__(
            self,
            sequence_prefix,
            positive_prefix,
            num_items,
            max_sequence_length,
            embedding_dim,
            num_heads,
            num_layers,
            dim_feedforward,
            dropout=0.0,
            activation='relu',
            layer_norm_eps=1e-9,
            initializer_range=0.02
    ):
        super().__init__(
            num_items=num_items,
            max_sequence_length=max_sequence_length,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            is_causal=True
        )
        self._sequence_prefix = sequence_prefix
        self._positive_prefix = positive_prefix

        self._init_weights(initializer_range)

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            sequence_prefix=config['sequence_prefix'],
            positive_prefix=config['positive_prefix'],
            num_items=kwargs['num_items'],
            max_sequence_length=kwargs['max_sequence_length'],
            embedding_dim=config['embedding_dim'],
            num_heads=config.get('num_heads', int(config['embedding_dim'] // 64)),
            num_layers=config['num_layers'],
            dim_feedforward=config.get('dim_feedforward', 4 * config['embedding_dim']),
            dropout=config.get('dropout', 0.0),
            initializer_range=config.get('initializer_range', 0.02)
        )

    def forward(self, inputs):
        all_sample_events = inputs['{}.ids'.format(self._sequence_prefix)]  # (all_batch_events)
        all_sample_lengths = inputs['{}.length'.format(self._sequence_prefix)]  # (batch_size)

        embeddings, mask = self._apply_sequential_encoder(
            all_sample_events, all_sample_lengths
        )  # (batch_size, seq_len, embedding_dim), (batch_size, seq_len)

        if self.training:  # training mode
            all_positive_sample_events = inputs['{}.ids'.format(self._positive_prefix)]  # (all_batch_events)

            all_sample_embeddings = embeddings[mask]  # (all_batch_events, embedding_dim)

            all_embeddings = self._item_embeddings.weight  # (num_items, embedding_dim)

            # a -- all_batch_events, n -- num_items, d -- embedding_dim
            all_scores = torch.einsum(
                'ad,nd->an',
                all_sample_embeddings,
                all_embeddings
            )  # (all_batch_events, num_items)

            positive_scores = torch.gather(
                input=all_scores,
                dim=1,
                index=all_positive_sample_events[..., None]
            )[:, 0]  # (all_batch_items)

            negative_scores = torch.gather(
                input=all_scores,
                dim=1,
                index=torch.randint(low=0, high=all_scores.shape[1], size=all_positive_sample_events.shape, device=all_positive_sample_events.device)[..., None]
            )[:, 0]  # (all_batch_items)

            # sample_ids, _ = create_masked_tensor(
            #     data=all_sample_events,
            #     lengths=all_sample_lengths
            # )  # (batch_size, seq_len)

            # sample_ids = torch.repeat_interleave(sample_ids, all_sample_lengths, dim=0)  # (all_batch_events, seq_len)

            # negative_scores = torch.scatter(
            #     input=all_scores,
            #     dim=1,
            #     index=sample_ids,
            #     src=torch.ones_like(sample_ids) * (-torch.inf)
            # )  # (all_batch_events, num_items)

            return {
                'positive_scores': positive_scores,
                'negative_scores': negative_scores
            }
        else:  # eval mode
            last_embeddings = self._get_last_embedding(embeddings, mask)  # (batch_size, embedding_dim)
            # b - batch_size, n - num_candidates, d - embedding_dim
            candidate_scores = torch.einsum(
                'bd,nd->bn',
                last_embeddings,
                self._item_embeddings.weight
            )  # (batch_size, num_items + 2)

            _, indices = torch.topk(
                candidate_scores,
                k=20, dim=-1, largest=True
            )  # (batch_size, 20)

            return {
                'predictions': indices
            }


class SasRecInBatchModel(SasRecModel, config_name='sasrec_in_batch'):

    def __init__(
            self,
            sequence_prefix,
            positive_prefix,
            num_items,
            max_sequence_length,
            embedding_dim,
            num_heads,
            num_layers,
            dim_feedforward,
            dropout=0.0,
            activation='relu',
            layer_norm_eps=1e-9,
            initializer_range=0.02
    ):
        super().__init__(
            sequence_prefix=sequence_prefix,
            positive_prefix=positive_prefix,
            num_items=num_items,
            max_sequence_length=max_sequence_length,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            initializer_range=initializer_range
        )

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            sequence_prefix=config['sequence_prefix'],
            positive_prefix=config['positive_prefix'],
            num_items=kwargs['num_items'],
            max_sequence_length=kwargs['max_sequence_length'],
            embedding_dim=config['embedding_dim'],
            num_heads=config.get('num_heads', int(config['embedding_dim'] // 64)),
            num_layers=config['num_layers'],
            dim_feedforward=config.get('dim_feedforward', 4 * config['embedding_dim']),
            dropout=config.get('dropout', 0.0),
            initializer_range=config.get('initializer_range', 0.02)
        )

    def forward(self, inputs):
        all_sample_events = inputs['{}.ids'.format(self._sequence_prefix)]  # (all_batch_events)
        all_sample_lengths = inputs['{}.length'.format(self._sequence_prefix)]  # (batch_size)

        embeddings, mask = self._apply_sequential_encoder(
            all_sample_events, all_sample_lengths
        )  # (batch_size, seq_len, embedding_dim), (batch_size, seq_len)

        if self.training:  # training mode
            # queries
            in_batch_queries_embeddings = embeddings[mask]  # (all_batch_events, embedding_dim)

            # positives
            in_batch_positive_events = inputs['{}.ids'.format(self._positive_prefix)]  # (all_batch_events)
            in_batch_positive_embeddings = self._item_embeddings(
                in_batch_positive_events
            )  # (all_batch_events, embedding_dim)

            # negatives
            batch_size = all_sample_lengths.shape[0]
            random_ids = torch.randperm(in_batch_positive_events.shape[0])
            in_batch_negative_ids = in_batch_positive_events[random_ids][:batch_size]

            in_batch_negative_embeddings = self._item_embeddings(
                in_batch_negative_ids
            )  # (batch_size, embedding_dim)

            return {
                'query_embeddings': in_batch_queries_embeddings,
                'positive_embeddings': in_batch_positive_embeddings,
                'negative_embeddings': in_batch_negative_embeddings
            }
        else:  # eval mode
            last_embeddings = self._get_last_embedding(embeddings, mask)  # (batch_size, embedding_dim)

            # b - batch_size, n - num_candidates, d - embedding_dim
            candidate_scores = torch.einsum(
                'bd,nd->bn',
                last_embeddings,
                self._item_embeddings.weight
            )  # (batch_size, num_items + 2)
            candidate_scores[:, 0] = -torch.inf
            candidate_scores[:, self._num_items + 1:] = -torch.inf

            _, indices = torch.topk(
                candidate_scores,
                k=20, dim=-1, largest=True
            )  # (batch_size, 20)

            return indices
