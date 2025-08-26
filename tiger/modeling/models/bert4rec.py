from models.base import SequentialTorchModel

import torch
import torch.nn as nn


class Bert4RecModel(SequentialTorchModel, config_name='bert4rec'):

    def __init__(
            self,
            sequence_prefix,
            labels_prefix,
            num_items,
            max_sequence_length,
            embedding_dim,
            num_heads,
            num_layers,
            dim_feedforward,
            dropout=0.0,
            activation='gelu',
            layer_norm_eps=1e-5,
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
            is_causal=False
        )
        self._sequence_prefix = sequence_prefix
        self._labels_prefix = labels_prefix

        self._output_projection = nn.Linear(
            in_features=embedding_dim,
            out_features=embedding_dim
        )

        self._bias = nn.Parameter(
            data=torch.zeros(num_items + 2),
            requires_grad=True
        )

        self._init_weights(initializer_range)

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            sequence_prefix=config['sequence_prefix'],
            labels_prefix=config['labels_prefix'],
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

        embeddings = self._output_projection(embeddings)  # (batch_size, seq_len, embedding_dim)
        embeddings = torch.nn.functional.gelu(embeddings)  # (batch_size, seq_len, embedding_dim)
        embeddings = torch.einsum(
            'bsd,nd->bsn', embeddings, self._item_embeddings.weight
        )  # (batch_size, seq_len, num_items)
        embeddings += self._bias[None, None, :]  # (batch_size, seq_len, num_items)

        if self.training:  # training mode
            all_sample_labels = inputs['{}.ids'.format(self._labels_prefix)]  # (all_batch_events)
            embeddings = embeddings[mask]  # (all_batch_events, num_items)
            labels_mask = (all_sample_labels != 0).bool()  # (all_batch_events)

            needed_logits = embeddings[labels_mask]  # (non_zero_events, num_items)
            needed_labels = all_sample_labels[labels_mask]  # (non_zero_events)

            return {'logits': needed_logits, 'labels.ids': needed_labels}
        else:  # eval mode
            candidate_scores = self._get_last_embedding(embeddings, mask)  # (batch_size, num_items)
            candidate_scores[:, 0] = -torch.inf
            candidate_scores[:, self._num_items + 1:] = -torch.inf

            _, indices = torch.topk(
                candidate_scores,
                k=20, dim=-1, largest=True
            )  # (batch_size, 20)

            return indices
