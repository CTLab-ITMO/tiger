import torch
import torch.nn as nn

from ..utils import create_masked_tensor
from ..models import TorchModel


class SasRecModel(TorchModel):
    def __init__(
            self,
            num_items,
            max_sequence_length,
            embedding_dim,
            num_heads,
            num_layers,
            dim_feedforward,
            activation,
            topk_k,
            dropout=0.0,
            layer_norm_eps=1e-9,
            initializer_range=0.02
    ):
        super().__init__()
        self._num_items = num_items
        self._num_heads = num_heads
        self._embedding_dim = embedding_dim

        self._item_embeddings = nn.Embedding(
            num_embeddings=num_items,
            embedding_dim=embedding_dim
        )
        self._position_embeddings = nn.Embedding(
            num_embeddings=max_sequence_length,
            embedding_dim=embedding_dim
        )

        self._topk_k = topk_k

        self._layernorm = nn.LayerNorm(embedding_dim, eps=layer_norm_eps)
        self._dropout = nn.Dropout(dropout)

        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=True
        )
        self._encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers)

        self._init_weights(initializer_range)

    def forward(self, inputs):
        all_sample_events = inputs['item.ids']  # (all_batch_events)
        all_sample_lengths = inputs['item.length']  # (batch_size)

        embeddings, mask = self._apply_sequential_encoder(
            all_sample_events, all_sample_lengths
        )  # (batch_size, seq_len, embedding_dim), (batch_size, seq_len)

        if self.training:  # training mode
            all_positive_sample_events = inputs['labels.ids']  # (all_batch_events)

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
                index=torch.randint(
                    low=0,
                    high=all_scores.shape[1],
                    size=all_positive_sample_events.shape,
                    device=all_positive_sample_events.device
                )[..., None]
            )[:, 0]  # (all_batch_items)

            return {
                'positive_scores': positive_scores,
                'negative_scores': negative_scores
            }
        else:  # eval mode
            lengths = torch.sum(mask, dim=-1) - 1  # (batch_size)
            last_masks = mask.gather(dim=1, index=lengths[:, None])  # (batch_size, 1)
            lengths = torch.tile(lengths[:, None, None], (1, 1, embeddings.shape[-1]))  # (batch_size, 1, emb_dim)
            last_embeddings = embeddings.gather(dim=1, index=lengths)  # (batch_size, 1, emb_dim)
            last_embeddings = last_embeddings[last_masks]  # (batch_size, emb_dim)

            # b - batch_size, n - num_candidates, d - embedding_dim
            candidate_scores = torch.einsum(
                'bd,nd->bn',
                last_embeddings,
                self._item_embeddings.weight
            )  # (batch_size, num_items + 1)

            _, indices = torch.topk(candidate_scores, k=self._topk_k, dim=-1, largest=True)  # (batch_size, topk_k)

            return {'predictions': indices}

    def _apply_sequential_encoder(self, events, lengths):
        embeddings = self._item_embeddings(events)  # (all_batch_events, embedding_dim)

        embeddings, mask = create_masked_tensor(
            data=embeddings,
            lengths=lengths
        )  # (batch_size, seq_len, embedding_dim), (batch_size, seq_len)

        batch_size = mask.shape[0]
        seq_len = mask.shape[1]

        positions = torch.arange(
            start=0, end=seq_len, device=mask.device
        )[None].expand(batch_size, -1)  # (batch_size, seq_len)

        position_embeddings = self._position_embeddings(positions)   # (batch_size, seq_len, embedding_dim)
        position_embeddings[~mask] = 0

        embeddings = embeddings + position_embeddings  # (batch_size, seq_len, embedding_dim)
        embeddings = self._layernorm(embeddings)  # (batch_size, seq_len, embedding_dim)
        embeddings = self._dropout(embeddings)  # (batch_size, seq_len, embedding_dim)
        embeddings[~mask] = 0
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).bool().to(embeddings.device)  # (seq_len, seq_len)
        embeddings = self._encoder(
            src=embeddings,
            mask=causal_mask,
            src_key_padding_mask=~mask
        )  # (batch_size, seq_len, embedding_dim)

        return embeddings, mask
