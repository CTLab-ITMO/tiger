import json

import torch
from transformers import T5ForConditionalGeneration, T5Config, LogitsProcessor

from ..utils import create_masked_tensor, DEVICE
from ..models import TorchModel


class CorrectItemsLogitsProcessor(LogitsProcessor):
    def __init__(self, num_codebooks, codebook_size, inter_path, visited_items):
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size

        with open(inter_path, 'r') as f:
            mapping = json.load(f)

        semantic_ids = []
        for i in range(len(mapping)):
            assert len(mapping[str(i)]) == num_codebooks, 'All semantic ids must have the same length'
            semantic_ids.append(mapping[str(i)])
        
        self.index_semantic_ids = torch.tensor(semantic_ids, dtype=torch.long, device=DEVICE)  # (num_items, semantic_ids)
        self.index_semantic_ids += torch.arange(num_codebooks, device=DEVICE)[None] * codebook_size  # (num_items, semantic_ids)
        
        batch_size, _ = visited_items.shape

        self.index_semantic_ids = torch.tile(self.index_semantic_ids[None], dims=[batch_size, 1, 1])  # (batch_size, num_items, semantic_ids)

        index = visited_items[..., None].tile(dims=[1, 1, num_codebooks])  # (batch_size, seq_len, semantic_ids)

        self.index_semantic_ids = torch.scatter(
            input=self.index_semantic_ids,
            dim=1,
            index=index,
            src=torch.zeros_like(index)  # Unexisting SIDs
        )  # (batch_size, num_items, semantic_ids)
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        next_sid_codebook_num = (torch.minimum((input_ids[:, -1].max() // self.codebook_size), torch.as_tensor(self.num_codebooks - 1)).item() + 1) % self.num_codebooks

        if next_sid_codebook_num == 0:
            scores[:, self.codebook_size:] = -torch.inf
        else:
            current_prefixes = input_ids[:, -next_sid_codebook_num:]  # (batch_size * num_beams, sid_len)
            possible_next_items_mask = (
                torch.eq(
                    current_prefixes[:, None, :], 
                    torch.tile(self.index_semantic_ids[:, :, :next_sid_codebook_num], dims=[100, 1, 1])
                ).long().sum(dim=-1) == next_sid_codebook_num
            )  # (batch_size * num_beams, num_items)

            a = torch.tile(self.index_semantic_ids[:, :, next_sid_codebook_num], dims=[100, 1])  # (batch_size * num_beams, num_items)
            a[~possible_next_items_mask] = 0
            scores_mask = torch.zeros_like(scores).bool()
            scores_mask = torch.scatter_add(
                input=scores_mask,
                dim=-1,
                index=a,
                src=torch.ones_like(a).bool()
            )
            scores_mask[:, :next_sid_codebook_num * self.codebook_size] = 0
            scores_mask[:, (next_sid_codebook_num + 1) * self.codebook_size:] = 0
            scores[~(scores_mask.bool())] = -torch.inf
        
        return scores



class TigerModel(TorchModel):
    def __init__(
            self,
            embedding_dim,
            codebook_size,
            sem_id_len,
            num_positions,
            user_ids_count,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            dim_feedforward,
            num_beams=100,
            num_return_sequences=20,
            d_kv=64,
            layer_norm_eps=1e-9,
            activation='relu',
            dropout=0.1,
            initializer_range=0.02,
            logits_processor=None
    ):
        super().__init__()
        self._embedding_dim = embedding_dim
        self._codebook_size = codebook_size
        self._num_positions = num_positions
        self._num_heads = num_heads
        self._num_encoder_layers = num_encoder_layers
        self._num_decoder_layers = num_decoder_layers
        self._dim_feedforward = dim_feedforward
        self._num_beams = num_beams
        self._num_return_sequences = num_return_sequences
        self._d_kv = d_kv
        self._layer_norm_eps = layer_norm_eps
        self._activation = activation
        self._dropout = dropout
        self._sem_id_len = sem_id_len
        self.user_ids_count = user_ids_count
        self.logits_processor = logits_processor

        unified_vocab_size = codebook_size * self._sem_id_len + self.user_ids_count + 10  # 10 for utilities
        self.config = T5Config(
            vocab_size=unified_vocab_size,
            d_model=self._embedding_dim,
            d_kv=self._d_kv,
            d_ff=self._dim_feedforward,
            num_layers=self._num_encoder_layers,
            num_decoder_layers=self._num_decoder_layers,
            num_heads=self._num_heads,
            dropout_rate=self._dropout,
            is_encoder_decoder=True,
            use_cache=False,
            pad_token_id=unified_vocab_size - 1,
            eos_token_id=unified_vocab_size - 2,
            decoder_start_token_id=unified_vocab_size - 3,
            layer_norm_epsilon=self._layer_norm_eps,
            feed_forward_proj=self._activation,
            tie_word_embeddings=False
        )
        self.model = T5ForConditionalGeneration(config=self.config)
        self._init_weights(initializer_range)

    def forward(self, inputs):
        all_sample_events = inputs['semantic_item.ids']  # (all_batch_events)
        all_sample_lengths = inputs['semantic_item.length']  # (batch_size)
        offsets = (
            torch.arange(
                start=0,
                end=all_sample_events.shape[0],
                device=all_sample_events.device,
                dtype=torch.long
            ) % self._sem_id_len
        ) * self._codebook_size
        all_sample_events = all_sample_events + offsets

        batch_size = all_sample_lengths.shape[0]

        input_semantic_ids, attention_mask = create_masked_tensor(
            data=all_sample_events,
            lengths=all_sample_lengths,
            is_right_aligned=True
        )

        input_semantic_ids[~attention_mask] = self.config.pad_token_id
        input_semantic_ids = torch.cat([
            self._sem_id_len * self._codebook_size + inputs['hashed_user.ids'][:, None],
            input_semantic_ids
        ], dim=-1)
        attention_mask = torch.cat([
            attention_mask,
            torch.ones(batch_size, 1, device=attention_mask.device, dtype=attention_mask.dtype)
        ], dim=-1)

        if self.training:
            positive_sample_events = inputs['semantic_labels.ids']  # (batch_size * sem_id_len)
            positive_sample_lengths = inputs['semantic_labels.length']  # (batch_size)
            offsets = (
                torch.arange(
                    start=0,
                    end=positive_sample_events.shape[0],
                    device=positive_sample_events.device,
                    dtype=torch.long
                ) % self._sem_id_len
            ) * self._codebook_size
            positive_sample_events = positive_sample_events + offsets

            target_semantic_ids, _ = create_masked_tensor(
                data=positive_sample_events,
                lengths=positive_sample_lengths,
                is_right_aligned=True
            )
            target_semantic_ids = torch.cat([
                torch.ones(
                    batch_size, 1,
                    dtype=torch.long,
                    device=target_semantic_ids.device
                ) * self.config.decoder_start_token_id,
                target_semantic_ids
            ], dim=-1)

            decoder_input_ids = target_semantic_ids[:, :-1].contiguous()
            labels = target_semantic_ids[:, 1:].contiguous()

            model_output = self.model(
                input_ids=input_semantic_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                labels=labels
            )

            return model_output
        else:
            visited_items, _ = create_masked_tensor(
                data=inputs['item.ids'],
                lengths=inputs['item.length']
            )  # (batch_size, seq_len)
            output = self.model.generate(
                input_ids=input_semantic_ids,
                attention_mask=attention_mask,
                num_beams=self._num_beams,
                num_return_sequences=self._num_return_sequences,
                max_length=self._sem_id_len + 1,
                decoder_start_token_id=self.config.decoder_start_token_id,
                eos_token_id=self.config.eos_token_id,
                pad_token_id=self.config.pad_token_id,
                do_sample=False,
                early_stopping=False,
                logits_processor=[self.logits_processor(visited_items=visited_items)] if self.logits_processor is not None else [],
            )
            return {
                'predictions': output[:, 1:].reshape(-1, self._num_return_sequences, self._sem_id_len)
            }
