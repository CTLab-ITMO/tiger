import torch
from transformers import T5ForConditionalGeneration, T5Config

from ..utils import create_masked_tensor
from ..models import TorchModel


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
            num_beams=50,
            num_return_sequences=20,
            d_kv=64,
            dropout=0.0,
            initializer_range=0.02,
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
        self._dropout = dropout
        self._sem_id_len = sem_id_len
        self.user_ids_count = user_ids_count

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
            tie_word_embeddings=False
        )
        self.model = T5ForConditionalGeneration(config=self.config)
        self._init_weights(initializer_range)

    def forward(self, inputs):
        all_sample_events = inputs["semantic_item.ids"]  # (all_batch_events)
        all_sample_lengths = inputs["semantic_item.length"]  # (batch_size)
        offsets = (torch.arange(start=0, end=all_sample_events.shape[0], device=all_sample_events.device,
                                dtype=torch.long) % self._sem_id_len) * self._codebook_size
        all_sample_events = all_sample_events + offsets

        batch_size = all_sample_lengths.shape[0]

        input_semantic_ids, attention_mask = create_masked_tensor(
            data=all_sample_events,
            lengths=all_sample_lengths,
            is_right_aligned=True
        )

        input_semantic_ids[~attention_mask] = self.config.pad_token_id
        input_semantic_ids = torch.cat(
            [self._sem_id_len * self._codebook_size + inputs['hashed_user.ids'][:, None],
             input_semantic_ids],
            dim=-1
        )
        attention_mask = torch.cat([
            attention_mask, torch.ones(batch_size, 1, device=attention_mask.device, dtype=attention_mask.dtype)
        ], dim=-1)

        if self.training:
            positive_sample_events = inputs["semantic_labels.ids"]  # (batch_size * sem_id_len)
            positive_sample_lengths = inputs["semantic_labels.length"]  # (batch_size)
            offsets = (torch.arange(start=0, end=positive_sample_events.shape[0], device=positive_sample_events.device,
                                    dtype=torch.long) % self._sem_id_len) * self._codebook_size
            positive_sample_events = positive_sample_events + offsets

            target_semantic_ids, _ = create_masked_tensor(
                data=positive_sample_events,
                lengths=positive_sample_lengths,
                is_right_aligned=True
            )
            target_semantic_ids = torch.cat(
                [torch.ones(batch_size, 1, dtype=torch.long,
                            device=target_semantic_ids.device) * self.config.decoder_start_token_id,
                 target_semantic_ids],
                dim=-1
            )

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
                early_stopping=False
            )
            return {
                'predictions': output[:, 1:].reshape(-1, self._num_return_sequences, self._sem_id_len)
            }
