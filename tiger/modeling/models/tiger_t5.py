from typing import Dict

import torch
from transformers import T5ForConditionalGeneration, T5Config

from ..models import TorchModel
from ..utils import create_masked_tensor, DEVICE



class TigerModelT5(TorchModel):
    def __init__(
            self,
            sequence_prefix,
            positive_prefix,
            embedding_dim,
            codebook_size,
            num_positions,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            dim_feedforward,
            dropout=0.0,
            activation='relu',
            layer_norm_eps=1e-9,
            initializer_range=0.02,
    ):
        super().__init__()

        self._sequence_prefix = sequence_prefix
        self._positive_prefix = positive_prefix
        self._embedding_dim = embedding_dim
        self._codebook_size = codebook_size
        self._num_positions = num_positions
        self._num_heads = num_heads
        self._num_encoder_layers = num_encoder_layers
        self._num_decoder_layers = num_decoder_layers
        self._dim_feedforward = dim_feedforward
        self._layer_norm_eps = layer_norm_eps
        self._sem_id_len = 4

        unified_vocab_size = codebook_size * self._sem_id_len + 2000 + 10  # 2000 for user ids, 10 for utilities
        t5_config = T5Config(
            vocab_size=unified_vocab_size,
            d_model=embedding_dim,
            d_kv=64,
            d_ff=dim_feedforward,
            num_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            num_heads=num_heads,
            dropout_rate=dropout,
            is_encoder_decoder=True,
            use_cache=False,
            pad_token_id=unified_vocab_size - 1,
            eos_token_id=unified_vocab_size - 2,
            decoder_start_token_id=unified_vocab_size - 3,
            tie_word_embeddings=False
        )
        self.config = t5_config
        self.model = T5ForConditionalGeneration(config=t5_config)
        self._init_weights(initializer_range)

    def forward(self, inputs):
        all_sample_events = inputs["semantic_{}.ids".format(self._sequence_prefix)]  # (all_batch_events)
        all_sample_lengths = inputs["semantic_{}.length".format(self._sequence_prefix)] # (batch_size)
        offsets = (torch.arange(start=0, end=all_sample_events.shape[0], device=all_sample_events.device, dtype=torch.long) % 4) * self._codebook_size
        all_sample_events = all_sample_events + offsets

        batch_size = all_sample_lengths.shape[0]

        input_semantic_ids, attention_mask = create_masked_tensor(
            data=all_sample_events,
            lengths=all_sample_lengths,
            is_tiger=True
        )

        input_semantic_ids[~attention_mask] = self.config.pad_token_id
        input_semantic_ids = torch.cat(
            [self._sem_id_len * self._codebook_size + (inputs['user.ids'][:, None] % 2000), input_semantic_ids],
            dim=-1
        )
        attention_mask = torch.cat([
            attention_mask, torch.ones(batch_size, 1, device=attention_mask.device, dtype=attention_mask.dtype)
        ], dim=-1)

        if self.training:
            positive_sample_events = inputs["semantic_{}.ids".format(self._positive_prefix)]  # (batch_size * 4)
            positive_sample_lengths = inputs["semantic_{}.length".format(self._positive_prefix)]  # (batch_size)
            offsets = (torch.arange(start=0, end=positive_sample_events.shape[0], device=positive_sample_events.device, dtype=torch.long) % 4) * self._codebook_size
            positive_sample_events = positive_sample_events + offsets

            target_semantic_ids, _ = create_masked_tensor(
                data=positive_sample_events,
                lengths=positive_sample_lengths,
                is_tiger=True
            )
            target_semantic_ids = torch.cat(
                [torch.ones(batch_size, 1, dtype=torch.long, device=target_semantic_ids.device) * self.config.decoder_start_token_id, target_semantic_ids],
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
                num_beams=50,
                num_return_sequences=20,
                max_length=5,
                decoder_start_token_id=self.config.decoder_start_token_id,
                eos_token_id=self.config.eos_token_id,
                pad_token_id=self.config.pad_token_id,
                do_sample=False,
                early_stopping=False,
                # logits_processor=[CorrectItemsLogitsProcessor(num_codebooks=self._sem_id_len, codebook_size=self._codebook_size)],
            )
            return {
                'predictions': output[:, 1:].reshape(-1, 20, 5 - 1)
            }


    @classmethod
    def create_from_config(cls, config: Dict, **kwargs):
        return cls(
            sequence_prefix=config["sequence_prefix"],
            positive_prefix=config['positive_prefix'],
            embedding_dim=config["embedding_dim"],
            codebook_size=config["codebook_size"],
            num_positions=config["num_positions"],
            num_heads=config.get("num_heads", int(config["embedding_dim"] // 64)),
            num_encoder_layers=config["num_encoder_layers"],
            num_decoder_layers=config["num_decoder_layers"],
            dim_feedforward=config.get("dim_feedforward", 4 * config["embedding_dim"]),
            dropout=config.get("dropout", 0.0),
            initializer_range=config.get("initializer_range", 0.02),
        )

