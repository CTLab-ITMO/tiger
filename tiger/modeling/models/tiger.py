import torch
import torch.nn as nn

from models import TorchModel
from utils import get_activation_function, create_masked_tensor, DEVICE

class TigerModel(TorchModel, config_name='tiger'):
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

        self._user_embedding = nn.Embedding(
            num_embeddings=2000,
            embedding_dim=self._embedding_dim
        )
        self.codebook_embeddings = nn.Embedding(
            num_embeddings=(self._codebook_size * self._sem_id_len),
            embedding_dim=self._embedding_dim
        )
        self.position_embeddings = nn.Embedding(
            num_embeddings=self._num_positions * 4, 
            embedding_dim=self._embedding_dim
        )
        self.sem_id_position_embeddings = nn.Embedding(
            num_embeddings=self._sem_id_len,
            embedding_dim=self._embedding_dim
        )
        self.bos_embedding = nn.Parameter(torch.randn(self._embedding_dim))

        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self._embedding_dim,
            nhead=self._num_heads,
            dim_feedforward=self._dim_feedforward,
            dropout=dropout,
            activation=get_activation_function(activation),
            layer_norm_eps=self._layer_norm_eps,
            batch_first=True,
        )
        transformer_decoder_layer = nn.TransformerDecoderLayer(
            d_model=self._embedding_dim,
            nhead=self._num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=get_activation_function(activation),
            layer_norm_eps=self._layer_norm_eps,
            batch_first=True,
        )

        self._encoder = nn.TransformerEncoder(transformer_encoder_layer, num_encoder_layers)
        self._decoder = nn.TransformerDecoder(transformer_decoder_layer, num_decoder_layers)

        self._layernorm = nn.LayerNorm(self._embedding_dim, eps=layer_norm_eps)
        self._dropout = nn.Dropout(dropout)

        self._init_weights(initializer_range)

        self.scale = torch.nn.Parameter(torch.tensor(0.0))

    @classmethod
    def create_from_config(cls, config, **kwargs):
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

    def _embed_semantic_tokens(self, sem_ids: torch.LongTensor) -> torch.Tensor:
        """
        Переводит семантические айдишники в токены

        sem_ids: (N,)
        embeds: (N, embedding_dim)
        """

        sem_ids_positions = torch.arange(sem_ids.size(0), device=DEVICE) % self._sem_id_len  # (N,)
        sem_id_offsets = sem_ids_positions * self._codebook_size + sem_ids
        assert sem_id_offsets.shape == sem_ids.shape
        return self.codebook_embeddings(sem_id_offsets)  # (N , embedding_dim)

    def _get_position_embeddings(self, mask: torch.BoolTensor) -> torch.Tensor:
        batch_size, seq_len = mask.shape
        position_ids = torch.arange(seq_len, device=DEVICE).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embeddings(position_ids)  # (batch_size, max_seq_len, embedding_dim)
        pos_emb[~mask] = 0.0
        return pos_emb  # (batch_size, max_seq_len, embedding_dim)

    def _get_sem_ids_position_embeddings(self, mask: torch.BoolTensor) -> torch.Tensor:
        batch_size, seq_len = mask.shape
        position_ids = torch.arange(seq_len, device=DEVICE).unsqueeze(0).expand(batch_size, -1)
        sem_pos_ids = position_ids.remainder(self._sem_id_len)
        sem_pos_emb = self.sem_id_position_embeddings(sem_pos_ids)  # (batch_size, max_seq_len, embedding_dim)
        sem_pos_emb[~mask] = 0.0
        return sem_pos_emb  # (batch_size, max_seq_len, embedding_dim)

    def _get_last_sem_ids_mask(self, all_sample_lengths: torch.Tensor) -> torch.Tensor:
        """Создает маску для последних sem_id_len токенов каждой последовательности"""
        total_tokens = all_sample_lengths.sum().item()
        tgt_end_idx = torch.cumsum(all_sample_lengths, dim=0)  # (batch_size)
        tgt_start_idx = tgt_end_idx - self._sem_id_len  # (batch_size)

        mask_flat_extended = torch.zeros(total_tokens + 1, dtype=torch.int, device=DEVICE)
        mask_flat_extended[tgt_start_idx] += 1
        mask_flat_extended[tgt_end_idx] -= 1

        mask_flat = torch.cumsum(mask_flat_extended, dim=0)[:total_tokens]
        return mask_flat.bool()
    
    def _prepare_sem_id_batch(
             self,
             encoder_embeddings_flat: torch.Tensor,  # (total_tokens, embedding_dim)
             encoder_lengths: torch.LongTensor,  # (batch_size,) 
             decoder_embeddings_flat: torch.Tensor,  # (total_tokens, embedding_dim)
             decoder_lengths: torch.LongTensor  # (batch_size,)
     ):
        """
        Формирует входы для энкодера и декодера
        
        """
        batch_size = encoder_lengths.size(0)
        # sem_id_len = self._sem_id_len

        # маска последних sem_id_len каждого батча
        # decoder_mask_flat = self._get_last_sem_ids_mask(lengths)  # (all_batch_semantic_ids)

        # разделение плоского тензора
        # encoder_emb_flat = embeddings_flat[~decoder_mask_flat]  # (all_batch_encoder_semantic_ids)
        # encoder_lengths = lengths - sem_id_len
        
        # decoder_emb_flat = embeddings_flat[decoder_mask_flat]  # (all_batch_decoder_semantic_ids)
        # decoder_lengths = torch.ones_like(lengths) * sem_id_len

        # эмбеддинги уже с добавленной размерностью max_seq_len и позиционными
        encoder_embeddings, encoder_mask = self._create_encoder_tensors(
            encoder_embeddings_flat, encoder_lengths
        )  # (batch_size, max_encoder_semantic_seq_len, embedding_dim), (batch_size, max_semantic_seq_len)
        decoder_embeddings = self._create_decoder_tensors(
            decoder_embeddings_flat, decoder_lengths
        )  # (batch_size, max_decoder_semantic_seq_len, embedding_dim)

        # BOS
        encoder_embeddings, encoder_mask = self._add_bos_to_encoder(encoder_embeddings, encoder_mask, batch_size)
        decoder_embeddings = self._add_bos_to_decoder(decoder_embeddings, batch_size)

        return encoder_embeddings, encoder_mask, decoder_embeddings

    def _create_encoder_tensors(self, encoder_emb_flat, encoder_lengths):
        """
        Переводит в dense формат, добавляет позиции
        """
        encoder_embeddings, encoder_mask = create_masked_tensor(encoder_emb_flat, encoder_lengths)

        # позиционные эмбеддинги
        pos_emb = self._get_position_embeddings(encoder_mask)
        sem_pos_emb = self._get_sem_ids_position_embeddings(encoder_mask)
        encoder_embeddings = encoder_embeddings + pos_emb + sem_pos_emb

        return encoder_embeddings, encoder_mask
    
    def _create_decoder_tensors(self, decoder_emb_flat, decoder_lengths):
        """
        Переводит в dense формат, добавляет позиции
        """
        decoder_embeddings, decoder_mask = create_masked_tensor(decoder_emb_flat, decoder_lengths)

        # позиционные эмбеддинги (только семантические)
        sem_pos_emb = self._get_sem_ids_position_embeddings(decoder_mask)
        decoder_embeddings = decoder_embeddings + sem_pos_emb

        return decoder_embeddings
    
    def _add_bos_to_encoder(self, encoder_embeddings, encoder_mask, batch_size):
        bos = self.bos_embedding[None, None].tile(dims=[batch_size, 1, 1])
        new_encoder_embeddings = torch.cat([bos, encoder_embeddings], dim=1)
        new_mask = torch.cat([
            torch.ones(batch_size, 1, dtype=torch.bool, device=DEVICE),
            encoder_mask
        ], dim=1)
        return new_encoder_embeddings, new_mask

    def _add_bos_to_decoder(self, decoder_embeddings, batch_size):
        bos = self.bos_embedding[None, None].tile(dims=[batch_size, 1, 1])
        new_decoder_embeddings = torch.cat([bos, decoder_embeddings], dim=1)
        return new_decoder_embeddings

    def forward(self, inputs):
        all_sample_events = inputs["semantic_{}.ids".format(self._sequence_prefix)]  # (all_batch_events)
        all_sample_lengths = inputs["semantic_{}.length".format(self._sequence_prefix)] # (batch_size)
        assert all_sample_events.shape[0] == sum(all_sample_lengths)

        positive_sample_events = inputs["semantic_{}.ids".format(self._positive_prefix)]  # (all_batch_events)
        positive_sample_lengths = inputs["semantic_{}.length".format(self._positive_prefix)]  # (all_batch_events)

        encoder_sem_id_embeddings = self._embed_semantic_tokens(all_sample_events)  # (all_batch_sids, embedding_dim)
        decoder_sem_id_embeddings = self._embed_semantic_tokens(positive_sample_events)

        assert encoder_sem_id_embeddings.shape[0] == sum(all_sample_lengths)

        (encoder_input_emb,  # (batch_size, seq_len - sem_id_len + 1, embedding_dim)
        encoder_input_mask,  # (batch_size, seq_len - sem_id_len + 1)
        decoder_input_embs) = (  # (batch_size, sem_id_len + 1, embedding_dim)
            self._prepare_sem_id_batch(encoder_sem_id_embeddings, all_sample_lengths, decoder_sem_id_embeddings, positive_sample_lengths)
        )

        user_ids = inputs['hashed_user.ids']  # (batch_size)
        user_embeddings = self._user_embedding(user_ids)
        encoder_input_emb[:, 0] = user_embeddings
        after_encoder_emb, after_encoder_mask = self._apply_encoder(encoder_input_emb, encoder_input_mask)

        if self.training:
            # # последние sem ids
            # target_tokens_mask = self._get_last_sem_ids_mask(all_sample_lengths)
            target_tokens = positive_sample_events.reshape(-1, self._sem_id_len)  # (batch_size, sem_id_len)

            # Подготовка входа декодера (BOS + первые 3 токена)
            tgt = decoder_input_embs[:, :-1, :]  # (batch_size, sem_id_len, embedding_dim)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                tgt.size(1), device=DEVICE
            )  # (sem_id_len, sem_id_len)

            decoder_output = self._decoder(
                tgt=tgt,
                memory=after_encoder_emb,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=~after_encoder_mask
                # должно быть True для паддинга и False для реальных позиций
            )  # (batch_size, sem_id_len, embedding_dim)

            losses = []  # [(batch_size, codebook_size) * sem_id_len]
            scores = []  # [(batch_size, codebook_size) * sem_id_len]
            argmaxes = []  # [(batch_size, ) * sem_id_len]

            for i in range(self._sem_id_len):
                weights = self.codebook_embeddings.weight[i * self._codebook_size: (i + 1) * self._codebook_size]
                logits = decoder_output[:, i, :] @ weights.T # (batch_size, codebook_size)
                scores.append(logits)

                pred_tokens = torch.argmax(logits, dim=-1)  # (batch_size,)
                argmaxes.append(pred_tokens)

                loss = -torch.gather(
                    torch.log_softmax(logits / torch.clip(torch.exp(self.scale), min=0.01, max=100), dim=-1),
                    dim=1,
                    index=target_tokens[:, i][..., None]
                ).mean()
                
                # loss = nn.functional.cross_entropy(logits, target_tokens[:, i])
                losses.append(loss)

            return {
                "decoder_loss_1": losses[0],  # (1, )
                "decoder_loss_2": losses[1],  # (1, )
                "decoder_loss_3": losses[2],  # (1, )
                "decoder_loss_4": losses[3],  # (1, )

                "decoder_scores_1": scores[0],  # (batch_size, codebook_size)
                "decoder_scores_2": scores[1],  # (batch_size, codebook_size)
                "decoder_scores_3": scores[2],  # (batch_size, codebook_size)
                "decoder_scores_4": scores[3],  # (batch_size, codebook_size)

                "decoder_argmax_1": argmaxes[0],  # (batch_size, )
                "decoder_argmax_2": argmaxes[1],  # (batch_size, )
                "decoder_argmax_3": argmaxes[2],  # (batch_size, )
                "decoder_argmax_4": argmaxes[3],  # (batch_size, )

                "scale": torch.exp(self.scale).item(),
            }
        else:
            # target_tokens_mask = self._get_last_sem_ids_mask(all_sample_lengths)
            target_tokens = positive_sample_events.reshape(-1, self._sem_id_len)  # (batch_size, sem_id_len)
            # target_tokens = all_sample_events[target_tokens_mask].view(-1, self._sem_id_len)

            batch_size = encoder_input_emb.size(0)

            tgt = self.bos_embedding[None, None].tile(dims=[batch_size, 1, 1])  # (batch_size, 1, embedding_dim)

            memory_key_padding_mask = ~after_encoder_mask

            argmaxes = []
            scores = []
            losses = []

            for step in range(self._sem_id_len):
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                    tgt.size(1), device=DEVICE
                )  # (L, L)

                decoder_output = self._decoder(
                    tgt=tgt,
                    memory=after_encoder_emb,
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=memory_key_padding_mask
                )  # (batch_size, L, embedding_dim)

                last_output = decoder_output[:, -1, :]  # (batch_size, 1, embedding_dim)
                weights = self.codebook_embeddings.weight[step * self._codebook_size: (step + 1) * self._codebook_size]
                logits = last_output @ weights.T  # (batch_size, codebook_size)
                scores.append(logits)

                pred_tokens = torch.argmax(logits, dim=-1)  # (batch_size,)
                argmaxes.append(pred_tokens)

                loss = nn.functional.cross_entropy(logits, target_tokens[:, step])
                losses.append(loss)

                if step < self._sem_id_len - 1:
                    next_embed = self.codebook_embeddings(
                        step * self._codebook_size + pred_tokens)  # (batch_size, embedding_dim)
                    pos_emb = self.sem_id_position_embeddings(
                        torch.tensor([step], device=DEVICE)
                    ).expand(batch_size, -1)
                    next_embed += pos_emb

                    next_embed = next_embed.unsqueeze(1)  # (batch_size, 1, embedding_dim)
                    tgt = torch.cat([tgt, next_embed], dim=1)
                
            all_items_semantic_ids = inputs['all_semantic_ids']  # (num_items, sid_length)
            all_items_semantic_ids = all_items_semantic_ids + 256 * torch.arange(4, device=all_items_semantic_ids.device)

            decoder_scores = torch.softmax(torch.stack(scores, dim=1) / torch.clip(torch.exp(self.scale), min=0.01, max=100), dim=-1)
            decoder_scores = decoder_scores.reshape(decoder_scores.shape[0], decoder_scores.shape[1] * decoder_scores.shape[2])

            all_items, id_dim = all_items_semantic_ids.shape
            batch_indices = torch.arange(batch_size).unsqueeze(1).unsqueeze(2)
            ids_expanded = all_items_semantic_ids.unsqueeze(0).expand(batch_size, -1, -1)

            all_item_scores = decoder_scores[batch_indices.expand(-1, all_items, id_dim), ids_expanded]  # (batch_size, num_items, sid_length)
            all_item_scores = all_item_scores.prod(dim=-1)

            sort_indices = torch.argsort(all_item_scores, dim=-1, descending=True, stable=True)
            # import code; code.interact(local=locals())
            # batch_size, num_items, sid_length = all_item_scores.shape
        
            # indices = torch.arange(num_items, device=all_item_scores.device).unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, 4)
            
            # for i in range(sid_length - 1, -1, -1):  # sid_length-1, sid_length-2, ..., 1, 0
            #     key_values = torch.gather(all_item_scores, dim=1, index=indices)[:, :, i]
            #     sort_indices = torch.argsort(key_values, dim=1, descending=True, stable=True).unsqueeze(-1).expand(batch_size, -1, 4)
            #     indices = torch.gather(indices, dim=1, index=sort_indices)
            
            # print(indices[:2, :2])
            # indices = indices[:, :, 0]
        
            return {
                "decoder_scores_1": scores[0],
                "decoder_scores_2": scores[1],
                "decoder_scores_3": scores[2],
                "decoder_scores_4": scores[3],

                "decoder_argmax_1": argmaxes[0],
                "decoder_argmax_2": argmaxes[1],
                "decoder_argmax_3": argmaxes[2],
                "decoder_argmax_4": argmaxes[3],

                "decoder_loss_1": losses[0],  # (1, )
                "decoder_loss_2": losses[1],  # (1, )
                "decoder_loss_3": losses[2],  # (1, )
                "decoder_loss_4": losses[3],  # (1, )

                "predictions": sort_indices,
                "scale": torch.exp(self.scale).item(),
            }

    def _apply_encoder(
            self,
            embeddings,  # (batch_size, max_seq_len, embedding_dim)
            mask,  # (batch_size, max_seq_len)
    ):
        assert embeddings.shape[0] == mask.shape[0]
        assert embeddings.shape[1] == mask.shape[1]

        embeddings = self._encoder(
            src=embeddings, src_key_padding_mask=~mask
        )  # (batch_size, seq_len, embedding_dim)

        return embeddings, mask