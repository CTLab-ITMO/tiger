import torch
from transformers import T5ForConditionalGeneration, T5Config
from typing import Dict, Optional
from dataclasses import dataclass

from models import TorchModel
from utils import create_masked_tensor, DEVICE


@dataclass
class TigerRecommenderConfig:
    """Конфигурация для TIGER рекомендательной системы"""
    
    # Архитектура T5
    d_model: int = 768                    # Размерность модели
    d_ff: int = 3072                      # Размерность feed-forward
    num_layers: int = 12                  # Количество слоев
    num_decoder_layers: int = 12
    num_heads: int = 12                   # Количество attention heads
    dropout: float = 0.1                  # Dropout rate
    
    # Семантические ID параметры
    num_semantic_ids: int = 4
    semantic_vocab_size: int = 256
    
    # Унифицированный словарь
    unified_vocab_size: int = None
    bos_token_id: int = None
    decoder_start_token_id: int = None
    eos_token_id: int = None
    pad_token_id: int = None

    def __post_init__(self):
        # Рассчитываем размер унифицированного словаря
        self.unified_vocab_size = self.num_semantic_ids * self.semantic_vocab_size + 10
        self.bos_token_id = self.num_semantic_ids * self.semantic_vocab_size + 1
        self.decoder_start_token_id = self.num_semantic_ids * self.semantic_vocab_size + 2
        self.eos_token_id = self.num_semantic_ids * self.semantic_vocab_size + 3
        self.pad_token_id = self.num_semantic_ids * self.semantic_vocab_size + 4


class TigerModelT5(TorchModel, config_name='tiger_t5'):
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

        self.config = TigerRecommenderConfig(
            d_model=embedding_dim,
            d_ff=self._dim_feedforward,
            num_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            num_heads=num_heads,
            dropout=dropout,
            num_semantic_ids=self._sem_id_len,
            semantic_vocab_size=codebook_size
        )

        self.model = TigerRecommender(config=self.config)

        print(self)

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

    def forward(self, inputs):
        all_sample_events = inputs["semantic_{}.ids".format(self._sequence_prefix)]  # (all_batch_events)
        all_sample_lengths = inputs["semantic_{}.length".format(self._sequence_prefix)] # (batch_size)

        positive_sample_events = inputs["semantic_{}.ids".format(self._positive_prefix)]  # (all_batch_events)
        positive_sample_lengths = inputs["semantic_{}.length".format(self._positive_prefix)]  # (batch_size)

        input_semantic_ids, attention_mask = create_masked_tensor(
            data=all_sample_events,
            lengths=all_sample_lengths
        )

        target_semantic_ids, _ = create_masked_tensor(
            data=positive_sample_events,
            lengths=positive_sample_lengths
        )

        input_semantic_ids = self.model.semantic_processor.semantic_ids_to_unified(input_semantic_ids)
        input_semantic_ids[~attention_mask] = self.model.tiger_config.pad_token_id

        if self.training:
            target_semantic_ids = self.model.semantic_processor.semantic_ids_to_unified(target_semantic_ids)
            target_semantic_ids = self.model.semantic_processor.add_special_tokens(target_semantic_ids, mode="train")

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
            output = self.model.generate_recommendations(
                input_ids=input_semantic_ids,
                attention_mask=attention_mask,
                num_beams=30,
                num_return_sequences=20,
                max_length=5
            )            
            return output

class SemanticIDProcessor:
    """Класс для обработки семантических ID с offset mapping"""
    
    def __init__(self, config: TigerRecommenderConfig):
        self.config = config
        self.num_semantic_ids = config.num_semantic_ids
        self.semantic_vocab_size = config.semantic_vocab_size
        
    def semantic_ids_to_unified(self, semantic_ids: torch.Tensor) -> torch.Tensor:
        _, seq_len = semantic_ids.shape

        offsets = torch.arange(start=0, end=self.num_semantic_ids, device=semantic_ids.device) * self.semantic_vocab_size
        return semantic_ids + offsets[torch.arange(start=0, end=seq_len, device=semantic_ids.device) % 4][None, :]

    def add_special_tokens(self, unified_ids: torch.Tensor, mode: str = "train") -> torch.Tensor:
        batch_size, seq_len = unified_ids.shape
        device = unified_ids.device
        
        if mode == "train":
            # Добавляем BOS в начало
            bos_tokens = torch.full((batch_size, 1), self.config.bos_token_id, device=device)
            processed_ids = torch.cat([bos_tokens, unified_ids], dim=1)
            
        elif mode == "inference":
            # Только BOS токен для начала генерации
            processed_ids = torch.full((batch_size, 1), self.config.bos_token_id, device=device)
            
        return processed_ids


class TigerRecommender(T5ForConditionalGeneration):
    """
    TIGER рекомендательная модель на основе T5 с унифицированным словарем
    """
    
    def __init__(self, config: TigerRecommenderConfig):
        # Создаем T5 конфигурацию
        t5_config = T5Config(
            vocab_size=config.unified_vocab_size, # ok
            d_model=config.d_model,  # ok
            d_ff=config.d_ff,  # ok
            num_layers=config.num_layers,  # ok
            num_decoder_layers=config.num_layers,  # ok
            num_heads=config.num_heads,  # ok
            dropout_rate=config.dropout,  # ok
            is_encoder_decoder=True,  # ok
            use_cache=True,  # ok
            pad_token_id=config.pad_token_id,  # ok
            eos_token_id=config.eos_token_id,  # ok
            decoder_start_token_id=config.bos_token_id,  # ???
        )
        
        # Инициализируем T5 модель
        super().__init__(t5_config)
        
        # Сохраняем конфигурацию
        self.tiger_config = config
        
        # Процессор семантических ID
        self.semantic_processor = SemanticIDProcessor(config)
        
        # Инициализируем веса
        self.init_weights()

    def generate_recommendations(self,
                                input_ids: torch.Tensor,
                                attention_mask: Optional[torch.Tensor] = None,
                                num_beams: int = 20,
                                num_return_sequences: int = 20,
                                max_length: int = 4,
                                **generation_kwargs) -> Dict[str, torch.Tensor]:        
        # Используем стандартный T5 generate с beam search
        generated_ids = super().generate(
            input_ids,
            attention_mask=attention_mask,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            max_length=max_length,
            decoder_start_token_id=self.tiger_config.bos_token_id,
            eos_token_id=self.tiger_config.eos_token_id,
            pad_token_id=self.tiger_config.pad_token_id,
            do_sample=False,
            early_stopping=True,
            **generation_kwargs
        )

        return {
            'predictions': generated_ids[:, 1:].reshape(-1, num_return_sequences, max_length - 1)
        }
