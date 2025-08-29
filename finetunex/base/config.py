""" Config structure for all models & variants"""
from dataclasses import dataclass

@dataclass
class Config:
    model_name: str = ""
    model_type: str = ""
    hidden_size: int =  1536 #dim per token
    hidden_act: str = "silu"
    intermediate_size : int  = 8960
    num_attention_heads : int = 12
    num_hidden_layers : int = 28
    num_key_value_heads : int = 2
    vocab_size : int = 151936
    max_position_embeddings : int = 32768 # max seq len
    tie_word_embeddings : bool = True
    rms_norm_eps: int = 1e-06
    rope_theta: float = 1000000.0

    _available_models = []

    @classmethod
    def config_from_model(cls, name: str, **kwargs):
        for cfg in cls._available_models:
            if cfg["model_name"] == name:
                config_dict = cfg.copy()
                config_dict.update(kwargs) # custom parameters
                return cls(**config_dict)
        raise ValueError(f"Unknown Model: {name}")
    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)
    
qwen2_variants = [
    dict(
        model_name = "Qwen2.5-0.5B",
        model_type= "qwen2",
        hidden_size =  896, #dim per token
        hidden_act = "silu",
        intermediate_size = 4864,
        num_attention_heads = 14,
        num_hidden_layers = 24,
        num_key_value_heads = 2,
        vocab_size = 151936,
        max_position_embeddings = 32768, # max seq len
        tie_word_embeddings = True,
        rms_norm_eps = 1e-06,
        rope_theta =  1000000.0
    ),

    dict(
        model_name = "Qwen2.5-1.5B",
        model_type= "qwen2",
        hidden_size =  1536, #dim per token
        hidden_act = "silu",
        intermediate_size = 8960,
        num_attention_heads = 12,
        num_hidden_layers = 28,
        num_key_value_heads = 2,
        vocab_size = 151936,
        max_position_embeddings = 131072,
        tie_word_embeddings = True,
        rms_norm_eps = 1e-06,
        rope_theta = 1000000.0
    )  
]

llama_variants = [
    dict(
        model_name = "Llama-3.2-1B",
        model_type= "llama",
        hidden_size =  2048, 
        hidden_act = "silu",
        intermediate_size = 8192,
        num_attention_heads = 32,
        num_hidden_layers = 16,
        num_key_value_heads = 8,
        vocab_size = 128256,
        max_position_embeddings = 131072, # max seq len
        tie_word_embeddings = True,
        rms_norm_eps = 1e-05,
        rope_theta = 500000.0

    )   
]

Config._available_models.extend(qwen2_variants)
Config._available_models.extend(llama_variants)