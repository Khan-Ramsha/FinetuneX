"""Transformer Block & main architecture"""
import torch
import torch.nn as nn
from finetunex.modules.attention import GroupQueryAttention
from finetunex.modules.norm import RMSNorm
from finetunex.modules.mlp import MLP
from finetunex.base.config import Config
from finetunex.modules.positional_encoding import RotaryEmbedding

class DecoderBlock(nn.Module):
    def __init__(self, config: Config, layer_idx : int):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size)
        self.attn = GroupQueryAttention(config.hidden_size, config.num_attention_heads, config.num_key_value_heads, layer_idx = layer_idx)
        self.post_attention_layernorm = RMSNorm(config.hidden_size)
        self.mlp = MLP(config)
    
    def forward(self, x,pos_emb=None, attention_mask = None):
        x = x + self.attn(self.input_layernorm(x), pos_emb, attention_mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x

class Qwen2(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size) #Token embedding
        self.layers = nn.ModuleList(
            [DecoderBlock(self.config, layer_idx) for layer_idx in range(self.config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size)
        self.rotary = RotaryEmbedding(self.config.hidden_size // self.config.num_attention_heads)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight #tie embeddings
        
    def forward(self, x, attention_mask = None):
        B, T = x.shape
        hidden_states = self.embed_tokens(x) 
        position_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        pos_emb = self.rotary(hidden_states, position_ids)
        # passing hidden states to stack of blocks
        for layer in self.layers:
            hidden_states = layer(hidden_states,pos_emb, attention_mask)
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits

from transformers import AutoModelForCausalLM, AutoTokenizer
from finetunex.text_generation.generate import generate
hf_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B") #from hf

# hf_state = hf_model.state_dict()
# print(list(hf_state.keys())[:50]) 
config = Config.config_from_model("Qwen2.5-0.5B")
model = Qwen2(config=config) #self implemented architecture

def mapping(name):
    hf_name = f"model.{name}" #make it same as hf naming conventions
    hf_name = hf_name.replace(".attn.", ".self_attn.")
    return hf_name

def load_pretrained_weights(model, hf_model):
    model_state_dict = model.state_dict()
    hfmodel_state_dict = hf_model.state_dict()
    for name, params in list(model_state_dict.items()):
        if "rotary.inv_freq" in name:
            continue #skip it
        hf_name = mapping(name)
        if hf_name in hfmodel_state_dict:
            model_state_dict[name] = hfmodel_state_dict[hf_name] #manually assigning weights
        else:
            print(f"Skipping, {hf_name} is not in hf state dict")
    return model_state_dict

model_state_dict = load_pretrained_weights(model, hf_model)
model.load_state_dict(model_state_dict, strict = False)
model.eval()
print("GENERATING RESPONSE")
prompt="What is capital of France?"
print(f"User prompt: {prompt}")

messages = [
    {"role": "user", "content": prompt}
]
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
chat_input = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
print(f"Chat input: {repr(chat_input)}")
inputs = tokenizer(chat_input, return_tensors = "pt") #tokenize the prompt
inputs= inputs["input_ids"]
print(inputs)
max_new_tokens = model.config.max_position_embeddings 
print(f"Max tokens by qwen2 0.5B: {max_new_tokens}")
with torch.no_grad():
    outputs = generate(
        model,
        inputs,
        max_new_tokens = max_new_tokens,
        top_p = 0.6, 
        top_k = 100,
        temperature = 0.5,
        stop_tokens=([tokenizer.eos_token_id],)
    )
output = list(outputs)
token_ids = [token.item() for token in output] 
full_response = tokenizer.decode(token_ids)
print(f"\nFull response:\n{full_response}")