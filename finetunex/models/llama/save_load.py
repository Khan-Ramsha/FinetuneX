import os
import torch
import json
from transformers import AutoModelForCausalLM
from finetunex.base.config import Config
from finetunex.models.llama.model import LlamaModel

hf_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B") #from hf
config = Config.config_from_model("Llama-3.2-1B")
model = LlamaModel(config=config) #self implemented architecture

def save_pretrained(outputdir, model_state_dict, config):
    os.makedirs(outputdir, exist_ok=True)
    torch.save(model_state_dict, os.path.join(outputdir, "model.bin")) #saving trained weights
    config_dict = config.__dict__.copy()
    with open(os.path.join(outputdir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

def from_pretrained(model_path): #load the model after finetuning
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    config = Config.from_dict(config_dict)
    model = LlamaModel(config)  #Model gets initialized
    # load weights
    weights_path = os.path.join(model_path, "model.bin")
    state_dict = torch.load(weights_path)
    model.load_state_dict(state_dict, strict=False)
    return model

def load_weights_into_llama(model, config, hf_model_state_dict):
 
    def assign(left, right, tensor_name="unknown"):
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch in tensor '{tensor_name}'. Left: {left.shape}, Right: {right.shape}")
        return torch.nn.Parameter(right.clone().detach() if isinstance(right, torch.Tensor) else torch.tensor(right))
    
    model.embed_tokens.weight = assign(model.embed_tokens.weight, hf_model_state_dict['model.embed_tokens.weight'], "model.embed_tokens.weight")

    for l in range(config.num_hidden_layers):
        block = model.layers[l]
        att = block.attn
        
        # Q, K, V projections
        att.q_proj.weight = assign(
            att.q_proj.weight,
            hf_model_state_dict[f"model.layers.{l}.self_attn.q_proj.weight"],
            f"model.layers.{l}.self_attn.q_proj.weight"
        )
        att.k_proj.weight = assign(
            att.k_proj.weight,
            hf_model_state_dict[f"model.layers.{l}.self_attn.k_proj.weight"],
            f"model.layers.{l}.self_attn.k_proj.weight"
        )
        att.v_proj.weight = assign(
            att.v_proj.weight,
            hf_model_state_dict[f"model.layers.{l}.self_attn.v_proj.weight"],
            f"model.layers.{l}.self_attn.v_proj.weight"
        )

        # Output projection
        att.o_proj.weight = assign(
            att.o_proj.weight,
            hf_model_state_dict[f"model.layers.{l}.self_attn.o_proj.weight"],
            f"model.layers.{l}.self_attn.o_proj.weight"
        )
        # Attention layernorm
        model.layers[l].input_layernorm.weight = assign(
            model.layers[l].input_layernorm.weight,
            hf_model_state_dict[f"model.layers.{l}.input_layernorm.weight"],
            f"model.layers.{l}.input_layernorm.weight"
        )
        model.layers[l].mlp.gate_proj.weight = assign(
                model.layers[l].mlp.gate_proj.weight,
                hf_model_state_dict[f"model.layers.{l}.mlp.gate_proj.weight"],
                f"model.layers.{l}.mlp.gate_proj.weight"
            )
        model.layers[l].mlp.up_proj.weight = assign(
            model.layers[l].mlp.up_proj.weight,
            hf_model_state_dict[f"model.layers.{l}.mlp.up_proj.weight"],
            f"model.layers.{l}.mlp.up_proj.weight"
        )
        model.layers[l].mlp.down_proj.weight = assign(
            model.layers[l].mlp.down_proj.weight,
            hf_model_state_dict[f"model.layers.{l}.mlp.down_proj.weight"],
            f"model.layers.{l}.mlp.down_proj.weight"
        )

        model.layers[l].post_attention_layernorm.weight = assign(
            model.layers[l].post_attention_layernorm.weight,
            hf_model_state_dict[f"model.layers.{l}.post_attention_layernorm.weight"],
            f"model.layers.{l}.post_attention_layernorm.weight"
        )
    model.norm.weight = assign(model.norm.weight, hf_model_state_dict["model.norm.weight"], "model.norm.weight")

    if "lm_head.weight" in hf_model_state_dict:
        model.lm_head.weight = assign(model.lm_head.weight, hf_model_state_dict["lm_head.weight"], "lm_head.weight")
    else:
        # Model uses tie embedding, hence assigning token embeddings 
        print("Model uses weight tying.")
        model.lm_head.weight = assign(model.lm_head.weight, hf_model_state_dict["model.embed_tokens.weight"], "model.embed_tokens.weight")