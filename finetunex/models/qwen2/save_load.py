""""This file includes:
    1) load_pretrained_weights => loading and mapping pretrained weights from hf


"""
import os
import torch
import json
from transformers import AutoModelForCausalLM
from finetunex.base.config import Config
from finetunex.models.qwen2.model import Qwen2Model

hf_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B") #from hf
config = Config.config_from_model("Qwen2.5-0.5B")
model = Qwen2Model(config=config) #self implemented architecture

def load_pretrained_weights(model, hf_model): #mapping
    model_state_dict = model.state_dict()
    hfmodel_state_dict = hf_model.state_dict()
    for name, params in list(model_state_dict.items()):
        if "rotary.inv_freq" in name:
            continue #skip it
        hf_name = f"model.{name}" #make it same as hf naming conventions
        hf_name = hf_name.replace(".attn.", ".self_attn.")
        if hf_name in hfmodel_state_dict:
            
            model_state_dict[name] = hfmodel_state_dict[hf_name] #manually assigning weights
        else:
            if(hf_name == "model.lm_head.weight"):
                print(f"hf_name: {hf_name}")
                print(f"model state dict name : {model_state_dict[name]}")
                model_state_dict[name] = hfmodel_state_dict["model.embed_tokens.weight"]
    return model_state_dict # to be loaded and used for infer

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
    model = Qwen2Model(config)  #Model gets initialized
    # load weights
    weights_path = os.path.join(model_path, "model.bin")
    state_dict = torch.load(weights_path)
    model.load_state_dict(state_dict, strict=False)
    return model
print(load_pretrained_weights(model, hf_model))