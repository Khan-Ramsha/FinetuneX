from finetunex.models.qwen2.model import Qwen2Model
from finetunex.models.llama.model import LlamaModel
import random
import numpy as np
import torch
import os
import json
from finetunex.base.config import Config
from sft_config import SFTConfig

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    arg = SFTConfig()
    if config.model_type == "qwen2":    
        model = Qwen2Model(config, args = arg)  #Model gets initialized
    elif config.model_type == "llama":
        model = LlamaModel(config, args = arg)  #Model gets initialized
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")
    # load weights
    weights_path = os.path.join(model_path, "model.bin")
    state_dict = torch.load(weights_path, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    return model

def token_accuracy(logits, labels):
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    preds = shift_logits.argmax(dim=-1)
    mask = shift_labels != -100
    correct = ((preds == shift_labels) & mask).sum().item()
    total = mask.sum().item()

    return correct, total