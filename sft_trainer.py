import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import from_pretrained, save_pretrained, set_seed
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup
)
from finetunex.models.qwen2.save_load import load_weights_into_qwen
from finetunex.models.llama.save_load import load_weights_into_llama
from finetunex.base.config import Config
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from evaluate import evaluate_model
from early_stopping import EarlyStopping
from sft_config import SFTConfig
from finetunex.models.qwen2.model import Qwen2Model
from finetunex.models.llama.model import LlamaModel
import wandb

class SFT:
    def __init__(self, model: str, pad_token: int, args: SFTConfig):
        self.args = args
        self.model_name = model
        self.pad_token = pad_token
        set_seed(self.args.seed)
        config = Config.config_from_model(self.model_name)
        if self.model_name == "Qwen2.5-0.5B":
            hf_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B") #from hf
            hf_model_state_dict = hf_model.state_dict()
            del hf_model
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
            self.model = Qwen2Model(config=config, args=self.args) #self implemented architecture
            load_weights_into_qwen(self.model, config, hf_model_state_dict)
        if self.model_name == "Llama-3.2-1B":
            hf_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B") #from hf
            hf_model_state_dict = hf_model.state_dict()
            del hf_model
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
            self.model = LlamaModel(config=config, args=self.args) #self implemented architecture
            load_weights_into_llama(self.model, config, hf_model_state_dict)
        # self.model.gradient_checkpointing_enable()
        # self.model.config.use_cache = False
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def prepare_dataset(self, dataset):
        def _apply_chat_template_ids(tokenizer, *args, **kwargs):
            """Return a plain list of token ids; newer transformers may return BatchEncoding."""
            out = tokenizer.apply_chat_template(*args, **kwargs)
            if isinstance(out, list):
                print('it is a list')
                return out
            ids = out["input_ids"] if hasattr(out, "input_ids") else out
            if hasattr(ids, "tolist"):
                ids = ids.tolist()
            return list(ids)

        def tokenize(example, tokenizer):
            messages = example["messages"]
            assistant_idx = next((i for i, m in enumerate(messages) if m["role"] == "assistant"), None)
            if assistant_idx is None:
                raise ValueError("No assistant message found!")

            user_prompt = messages[:assistant_idx]
            full_convo = messages[:assistant_idx + 1]

            if self.model_name == "Qwen2.5-0.5B":
                user_prompt_ids = _apply_chat_template_ids(
                    tokenizer,
                    user_prompt,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors=None,
                )
                full_convo_ids = _apply_chat_template_ids(
                    tokenizer,
                    full_convo,
                    tokenize=True,
                    add_generation_prompt=False,
                    return_tensors=None,
                )
            if self.model_name == "Llama-3.2-1B":
                user_prompt_ids = _apply_chat_template_ids(
                    tokenizer,
                    user_prompt,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors=None,
                )
                full_convo_ids = _apply_chat_template_ids(
                    tokenizer,
                    full_convo,
                    tokenize=True,
                    add_generation_prompt=False,
                    return_tensors=None,
                )

            if full_convo_ids[-1] != self.tokenizer.eos_token_id:
                full_convo_ids.append(self.tokenizer.eos_token_id)

            completion_mask = [0] * len(user_prompt_ids) + [1] * (len(full_convo_ids) - len(user_prompt_ids))

            if sum(completion_mask) == 0:
                print(f"Prompt Ids: {tokenizer.decode(user_prompt_ids)}")
                print(f"Full Conversation Ids: {tokenizer.decode(full_convo_ids)}")
                raise ValueError("No completion tokens to learn from!")

            assert len(full_convo_ids) == len(completion_mask), "Length Mismatch!"
            return {
                "input_ids": full_convo_ids,
                "completion_mask": completion_mask
            }

        dataset = dataset.map(tokenize, fn_kwargs={"tokenizer": self.tokenizer})
        return dataset

    def _save_checkpoint(self):
        
        return

    def train_model(self, dataset, data_collator, eval_dataset):

        wandb.init(
            project="finetunex-sft",
            config=vars(self.args),
            name=f"{self.model_name}-{self.args.epochs}epochs"
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.train()

        data_loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True, collate_fn=data_collator)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)

        # number of optimizer updates per epoch (round up so the trailing partial window still counts)
        steps_per_epoch = math.ceil(len(data_loader) / self.args.gradient_accumulation_steps)
        total_steps = steps_per_epoch * self.args.epochs
        warmup_steps = int(0.1 * total_steps)  # 10% warmup
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        early_stop = EarlyStopping()
        global_step = 0
        num_microbatches = len(data_loader)

        for epoch in range(self.args.epochs):
            epoch_loss = 0.0
            window_loss, window_count = 0.0, 0
            optimizer.zero_grad(set_to_none=True)

            for step, batch in enumerate(data_loader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = self.model(input_ids=input_ids, attn_mask=attention_mask, labels=labels)
                raw_loss = outputs['loss']
                loss = raw_loss / self.args.gradient_accumulation_steps

                loss.backward()
                epoch_loss += raw_loss.item()
                window_loss += raw_loss.item()
                window_count += 1

                is_accum_boundary = (step + 1) % self.args.gradient_accumulation_steps == 0
                is_last_microbatch = (step + 1) == num_microbatches

                if is_accum_boundary or is_last_microbatch:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    global_step += 1

                    step_loss = window_loss / max(window_count, 1)
                    window_loss, window_count = 0.0, 0

                    wandb.log({
                        "loss": step_loss,
                        "learning_rate": scheduler.get_last_lr()[0],
                        "epoch": epoch,
                        "global_step": global_step
                    })

            avg_epoch_loss = epoch_loss / num_microbatches
            wandb.log({"epoch_loss": avg_epoch_loss, "epoch": epoch})

            if eval_dataset is not None:
                eval_metrics = evaluate_model(self.model, eval_dataset, data_collator, self.args.batch_size, device)
                self.model.train()
                wandb.log({"eval_loss": eval_metrics["loss"], "epoch": epoch, "perplexity": eval_metrics["perplexity"]})
                if early_stop.early_stopping(eval_metrics["loss"]):
                    wandb.log(f"Early stopping triggered at {epoch}")
                    break

        self._save_checkpoint()
        wandb.finish()