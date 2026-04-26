import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import save_pretrained, set_seed, token_accuracy
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
        elif self.model_name == "Llama-3.2-1B":
            hf_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B") #from hf
            hf_model_state_dict = hf_model.state_dict()
            del hf_model
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
            self.model = LlamaModel(config=config, args=self.args) #self implemented architecture
            load_weights_into_llama(self.model, config, hf_model_state_dict)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def prepare_dataset(self, dataset):
        def _apply_chat_template_ids(tokenizer, *args, **kwargs):
            """Return a plain list of token ids; newer transformers may return BatchEncoding."""
            out = tokenizer.apply_chat_template(*args, **kwargs)
            if isinstance(out, list):
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

    def _save_checkpoint(self, optimizer, scheduler, global_step, epoch, val_loss):
        os.makedirs(self.args.output_dir, exist_ok=True)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "global_step": global_step,
            "epoch": epoch,
        }
        if val_loss is not None:
            checkpoint["val_loss"] = val_loss
        
        torch.save(
                checkpoint, 
                f"{self.args.output_dir}/checkpoint_epoch_{epoch}.pt"
        )

    def train_model(self, dataset, data_collator, eval_dataset):

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
        best_val_loss = float('inf')
        for epoch in range(self.args.epochs):
            epoch_loss = 0.0
            window_loss, window_count = 0.0, 0
            window_correct, window_total = 0.0, 0

            for step, batch in enumerate(data_loader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = self.model(input_ids=input_ids, attn_mask=attention_mask, labels=labels)
                raw_loss = outputs['loss']
                logits = outputs["logits"]
                correct, total = token_accuracy(logits, labels)
                window_correct += correct
                window_total += total
                loss = raw_loss / self.args.gradient_accumulation_steps
                loss.backward()
                epoch_loss += raw_loss.item()
                window_loss += raw_loss.item()
                window_count += 1

                is_accum_boundary = (step + 1) % self.args.gradient_accumulation_steps == 0
                is_last_microbatch = (step + 1) == num_microbatches

                if is_accum_boundary or is_last_microbatch:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    step_loss = window_loss / max(window_count, 1)
                    step_acc = window_correct / max(window_total, 1)

                    wandb.log({"train/step_loss": step_loss, "step_acc": step_acc}) if self.args.report_to_wandb else print(f"loss per step: {step_loss}, Step accuracy: {step_acc}")
                    window_loss, window_count = 0.0, 0
                    window_correct, window_total = 0, 0
            avg_epoch_loss = epoch_loss / num_microbatches

            if self.args.report_to_wandb:
                wandb.log({
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    "train/gpu_mem_gb": torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
                    "train/global_step": global_step,
                    "train/epoch_loss": avg_epoch_loss,
                    "epoch": epoch
                })

            if eval_dataset is not None:
                eval_metrics = evaluate_model(self.model, eval_dataset, data_collator, self.args.batch_size, device)
                self.model.train()
                is_best = eval_metrics["loss"] < best_val_loss
                if is_best: 
                    best_val_loss = eval_metrics["loss"]
                    self._save_checkpoint(optimizer, scheduler, global_step, epoch, val_loss=best_val_loss)
                if self.args.report_to_wandb:
                    wandb.log({"eval/avg_loss": eval_metrics["loss"], "eval/perplexity": eval_metrics["perplexity"], "eval/token_acc": eval_metrics["token_accuracy"]})
                if early_stop.early_stopping(eval_metrics["loss"]):
                    if self.args.report_to_wandb:
                        wandb.run.summary["early_stopping_epoch"] = epoch  
                        wandb.finish()                  
                    break
        save_pretrained(self.args.output_dir, self.model.state_dict(), self.model.config)