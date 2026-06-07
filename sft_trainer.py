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
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

class SFT:
    def __init__(self, model: str, pad_token: int, args: SFTConfig, rank: int = 0):
        self.args = args
        self.model_name = model
        self.pad_token = pad_token
        self.rank = rank
        set_seed(self.args.seed)
        config = Config.config_from_model(self.model_name)
        if self.model_name == "Qwen2.5-0.5B":
            hf_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
            hf_model_state_dict = hf_model.state_dict()
            del hf_model
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
            self.model = Qwen2Model(config=config, args=self.args)
            load_weights_into_qwen(self.model, config, hf_model_state_dict)
        elif self.model_name == "Llama-3.2-1B":
            hf_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
            hf_model_state_dict = hf_model.state_dict()
            del hf_model
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
            self.model = LlamaModel(config=config, args=self.args)
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
            MAXLEN=256
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

            full_convo_ids = full_convo_ids[:MAXLEN]
            completion_mask = [0] * len(user_prompt_ids) + [1] * (len(full_convo_ids) - len(user_prompt_ids))
            completion_mask = completion_mask[:MAXLEN]
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
        model = self.model.module if hasattr(self.model, "module") else self.model
        checkpoint = {
            "model_state_dict": model.state_dict(),
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
    def train_model(
        self,
        dataset,
        data_collator,
        eval_dataset,
        use_distributed,
        rank=0,
        world_size=1,
    ):
        is_main = (rank == 0)
        strategy = self.args.distributed_strategy
        device = (
            torch.device(f"cuda:{rank}")
            if strategy in ["ddp", "fsdp"]
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
    
        self.model.to(device)
        self.model.train()
    
        if strategy == "ddp":
            self.model = DDP(self.model, device_ids=[rank])
        if strategy != "single":
            sampler = DistributedSampler(dataset)
            data_loader = DataLoader(
                dataset,
                batch_size=self.args.batch_size,
                shuffle=False,
                collate_fn=data_collator,
                sampler=sampler,
            )
        else:
            data_loader = DataLoader(
                dataset,
                batch_size=self.args.batch_size,
                shuffle=True,
                collate_fn=data_collator,
            )
    
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
    
        steps_per_epoch = math.ceil(
            len(data_loader) / self.args.gradient_accumulation_steps
        )
        total_steps = steps_per_epoch * self.args.epochs
        warmup_steps = int(0.1 * total_steps)
    
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
    
        early_stop = EarlyStopping()
        global_step = 0
        num_microbatches = len(data_loader)
        best_val_loss = float("inf")
    
        for epoch in range(self.args.epochs):
    
            # set_epoch so each epoch gets different shuffling across GPUs
            if strategy != "single":
                data_loader.sampler.set_epoch(epoch)
    
            epoch_loss = 0.0
            window_loss, window_count = 0.0, 0
            window_correct, window_total = 0.0, 0

            optimizer.zero_grad()
    
            for step, batch in enumerate(data_loader):
    
                input_ids = batch["input_ids"].to(device)
                print(f"input_ids shape: {input_ids.shape}")

                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
    
                outputs = self.model(
                    input_ids=input_ids,
                    attn_mask=attention_mask,
                    labels=labels,
                )
    
                raw_loss = outputs["loss"]
                logits = outputs["logits"]
    
                correct, total = token_accuracy(logits, labels)
                window_correct += correct
                window_total += total
    
                loss = raw_loss / self.args.gradient_accumulation_steps
                # loss.backward()
    
                epoch_loss += raw_loss.item()
                window_loss += raw_loss.item()
                window_count += 1
    
                is_accum_boundary = (
                    (step + 1) % self.args.gradient_accumulation_steps == 0
                )
                is_last_microbatch = (step + 1) == num_microbatches

                should_sync = is_accum_boundary or is_last_microbatch
                
                if strategy in ["ddp", "fsdp"] and not should_sync:
                    with self.model.no_sync():
                        loss.backward()
                else:
                    loss.backward()
                    
                if should_sync:
    
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.args.max_grad_norm,
                    )
    
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
    
                    global_step += 1

                    # FIX 3: reduce metrics across all ranks before logging
                    if strategy in ["ddp", "fsdp"]:
                        correct_t = torch.tensor(window_correct, dtype=torch.float32, device=device)
                        total_t   = torch.tensor(window_total,   dtype=torch.float32, device=device)
                        loss_t    = torch.tensor(window_loss,    dtype=torch.float32, device=device)
                        dist.all_reduce(correct_t)
                        dist.all_reduce(total_t)
                        dist.all_reduce(loss_t)
                        step_loss = (loss_t / max(window_count * world_size, 1)).item()
                        step_acc  = (correct_t / total_t.clamp(min=1)).item()
                    else:
                        step_loss = window_loss / max(window_count, 1)
                        step_acc  = window_correct / max(window_total, 1)
                    # step_loss = window_loss / max(window_count, 1)
                    # step_acc = window_correct / max(window_total, 1)
    
                    if is_main:
                        if self.args.report_to_wandb:
                            wandb.log(
                                {
                                    "train/step_loss": step_loss,
                                    "train/step_acc": step_acc,
                                }
                            )
                        else:
                            print(
                                f"[Rank {rank}] Step {global_step} | "
                                f"loss: {step_loss:.4f} | acc: {step_acc:.4f}"
                            )
    
                    window_loss, window_count = 0.0, 0
                    window_correct, window_total = 0, 0
    
            # FIX 4: reduce epoch loss across ranks for accurate logging
            if strategy in ["ddp", "fsdp"]:
                epoch_loss_t = torch.tensor(epoch_loss, dtype=torch.float32, device=device)
                dist.all_reduce(epoch_loss_t)
                avg_epoch_loss = (epoch_loss_t / (num_microbatches * world_size)).item()
            else:
                avg_epoch_loss = epoch_loss / num_microbatches
    
            if is_main and self.args.report_to_wandb:
                wandb.log(
                    {
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/gpu_mem_gb": (
                            torch.cuda.memory_allocated() / 1e9
                            if torch.cuda.is_available()
                            else 0
                        ),
                        "train/global_step": global_step,
                        "train/epoch_loss": avg_epoch_loss,
                        "epoch": epoch,
                    }
                )
            if strategy in ["ddp", "fsdp"]:
                dist.barrier()
            should_stop = False
    
            # eval, checkpointing, early stopping — rank 0 only
            if eval_dataset is not None and is_main:
    
                eval_model = (
                    self.model.module
                    if hasattr(self.model, "module")
                    else self.model
                )
    
                eval_metrics = evaluate_model(
                    eval_model,
                    eval_dataset,
                    data_collator,
                    self.args.batch_size,
                    device,
                )
    
                self.model.train()
    
                is_best = eval_metrics["loss"] < best_val_loss
    
                if is_best:
                    best_val_loss = eval_metrics["loss"]
    
                    self._save_checkpoint(
                        optimizer,
                        scheduler,
                        global_step,
                        epoch,
                        val_loss=best_val_loss,
                    )
    
                if self.args.report_to_wandb:
                    wandb.log(
                        {
                            "eval/avg_loss": eval_metrics["loss"],
                            "eval/perplexity": eval_metrics["perplexity"],
                            "eval/token_acc": eval_metrics["token_accuracy"],
                        }
                    )
    
                should_stop = early_stop.early_stopping(eval_metrics["loss"])    
                if should_stop and self.args.report_to_wandb:
                    wandb.run.summary["early_stopping_epoch"] = epoch
                    wandb.finish()
            if strategy in ["ddp", "fsdp"]:
                stop_flag = torch.tensor([int(should_stop)], device=device)
                dist.broadcast(stop_flag, src=0)
                should_stop = bool(stop_flag.item())

            # FIX 1 (continued): second barrier so everyone waits for rank 0
            # to finish eval + checkpoint before starting the next epoch
            if strategy in ["ddp", "fsdp"]:
                dist.barrier()

            if should_stop:
                break

    
        if is_main:
            model_to_save = (
                self.model.module
                if hasattr(self.model, "module")
                else self.model
            )
    
            save_pretrained(
                self.args.output_dir,
                model_to_save.state_dict(),
                model_to_save.config,
            )