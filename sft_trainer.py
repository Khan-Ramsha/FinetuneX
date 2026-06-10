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
import torch.distributed.checkpoint as dcp
from finetunex.distributed.distributed_checkpoint import AppState
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
from torch.distributed._composable.fsdp import fully_shard
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
            MAXLEN = 512
            messages = example["messages"]
            assistant_idx = next((i for i, m in enumerate(messages) if m["role"] == "assistant"), None)
            if assistant_idx is None:
                raise ValueError("No assistant message found!")

            user_prompt = messages[:assistant_idx]
            full_convo  = messages[:assistant_idx + 1]

            user_prompt_ids = _apply_chat_template_ids(
                tokenizer, user_prompt, tokenize=True,
                add_generation_prompt=True, return_tensors=None,
            )
            full_convo_ids = _apply_chat_template_ids(
                tokenizer, full_convo, tokenize=True,
                add_generation_prompt=False, return_tensors=None,
            )

            if full_convo_ids[-1] != self.tokenizer.eos_token_id:
                full_convo_ids.append(self.tokenizer.eos_token_id)

            if len(full_convo_ids) > MAXLEN:
                return {"input_ids": [], "completion_mask": []}

            completion_mask = [0] * len(user_prompt_ids) + [1] * (len(full_convo_ids) - len(user_prompt_ids))

            if sum(completion_mask) == 0:
                raise ValueError("No completion tokens to learn from!")

            assert len(full_convo_ids) == len(completion_mask), "Length Mismatch!"
            return {
                "input_ids": full_convo_ids,
                "completion_mask": completion_mask
            }

        dataset = dataset.map(tokenize, fn_kwargs={"tokenizer": self.tokenizer})
        dataset = dataset.filter(lambda x: len(x["input_ids"]) > 0)
        return dataset

    def _save_checkpoint(self, optimizer, scheduler, global_step, epoch, val_loss):
        strategy = self.args.distributed_strategy
        ckpt_dir = f"{self.args.output_dir}/checkpoint_epoch_{epoch}"

        if strategy == "fsdp":
            dcp.save(
                {"app": AppState(self.model, optimizer)},
                checkpoint_id=ckpt_dir
            )
            if self.rank == 0:
                torch.save({
                    "scheduler_state_dict": scheduler.state_dict(),
                    "global_step": global_step,
                    "epoch": epoch,
                    "val_loss": val_loss,
                }, f"{ckpt_dir}/training_state.pt")
        else:
            os.makedirs(ckpt_dir, exist_ok=True)
            model_to_save = self.model.module if hasattr(self.model, "module") else self.model
            if self.rank == 0:
                torch.save({
                    "model_state_dict": model_to_save.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "global_step": global_step,
                    "epoch": epoch,
                    "val_loss": val_loss,
                }, f"{ckpt_dir}/checkpoint.pt")

    def train_model(
        self,
        dataset,
        data_collator,
        eval_dataset,
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
        if strategy == "ddp":
            self.model = DDP(self.model, device_ids=[rank])
        elif strategy == "fsdp":
            for layer in self.model.layers:
                fully_shard(layer)
            fully_shard(self.model)

        self.model.train()

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

            if strategy != "single":
                data_loader.sampler.set_epoch(epoch)

            epoch_loss = 0.0
            grad_norm = 0.0
            optimizer.zero_grad()

            for step, batch in enumerate(data_loader):

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = self.model(
                    input_ids=input_ids,
                    attn_mask=attention_mask,
                    labels=labels,
                )

                raw_loss = outputs["loss"]
                loss = raw_loss / self.args.gradient_accumulation_steps
                epoch_loss += raw_loss.item()

                is_accum_boundary = (
                    (step + 1) % self.args.gradient_accumulation_steps == 0
                )
                is_last_microbatch = (step + 1) == num_microbatches
                should_sync = is_accum_boundary or is_last_microbatch

                if strategy == "ddp" and not should_sync:
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

            # reduce epoch loss across ranks
            if strategy in ["ddp", "fsdp"]:
                epoch_loss_t = torch.tensor(epoch_loss, dtype=torch.float32, device=device)
                dist.all_reduce(epoch_loss_t)
                avg_epoch_loss = (epoch_loss_t / (num_microbatches * world_size)).item()
            else:
                avg_epoch_loss = epoch_loss / num_microbatches

            if is_main:
                print(
                    f"Epoch {epoch+1}/{self.args.epochs} | "
                    f"train_loss: {avg_epoch_loss:.4f} | "
                )
                if self.args.report_to_wandb:
                    wandb.log({
                        "train/epoch_loss": avg_epoch_loss,
                        "train/gpu_mem_gb": (
                            torch.cuda.memory_allocated() / 1e9
                            if torch.cuda.is_available() else 0
                        ),
                        "train/global_step": global_step,
                        "epoch": epoch,
                    })

            if strategy in ["ddp", "fsdp"]:
                dist.barrier()

            should_stop = False
            is_best = False

            if eval_dataset is not None:

                if strategy == "fsdp":
                    # all ranks run eval on the sharded model
                    eval_metrics = evaluate_model(
                        self.model,
                        eval_dataset,
                        data_collator,
                        self.args.batch_size,
                        device,
                    )
                    # evaluate_model runs on full dataset on every rank — no all_reduce needed

                elif is_main:
                    eval_model = self.model.module if hasattr(self.model, "module") else self.model
                    eval_metrics = evaluate_model(
                        eval_model,
                        eval_dataset,
                        data_collator,
                        self.args.batch_size,
                        device,
                    )

                self.model.train()

                if is_main:
                    is_best = eval_metrics["loss"] < best_val_loss
                    if is_best:
                        best_val_loss = eval_metrics["loss"]
                    print(
                        f"  val_loss: {eval_metrics['loss']:.4f} | "
                        f"val_ppl: {eval_metrics['perplexity']:.2f} | "
                        f"val_acc: {eval_metrics['token_accuracy']:.4f}"
                    )
                    if self.args.report_to_wandb:
                        wandb.log({
                            "eval/avg_loss": eval_metrics["loss"],
                            "eval/perplexity": eval_metrics["perplexity"],
                            "eval/token_acc": eval_metrics["token_accuracy"],
                        })
                    should_stop = early_stop.early_stopping(eval_metrics["loss"])
                    if should_stop and self.args.report_to_wandb:
                        wandb.run.summary["early_stopping_epoch"] = epoch
                        wandb.finish()

            # broadcast is_best and should_stop so all ranks agree
            if strategy in ["ddp", "fsdp"]:
                is_best_t = torch.tensor([int(is_best)], device=device)
                dist.broadcast(is_best_t, src=0)
                is_best = bool(is_best_t.item())

                stop_flag = torch.tensor([int(should_stop)], device=device)
                dist.broadcast(stop_flag, src=0)
                should_stop = bool(stop_flag.item())

            if is_best:
                self._save_checkpoint(optimizer, scheduler, global_step, epoch, val_loss=best_val_loss)

            if strategy in ["ddp", "fsdp"]:
                dist.barrier()

            if should_stop:
                break

        # save final model weights after training loop ends
        if strategy == "fsdp":
            from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
            # collective — all ranks must call this
            model_state = get_model_state_dict(
                self.model,
                options=StateDictOptions(full_state_dict=True, cpu_offload=True)
            )
            if self.rank == 0:
                save_pretrained(self.args.output_dir, model_state, self.model.config)
        else:
            if self.rank == 0:
                model_to_save = self.model.module if hasattr(self.model, "module") else self.model
                save_pretrained(self.args.output_dir, model_to_save.state_dict(), model_to_save.config)