# Reference:   https://arxiv.org/abs/2305.18290
#
# Design principles
# -----------------
# 1. Mirrors SFT's class layout: __init__ / train / _save_checkpoint /
#    save_pretrained so it integrates with the rest of FinetuneX.
# 2. Reference model is frozen at construction time.  Two strategies:
#      a. ref_model_path=None  → deep-copy the policy model, freeze it.
#      b. ref_model_path=<dir> → load from a saved FinetuneX checkpoint,
#         freeze it.  Lets you use a *different* base model as reference
#         (e.g. the SFT'd checkpoint).
# 3. Log-probability computation is masked to response tokens only
#    (chosen_mask / rejected_mask), matching SFT's completion_only_loss.
# 4. Supports both 'sigmoid' (original DPO) and 'ipo' loss variants.
# 5. Cosine LR schedule with 10 % warmup – identical to SFT.
# 6. Gradient accumulation + gradient clipping – identical to SFT.
# 7. Optional W&B logging.
# ---------------------------------------------------------------------------

import copy
import math
import os

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from dpo_config import DPOConfig
from utils import save_pretrained, set_seed
try:
    import wandb

    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from finetunex.base.config import Config
from finetunex.models.qwen2.model import Qwen2Model
from finetunex.models.qwen2.save_load import load_weights_into_qwen
from finetunex.models.llama.model import LlamaModel
from finetunex.models.llama.save_load import load_weights_into_llama

def _build_model_and_tokenizer(model_name: str, args):
    """
    Instantiate a FinetuneX model + tokenizer from HuggingFace weights,
    exactly as SFT.__init__ does.  Returns (model, tokenizer).
    """
    config = Config.config_from_model(model_name)

    if model_name == "Qwen2.5-0.5B":
        hf = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
        sd = hf.state_dict()
        del hf
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
        model = Qwen2Model(config=config, args=args)
        load_weights_into_qwen(model, config, sd)

    elif model_name == "Llama-3.2-1B":
        hf = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
        sd = hf.state_dict()
        del hf
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        model = LlamaModel(config=config, args=args)
        load_weights_into_llama(model, config, sd)

    else:
        raise ValueError(
            f"Unknown model_name '{model_name}'. "
            "Expected 'Qwen2.5-0.5B' or 'Llama-3.2-1B'."
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def _load_reference_from_path(ref_model_path: str, model_name: str, args):
    """
    Load a FinetuneX checkpoint saved by utils.save_pretrained.
    The checkpoint directory must contain model.bin (or checkpoint.pt)
    and config.json.
    """
    from utils import from_pretrained  # uses the existing FinetuneX helper

    ref_model = from_pretrained(ref_model_path)
    return ref_model


def _freeze(model: torch.nn.Module) -> torch.nn.Module:
    """Freeze all parameters in-place and switch to eval mode."""
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model

def _response_log_probs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Return the sum of log-probabilities over response tokens for each
    sequence in the batch.

    We use the standard auto-regressive cross-entropy trick:
        logits[:, :-1, :]  predicts  input_ids[:, 1:]
    so response_mask is also shifted left by one position.

    The sum (not mean) over response tokens is used because DPO's derivation
    works with log p(y|x) = Σ log p(t_i | t_{<i}, x), i.e. the joint
    log-probability of the full response.
    """
    outputs = model(input_ids=input_ids, attn_mask=attention_mask)
    logits = outputs["logits"]

    shift_logits = logits[:, :-1, :]  # [B, L-1, V]
    shift_labels = input_ids[:, 1:]  # [B, L-1]
    shift_mask = response_mask[:, 1:]  # [B, L-1]  shifted response mask

    # Token-level log-probabilities via log-softmax
    log_probs = F.log_softmax(shift_logits, dim=-1)  # [B, L-1, V]

    # Gather the log-prob of the actual next token
    token_log_probs = log_probs.gather(
        dim=-1,
        index=shift_labels.unsqueeze(-1),
    ).squeeze(
        -1
    )  # [B, L-1]

    masked_log_probs = token_log_probs * shift_mask.float()  # [B, L-1]
    return masked_log_probs.sum(dim=-1)  # [B]

def _dpo_loss(
    policy_chosen_logps: torch.Tensor,  # [B]
    policy_rejected_logps: torch.Tensor,  # [B]
    ref_chosen_logps: torch.Tensor,  # [B]
    ref_rejected_logps: torch.Tensor,  # [B]
    beta: float,
    label_smoothing: float,
    loss_type: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the DPO (or IPO) loss and return
        (loss, chosen_rewards, rejected_rewards).

    Chosen / rejected rewards are the scaled log-ratio differences —
    useful for monitoring training progress.
    """
    # Log-ratios  log π(y|x) / π_ref(y|x)
    chosen_log_ratios = policy_chosen_logps - ref_chosen_logps  # [B]
    rejected_log_ratios = policy_rejected_logps - ref_rejected_logps  # [B]

    if loss_type == "sigmoid":
        # Equation (7) from Rafailov et al. 2023
        logits = beta * (chosen_log_ratios - rejected_log_ratios)  # [B]

        if label_smoothing > 0:
            # Conservative DPO (Mitchell et al. 2023 / TRL implementation)
            loss = (
                -(1.0 - label_smoothing) * F.logsigmoid(logits)
                - label_smoothing * F.logsigmoid(-logits)
            ).mean()
        else:
            loss = -F.logsigmoid(logits).mean()

    elif loss_type == "ipo":
        # IPO: (log π/π_ref difference − 1/(2β))²
        logits = chosen_log_ratios - rejected_log_ratios  # [B]
        loss = ((logits - 1.0 / (2.0 * beta)) ** 2).mean()

    else:
        raise ValueError(
            f"Unknown loss_type='{loss_type}'. Expected 'sigmoid' or 'ipo'."
        )

    # Rewards for logging (detached – not used in backward pass)
    chosen_rewards = (beta * chosen_log_ratios).detach()
    rejected_rewards = (beta * rejected_log_ratios).detach()

    return loss, chosen_rewards, rejected_rewards

class DPOTrainer:
    """
    Direct Preference Optimisation trainer for FinetuneX.

    Usage
    -----
    >>> from dpo_config import DPOConfig
    >>> from dpo_dataset import PreferenceDPODataset
    >>> from dpo_data_collator import DataCollatorForDPO
    >>>
    >>> args = DPOConfig(model_name="Qwen2.5-0.5B", beta=0.1, epochs=3)
    >>> trainer = DPOTrainer(args)
    >>>
    >>> train_ds = PreferenceDPODataset(train_data, trainer.tokenizer)
    >>> eval_ds  = PreferenceDPODataset(eval_data,  trainer.tokenizer)
    >>> collator = DataCollatorForDPO(pad_token_id=trainer.tokenizer.pad_token_id)
    >>>
    >>> trainer.train(train_ds, eval_ds, collator)
    Example
    from datasets import load_dataset
    from dpo_config import DPOConfig
    from dpo_trainer import DPOTrainer
    from dpo_data_preprocessing import PreferenceDPODataset
    from dpo_data_collator import DataCollatorForDPO
    import pandas as pd
    from datasets import load_dataset
    training_args = DPOConfig(model_name="Qwen2.5-0.5B", beta=0.1, epochs=3)

    trainer = DPOTrainer(training_args)

    df=load_dataset("json", data_files="./data/dpo_pairs_base.jsonl")
    print(df)
    split = df["train"].train_test_split(test_size=0.2,seed=42)
    train_ds = PreferenceDPODataset(split["train"], trainer.tokenizer)
    eval_ds  = PreferenceDPODataset(split["test"],  trainer.tokenizer)
    collator = DataCollatorForDPO(pad_token_id=trainer.tokenizer.pad_token_id)
    trainer.train(train_ds, eval_ds, collator)
    """

    def __init__(self, args: DPOConfig):
        self.args = args
        set_seed(args.seed)

        # Load policy model
        print(f"[DPOTrainer] Loading policy model: {args.model_name}")
        self.model, self.tokenizer = _build_model_and_tokenizer(args.model_name, args)

        #  Build reference model
        if args.ref_model_path is None:
            # Strategy A: deep-copy the freshly loaded policy weights.
            # This is the most common setup – the reference is the base model
            # (or an SFT'd model) before any DPO updates.
            print(
                "[DPOTrainer] ref_model_path=None "
                "deep-copying policy as reference model."
            )
            self.ref_model = copy.deepcopy(self.model)
        else:
            # Strategy B: load from a previously saved FinetuneX checkpoint.
            print(f"[DPOTrainer] Loading reference model from: {args.ref_model_path}")
            self.ref_model = _load_reference_from_path(
                args.ref_model_path, args.model_name, args
            )

        _freeze(self.ref_model)
        print("[DPOTrainer] Reference model frozen")

    def _save_checkpoint(self, optimizer, scheduler, global_step, epoch, metrics=None):
        """Save a resumable training checkpoint (policy only)."""
        os.makedirs(self.args.output_dir, exist_ok=True)
        ckpt = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "global_step": global_step,
            "epoch": epoch,
        }
        if metrics:
            ckpt.update(metrics)
        path = os.path.join(self.args.output_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save(ckpt, path)
        print(f"[DPOTrainer] Checkpoint saved → {path}")

    # Training

    def train(
        self,
        train_dataset,
        eval_dataset=None,
        collator=None,
    ):
        """
        Run DPO training.

        Parameters
        ----------
        train_dataset : PreferenceDPODataset
        eval_dataset  : PreferenceDPODataset or None
        collator      : DataCollatorForDPO  (if None, one is created
                        automatically from tokenizer.pad_token_id)
        """
        from dpo_data_collator import DataCollatorForDPO

        if collator is None:
            collator = DataCollatorForDPO(pad_token_id=self.tokenizer.pad_token_id)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.ref_model.to(device)

        self.model.train()
        # ref_model stays in eval() from _freeze()

        loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            collate_fn=collator,
        )
        train_dataset.attach_ref_logps(self.ref_model, collator, self.args.batch_size, device)
        if eval_dataset is not None:
            eval_dataset.attach_ref_logps(self.ref_model, collator, self.args.batch_size, device)

        self.ref_model.to("cpu")
        del self.ref_model
        torch.cuda.empty_cache()
        self.ref_model = None  # mark as precomputed mode
        # Optimiser & LR schedule
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )

        steps_per_epoch = math.ceil(len(loader) / self.args.gradient_accumulation_steps)
        total_steps = steps_per_epoch * self.args.epochs
        warmup_steps = max(1, int(0.1 * total_steps))  # 10 % warmup

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        if self.args.report_to_wandb:
            if not _WANDB_AVAILABLE:
                raise ImportError(
                    "report_to_wandb=True but wandb is not installed. "
                    "Run: pip install wandb"
                )
            wandb.init(
                project="finetunex-dpo",
                config=vars(self.args),
                name=f"{self.args.model_name}-dpo-{self.args.epochs}ep",
            )

        global_step = 0
        best_eval_acc = -1.0  # reward accuracy on eval set
        num_microbatches = len(loader)

        for epoch in range(self.args.epochs):
            self.model.train()

            # Running accumulators for one gradient-accumulation window
            window_loss = 0.0
            window_chosen_r = 0.0
            window_rejected_r = 0.0
            window_count = 0

            epoch_loss = 0.0

            pbar = tqdm(
                enumerate(loader),
                total=num_microbatches,
                desc=f"Epoch {epoch + 1}/{self.args.epochs}",
            )

            for step, batch in pbar:
                chosen = batch["chosen"].to(device)  # [B, L]
                rejected = batch["rejected"].to(device)  # [B, L]
                chosen_mask = batch["chosen_mask"].to(device)  # [B, L]
                rejected_mask = batch["rejected_mask"].to(device)  # [B, L]
                chosen_attn_mask = batch["chosen_attention_mask"].to(device)
                rejected_attn_mask = batch["rejected_attention_mask"].to(device)

                policy_chosen_logps = _response_log_probs(
                    self.model, chosen, chosen_attn_mask, chosen_mask
                )
                policy_rejected_logps = _response_log_probs(
                    self.model, rejected, rejected_attn_mask, rejected_mask
                )

                if self.ref_model is None:
                    ref_chosen_logps = batch["ref_chosen_logps"].to(device)
                    ref_rejected_logps = batch["ref_rejected_logps"].to(device)
                else:
                    with torch.no_grad():
                        ref_chosen_logps = _response_log_probs(
                            self.ref_model, chosen, chosen_attn_mask, chosen_mask
                        )
                        ref_rejected_logps = _response_log_probs(
                            self.ref_model, rejected, rejected_attn_mask, rejected_mask
                        )
                # DPO loss
                loss, chosen_rewards, rejected_rewards = _dpo_loss(
                    policy_chosen_logps,
                    policy_rejected_logps,
                    ref_chosen_logps,
                    ref_rejected_logps,
                    beta=self.args.beta,
                    label_smoothing=self.args.label_smoothing,
                    loss_type=self.args.loss_type,
                )

                scaled_loss = loss / self.args.gradient_accumulation_steps
                scaled_loss.backward()

                # Accumulate window metrics
                raw_loss = loss.item()
                epoch_loss += raw_loss
                window_loss += raw_loss
                window_chosen_r += chosen_rewards.mean().item()
                window_rejected_r += rejected_rewards.mean().item()
                window_count += 1

                is_update = (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    step + 1
                ) == num_microbatches

                if is_update:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.args.max_grad_norm
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    avg_loss = window_loss / window_count
                    avg_chosen_r = window_chosen_r / window_count
                    avg_rejected_r = window_rejected_r / window_count
                    # Reward accuracy: fraction of batch where chosen > rejected
                    # (computed over this window's final micro-batch tensors)
                    reward_acc = (
                        (chosen_rewards > rejected_rewards).float().mean().item()
                    )
                    reward_margin = avg_chosen_r - avg_rejected_r

                    step_metrics = {
                        "train/step_loss": avg_loss,
                        "train/chosen_reward": avg_chosen_r,
                        "train/rejected_reward": avg_rejected_r,
                        "train/reward_accuracy": reward_acc,
                        "train/reward_margin": reward_margin,
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/grad_norm": (
                            grad_norm.item()
                            if isinstance(grad_norm, torch.Tensor)
                            else grad_norm
                        ),
                        "train/global_step": global_step,
                    }

                    if self.args.report_to_wandb:
                        wandb.log(step_metrics)
                    else:
                        pbar.set_postfix(
                            loss=f"{avg_loss:.4f}",
                            r_acc=f"{reward_acc:.3f}",
                            margin=f"{reward_margin:.4f}",
                        )

                    window_loss = window_chosen_r = window_rejected_r = 0.0
                    window_count = 0

            avg_epoch_loss = epoch_loss / num_microbatches
            epoch_metrics = {
                "train/epoch_loss": avg_epoch_loss,
                "epoch": epoch,
                "train/gpu_mem_gb": (
                    torch.cuda.memory_allocated() / 1e9
                    if torch.cuda.is_available()
                    else 0.0
                ),
            }
            if self.args.report_to_wandb:
                wandb.log(epoch_metrics)
            else:
                print(f"\n[Epoch {epoch + 1}] avg_loss={avg_epoch_loss:.4f}")

            if eval_dataset is not None:
                eval_metrics = self.evaluate(eval_dataset, collator, device)
                self.model.train()

                is_best = eval_metrics["reward_accuracy"] > best_eval_acc
                if is_best:
                    best_eval_acc = eval_metrics["reward_accuracy"]
                    self._save_checkpoint(
                        optimizer,
                        scheduler,
                        global_step,
                        epoch,
                        metrics={"best_eval_reward_acc": best_eval_acc},
                    )

                log_eval = {f"eval/{k}": v for k, v in eval_metrics.items()}
                log_eval["epoch"] = epoch
                if self.args.report_to_wandb:
                    wandb.log(log_eval)
                else:
                    print(
                        f"[Eval   {epoch + 1}] "
                        + " | ".join(f"{k}={v:.4f}" for k, v in eval_metrics.items())
                    )
            else:
                # No eval set: save after every epoch
                self._save_checkpoint(optimizer, scheduler, global_step, epoch)

        save_pretrained(
            self.args.output_dir,
            self.model.state_dict(),
            self.model.config,
        )
        print(
            f"[DPOTrainer] Training complete. " f"Model saved to {self.args.output_dir}"
        )

        if self.args.report_to_wandb:
            wandb.finish()

    @torch.no_grad()
    def evaluate(self, eval_dataset, collator, device) -> dict:
        """
        Compute DPO-specific evaluation metrics on *eval_dataset*.

        Returns a dict with:
            loss - average DPO loss
            reward_accuracy  - fraction of pairs where chosen reward > rejected
            reward_margin - mean(chosen_reward - rejected_reward)
            chosen_reward - mean chosen reward
            rejected_reward - mean rejected reward
        """
        self.model.eval()
        loader = DataLoader(
            eval_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            collate_fn=collator,
        )

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        chosen_rewards_all = []
        rejected_rewards_all = []

        for batch in loader:
            chosen = batch["chosen"].to(device)
            rejected = batch["rejected"].to(device)
            chosen_mask = batch["chosen_mask"].to(device)
            rejected_mask = batch["rejected_mask"].to(device)
            chosen_attn_mask = batch["chosen_attention_mask"].to(device)
            rejected_attn_mask = batch["rejected_attention_mask"].to(device)
            policy_chosen_logps = _response_log_probs(
                self.model, chosen, chosen_attn_mask, chosen_mask
            )
            policy_rejected_logps = _response_log_probs(
                self.model, rejected, rejected_attn_mask, rejected_mask
            )
           
            if self.ref_model is None:
                ref_chosen_logps = batch["ref_chosen_logps"].to(device)
                ref_rejected_logps = batch["ref_rejected_logps"].to(device)
            else:
                ref_chosen_logps = _response_log_probs(
                    self.ref_model, chosen, chosen_attn_mask, chosen_mask
                )
                ref_rejected_logps = _response_log_probs(
                    self.ref_model, rejected, rejected_attn_mask, rejected_mask
                )
            loss, c_rewards, r_rewards = _dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                ref_chosen_logps,
                ref_rejected_logps,
                beta=self.args.beta,
                label_smoothing=self.args.label_smoothing,
                loss_type=self.args.loss_type,
            )

            B = chosen.size(0)
            total_loss += loss.item() * B
            total_correct += (c_rewards > r_rewards).sum().item()
            total_samples += B

            chosen_rewards_all.append(c_rewards)
            rejected_rewards_all.append(r_rewards)

        all_chosen = torch.cat(chosen_rewards_all)
        all_rejected = torch.cat(rejected_rewards_all)

        return {
            "loss": total_loss / max(total_samples, 1),
            "reward_accuracy": total_correct / max(total_samples, 1),
            "reward_margin": (all_chosen - all_rejected).mean().item(),
            "chosen_reward": all_chosen.mean().item(),
            "rejected_reward": all_rejected.mean().item(),
        }