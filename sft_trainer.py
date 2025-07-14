import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup
)
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from accelerate import Accelerator
from early_stopping import EarlyStopping
from evaluate import evaluate_model
import matplotlib.pyplot as plt
import numpy as np

output_dir = "./finetuned_qwen"
os.makedirs(output_dir, exist_ok=True)

class SFT:
    def __init__(self, model: str, pad_token: int):
        self.model_name = model
        self.pad_token = pad_token
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.accelerator = Accelerator(gradient_accumulation_steps=8,mixed_precision="bf16")
        self.model.gradient_checkpointing_enable()
        self.model.config.use_cache = False
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"EOS token: {self.tokenizer.eos_token}")
        print(f"EOS token ID: {self.tokenizer.eos_token_id}")
        print(f"PAD token: {self.tokenizer.pad_token}")
        print(f"PAD token ID: {self.tokenizer.pad_token_id}")

        # For tracking learning rate
        self.lr_history = []
        self.step_history = []
        self.training_loss = []
        self.validation_loss = []
    def prepare_dataset(self, dataset):
        def tokenize(example, tokenizer):
            messages = example["messages"]
            assistant_idx = next((i for i, m in enumerate(messages) if m["role"] == "assistant"), None)
            if assistant_idx is None:
                raise ValueError("No assistant message found!")

            user_prompt = messages[:assistant_idx]
            full_convo = messages[:assistant_idx + 1]

            user_prompt_ids = tokenizer.apply_chat_template(
                user_prompt,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors=None
            )
            full_convo_ids = tokenizer.apply_chat_template(
                full_convo,
                tokenize=True,
                add_generation_prompt=False,
                return_tensors=None
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

    def train_model(self, dataset, data_collator, batch_size, epochs, learning_rate, eval_dataset, gradient_accumulation_steps):
        self.model.train()
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate,weight_decay=0.001) #base learning rate

        total_batches = epochs * len(data_loader)
        total_steps = total_batches // gradient_accumulation_steps
        warmup_steps = int(0.1 * total_steps)  # 10% warmup

        # short warmup phase + cosine decay
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # Prepare with accelerator
        self.model, optimizer, data_loader, scheduler = self.accelerator.prepare(
            self.model, optimizer, data_loader, scheduler
        )

        # Print scheduler info
        print(f"Total training steps: {total_steps}")
        print(f"Warmup steps: {warmup_steps}")
        print(f"Base learning rate: {learning_rate}")
        print(f"[Before training] LR = {scheduler.get_last_lr()[0]:.8f}")

        global_step = 0
        early_stop = EarlyStopping()
        for epoch in range(epochs):
            if self.accelerator.is_main_process:
                progress = tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}")
            else:
                progress = data_loader

            total_loss = 0.0
            num_batches = 0

            for step, batch in enumerate(progress):
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"]

                with self.accelerator.accumulate(self.model):
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    global_step += 1
                    current_lr = scheduler.get_last_lr()[0]

                    self.lr_history.append(current_lr)
                    self.step_history.append(global_step)
                    
                    # phase info
                    if self.accelerator.is_main_process:
                        if global_step <= warmup_steps:
                            phase = "WARMUP"
                        else:
                            phase = "COSINE DECAY"
    
                        print(f"Epoch {epoch+1}, Step {global_step}, LR: {current_lr:.8f} [{phase}]")

                if step % 10 == 0:
                    torch.cuda.empty_cache()

                loss_for_logging = self.accelerator.gather(outputs.loss.detach()).mean()
                total_loss += loss_for_logging.item()
                num_batches += 1

                if self.accelerator.is_main_process:
                    progress.set_postfix(loss=loss_for_logging.item())

            avg_epoch_loss = total_loss / num_batches
            self.training_loss.append(avg_epoch_loss)
            self.accelerator.print(f"Epoch {epoch+1} average loss: {avg_epoch_loss:.4f}")
            torch.cuda.empty_cache()

            # Evaluation
            if eval_dataset is not None:
                self.accelerator.wait_for_everyone()
                loss = evaluate_model(self.model, eval_dataset, data_collator, batch_size, self.accelerator)
                self.validation_loss.append(loss)
                if early_stop.early_stopping(loss):
                    self.accelerator.print("Early stopping triggered!")
                    break
                self.model.train()

        # get plots ready
        if self.accelerator.is_main_process:
            self.plot_lr_schedule(warmup_steps, total_steps)
            self.plot_loss_curves()

        #Save model after training
        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        if self.accelerator.is_main_process:
            unwrapped_model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)

        print("\n" + "=" * 50)
        print("MODEL TRAINING DONE!")
        print("\n" + "=" * 50)
        print(f"Model saved to {output_dir}")
        
    def plot_loss_curves(self):
        import matplotlib.pyplot as plt
        epochs = range(1, len(self.training_loss) + 1)
        
        plt.figure(figsize=(12, 8))
        plt.plot(epochs, self.training_loss, 'b-', linewidth=2, marker='o', 
                label='Training Loss', markersize=6)        
        plt.plot(epochs, self.validation_loss, 'r-', 
                 linewidth=2, marker='s',label='Validation Loss', markersize=6)
        plt.title('Training Progress: Loss Over Time', fontsize=16, fontweight='bold')
        plt.xlabel('Epoch',fontsize=12)
        plt.ylabel('Loss',fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'loss_curves.png'), dpi=300, bbox_inches='tight')
        
    def plot_lr_schedule(self, warmup_steps, total_steps):
        plt.figure(figsize=(12, 6))
        plt.plot(self.step_history, self.lr_history, linewidth=2, color='green')
        plt.axvline(x=warmup_steps, color='red', linestyle='--', alpha=0.7, 
                   label=f'Warmup End (Step {warmup_steps})')
        
        plt.xlabel('Training Step')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule: Warmup + Cosine Decay', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add annotations
        plt.text(0.02, 0.98, 'Warmup Phase:\nGradually increase LR', 
                transform=plt.gca().transAxes, fontsize=10, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        plt.text(0.6, 0.98, 'Cosine Decay:\nGradually decrease LR', 
                transform=plt.gca().transAxes, fontsize=10, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'lr_schedule.png'), dpi=300, bbox_inches='tight')
    
        # Print some statistics
        max_lr = max(self.lr_history)
        min_lr = min(self.lr_history)
        warmup_end_lr = self.lr_history[warmup_steps-1] if warmup_steps > 0 else self.lr_history[0]
    
        print(f"\nLearning Rate Schedule Summary:")
        print(f"  Max LR: {max_lr:.8f}")
        print(f"  Min LR: {min_lr:.8f}")
        print(f"  LR at warmup end: {warmup_end_lr:.8f}")
        print(f"  Warmup steps: {warmup_steps}")
        print(f"  Total steps: {total_steps}")
        print(f"  Actual recorded steps: {len(self.lr_history)}")  