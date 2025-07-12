import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup
)
from evaluate import evaluate_model
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from accelerate import Accelerator

output_dir = "./finetuned_qwen"
os.makedirs(output_dir, exist_ok=True)

class SFT:
    def __init__(self, model: str, pad_token: int):
        self.model_name = model
        self.pad_token = pad_token
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.accelerator = Accelerator(mixed_precision="bf16")
        self.model.gradient_checkpointing_enable()
        self.model.config.use_cache = False
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"EOS token: {self.tokenizer.eos_token}")
        print(f"EOS token ID: {self.tokenizer.eos_token_id}")
        print(f"PAD token: {self.tokenizer.pad_token}")
        print(f"PAD token ID: {self.tokenizer.pad_token_id}")

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

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate,weight_decay=0.005) #base learning rate

        total_steps = epochs * len(data_loader) // gradient_accumulation_steps
        warmup_steps = int(0.1 * total_steps)  # 10% warmup

        # short warmup phase + cosine decay
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        self.model, optimizer, data_loader, scheduler = self.accelerator.prepare(
            self.model, optimizer, data_loader, scheduler
        )

        # Print scheduler info
        print(f"Total training steps: {total_steps}")
        print(f"Warmup steps: {warmup_steps}")
        print(f"Base learning rate: {learning_rate}")
        print(f"[Before training] LR = {scheduler.get_last_lr()[0]:.8f}")

        global_step = 0

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

                    if self.accelerator.is_main_process:
                        # phase info
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
            self.accelerator.print(f"Epoch {epoch+1} average loss: {avg_epoch_loss:.4f}")
            torch.cuda.empty_cache()

            # Evaluation
            if eval_dataset is not None:
                self.accelerator.wait_for_everyone()
                evaluate_model(self.model, eval_dataset, data_collator, batch_size, self.accelerator)
                self.model.train()

        # Save model after training
        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        if self.accelerator.is_main_process:
            unwrapped_model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)

        print("\n" + "=" * 50)
        print("MODEL TRAINING DONE!")
        print("\n" + "=" * 50)
        print(f"Model saved to {output_dir}")