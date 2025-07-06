
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
from torch.optim import AdamW
from torch.utils.data import DataLoader
from evaluate import evaluate_model
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
        self.evaluate_model = evaluate_model
        self.accelerator = Accelerator(mixed_precision="fp16")
        self.model.gradient_checkpointing_enable()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"EOS token: {self.tokenizer.eos_token}")
        print(f"EOS token ID: {self.tokenizer.eos_token_id}")
        print(f"PAD token: {self.tokenizer.pad_token}")
        print(f"PAD token ID: {self.tokenizer.pad_token_id}")

    def prepare_dataset(self, dataset, packing):
        def tokenize(example, tokenizer):
            messages = example["messages"] # multiple lists of dicts

            assistant_idx = next((i for i, m in enumerate(messages) if m["role"] == "assistant"), None)
            if assistant_idx is None:
                raise ValueError("No assistant message found!")
            
            user_prompt = messages[:assistant_idx]
            full_convo = messages[:assistant_idx + 1]

            user_prompt_ids = tokenizer.apply_chat_template(
                user_prompt,
                tokenize = True,
                add_generation_prompt = True,
                return_tensors = None
            )
            full_convo_ids = tokenizer.apply_chat_template(
                full_convo,
                tokenize = True,
                add_generation_prompt = False,
                return_tensors = None
            )

            # add EOS
            if full_convo_ids[-1] != self.tokenizer.eos_token_id:
                full_convo_ids.append(self.tokenizer.eos_token_id)
            
            completion_mask = [0] * len(user_prompt_ids) + [1] * (len(full_convo_ids) - len(user_prompt_ids))

            if(sum(completion_mask) == 0):
                print(f"Prompt Ids: {tokenizer.decode(user_prompt_ids)}")
                print(f"Full Conversation Ids: {tokenizer.decode(full_convo_ids)}")
                raise ValueError("No completion tokens to learn from!")
            
            assert len(full_convo_ids) == len(completion_mask), "Length Mismatch!"
            print(f"User prompt length: {len(user_prompt_ids)}, Completion length(Model to be trained on): {sum(completion_mask)}, Full Length: {len(full_convo_ids)}, Completion mask length: {len(completion_mask)}")
            return {
                "input_ids": full_convo_ids,
                "completion_mask": completion_mask
            }
        dataset = dataset.map(
            tokenize,
            fn_kwargs={
                "tokenizer": self.tokenizer
            }
        )
        if packing:
            def truncate(example):
                max_len = 200
                if(len(example["input_ids"]) > max_len):
                    # need to truncate
                    start_idx = next((i for i,m in enumerate(example["completion_mask"]) if m == 1), None)
                    if start_idx is not None:
                        # keep atleast some completion tokens, min 10
                        remaining = len(example["input_ids"]) - start_idx
                        mini = min(10, remaining)
                        max_len = max(max_len, start_idx + mini)

                for key in ["input_ids", "completion_mask"]:
                    if key in example:
                        example[key] = example[key][:max_len]
                if(sum(example["completion_mask"]) == 0):
                    raise ValueError("Truncation removed all completion tokens")
                return example
            
            dataset = dataset.map(truncate)
        return dataset

    def train_model(self, dataset, data_collator, batch_size, epochs, learning_rate, eval_dataset, gradient_accumulation_steps):

        self.model.train()
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        self.model, optimizer, data_loader = self.accelerator.prepare(self.model, optimizer, data_loader)

        for epoch in range(epochs):
            if self.accelerator.is_main_process:
                progress = tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}") #tqdm shows visualization (progress bar) start from epoch 1
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
                    self.accelerator.backward(loss) #backward pass
                    if self.accelerator.sync_gradients: #gradient clipping for normalizing gradient values
                        self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                
                loss_for_logging = self.accelerator.gather(loss.detach()).mean()
                total_loss += loss_for_logging.item()
                num_batches += 1
                
                if self.accelerator.is_main_process:
                    progress.set_postfix(loss=loss_for_logging.item())

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            avg_epoch_loss = total_loss / num_batches
            self.accelerator.print(f"Epoch {epoch+1} average loss: {avg_epoch_loss:.4f}")

                # Evaluate after each epoch
            if eval_dataset is not None:
                self.accelerator.wait_for_everyone()
                self.model.eval()
                evaluate_model(self.model, eval_dataset, data_collator, batch_size, self.accelerator)
                self.model.train()  # Switch back to training mode

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