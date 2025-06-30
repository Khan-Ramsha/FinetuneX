
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from evaluate import evaluate_model
import os

output_dir = "./finetuned_qwen"
os.makedirs(output_dir, exist_ok=True)

class SFT:
    def __init__(self, model: str, pad_token: int):
        self.model_name = model
        self.pad_token = pad_token
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.evaluate_model = evaluate_model
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"EOS token: {self.tokenizer.eos_token}")
        print(f"EOS token ID: {self.tokenizer.eos_token_id}")
        print(f"PAD token: {self.tokenizer.pad_token}")
        print(f"PAD token ID: {self.tokenizer.pad_token_id}")

    def prepare_dataset(self, dataset, packing):
        def tokenize(example, tokenizer, dataset_text_field):
            messages = example["messages"]

            try:
                # Get user messages with generation prompt
                user_messages = []
                for msg in messages:
                    user_messages.append(msg)
                    if msg["role"] == "user":
                        break
                
                print(f"Messages: {messages}")
                print(f"User Messages: {user_messages}")
                
                user_part_ids = tokenizer.apply_chat_template(
                    user_messages,
                    add_generation_prompt=True,
                    return_tensors=None
                )

                full_input_ids = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=False,
                    return_tensors=None
                )

                if full_input_ids[-1] != tokenizer.eos_token_id:
                    full_input_ids.append(tokenizer.eos_token_id)

                completion_mask = [0] * len(full_input_ids)
                assistant_start = len(user_part_ids)

                if assistant_start >= len(full_input_ids):
                    assistant_start = len(full_input_ids) - 1

                for i in range(assistant_start, len(full_input_ids)):
                    completion_mask[i] = 1

                print(f"User prompt length: {len(user_part_ids)}, Completion length(Model to be trained on): {sum(completion_mask)}, Full Length: {len(full_input_ids)}")
                return {
                    "input_ids": full_input_ids,
                    "completion_mask": completion_mask
                }

            except Exception as e:
                print(f"Tokenization failed: {e}")
                input_ids = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=False,
                    return_tensors=None
                )

                if input_ids[-1] != tokenizer.eos_token_id:
                    input_ids.append(tokenizer.eos_token_id)

                completion_mask = [1] * len(input_ids)  # Fallback: train on everything
                
                return {
                    "input_ids": input_ids,
                    "completion_mask": completion_mask
                }

        dataset = dataset.map(
            tokenize,
            fn_kwargs={
                "tokenizer": self.tokenizer,
                "dataset_text_field": "content"
            }
        )

        if packing:
            def truncate(example):
                for key in ["input_ids", "completion_mask"]:
                    if key in example:
                        example[key] = example[key][:256]
                return example
            dataset = dataset.map(truncate)

        return dataset

    def train_model(self, dataset, data_collator, batch_size, epochs, learning_rate, eval_dataset):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.train()
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)

        optimizer = AdamW(self.model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            progress = tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}")
            total_loss = 0.0
            num_batches = 0

            for batch in progress:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                print(f"Loss: {loss}")
                total_loss += loss.item()
                num_batches += 1
                progress.set_postfix(loss=loss.item())

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            avg_epoch_loss = total_loss / num_batches
            print(f"Epoch {epoch+1} average loss: {avg_epoch_loss:.4f}")

                # Evaluate after each epoch
            if eval_dataset is not None:
                evaluate_model(self.model, eval_dataset, data_collator, batch_size)
                self.model.train()  # Switch back to training mode

            # Save model after training
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print("\n" + "=" * 50)
        print("MODEL TRAINING DONE!")
        print("\n" + "=" * 50)
        print(f"Model saved to {output_dir}")