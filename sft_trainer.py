import torch
import torch.nn as nn
from torch.nn import functional as F
from finetunex.models.qwen2.save_load import from_pretrained
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup
)
from finetunex.base.config import Config
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from accelerate import Accelerator
from evaluate import evaluate_model
from early_stopping import EarlyStopping
from sft_config import SFTConfig
from finetunex.models.qwen2.save_load import load_weights_into_qwen
from finetunex.models.qwen2.save_load import save_pretrained
from finetunex.models.qwen2.model import Qwen2Model
from finetunex.models.llama.model import LlamaModel
from finetunex.models.llama.save_load import save_pretrained, load_weights_into_llama

output_dir = "./finetuned"
os.makedirs(output_dir, exist_ok=True)


class SFT:
    def __init__(self, model: str, pad_token: int, args: SFTConfig):
        self.args = args
        self.model_name = model
        self.pad_token = pad_token
        config = Config.config_from_model(self.model_name)
        if self.model_name == "Qwen2.5-0.5B":
            hf_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B") #from hf
            hf_model_state_dict = hf_model.state_dict()
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
            self.model = Qwen2Model(config=config) #self implemented architecture
            load_weights_into_qwen(self.model, config, hf_model_state_dict)
        if self.model_name == "Llama-3.2-1B":
            hf_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B") #from hf
            hf_model_state_dict = hf_model.state_dict()
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
            self.model = LlamaModel(config=config) #self implemented architecture
            load_weights_into_llama(self.model, config, hf_model_state_dict)
        self.accelerator = Accelerator(gradient_accumulation_steps=8)
        # self.model.gradient_checkpointing_enable()
        # self.model.config.use_cache = False
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"EOS token: {self.tokenizer.eos_token}")
        print(f"EOS token ID: {self.tokenizer.eos_token_id}")
        print(f"PAD token: {self.tokenizer.pad_token}")
        print(f"PAD token ID: {self.tokenizer.pad_token_id}")

        # For tracking learning rate
        self.lr_history = []
        self.step_history = []
    
    def prepare_dataset(self, dataset):
        def tokenize(example, tokenizer):
            messages = example["messages"]
            assistant_idx = next((i for i, m in enumerate(messages) if m["role"] == "assistant"), None)
            if assistant_idx is None:
                raise ValueError("No assistant message found!")

            user_prompt = messages[:assistant_idx]
            full_convo = messages[:assistant_idx + 1]

            if self.model_name == "Qwen2.5-0.5B":
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
            if self.model_name == "Llama-3.2-1B":
                 # Convert messages to string format for Llama
                user_prompt_str = ""
                for msg in user_prompt:
                    user_prompt_str += f"{msg['role']}: {msg['content']}\n"
                user_prompt_str += "assistant: "  # Add generation prompt
                
                full_convo_str = ""
                for msg in full_convo:
                    full_convo_str += f"{msg['role']}: {msg['content']}\n"
                
                user_prompt_ids = tokenizer(
                    user_prompt_str,
                    return_tensors=None,  
                    add_special_tokens=True
                )["input_ids"]
                
                full_convo_ids = tokenizer(
                    full_convo_str,
                    return_tensors=None,  
                    add_special_tokens=True
                )["input_ids"]

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

    def train_model(self, dataset, data_collator, batch_size, eval_dataset, gradient_accumulation_steps):
        self.model.train()
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate,weight_decay=self.args.weight_decay) #base learning rate

        total_batches = self.args.epochs * len(data_loader)
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
        print(f"Base learning rate: {self.args.learning_rate}")
        print(f"[Before training] LR = {scheduler.get_last_lr()[0]:.8f}")

        global_step = 0
        early_stop = EarlyStopping()
        for epoch in range(self.args.epochs):
            if self.accelerator.is_main_process:
                progress = tqdm(data_loader, desc=f"Epoch {epoch+1}/{self.args.epochs}")
            else:
                progress = data_loader

            total_loss = 0.0
            num_batches = 0

            for step, batch in enumerate(progress):
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"]

                with self.accelerator.accumulate(self.model):
                    outputs = self.model(input_ids=input_ids, attn_mask = attention_mask, labels=labels)
                    loss = outputs['loss']
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

                loss_for_logging = self.accelerator.gather(outputs['loss'].detach()).mean()
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
                loss = evaluate_model(self.model, eval_dataset, data_collator, batch_size, self.accelerator)
                if early_stop.early_stopping(loss):
                    self.accelerator.print("Early stopping triggered!")
                    break
                self.model.train()

        #Save model after training
        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        if self.accelerator.is_main_process:
            model_state_dict = unwrapped_model.state_dict()
            save_pretrained(output_dir, model_state_dict, unwrapped_model.config)
            # self.tokenizer.save_pretrained(output_dir)

        print("\n" + "=" * 50)
        print("MODEL TRAINING DONE!")
        print("\n" + "=" * 50)
        print(f"Model saved to {output_dir}")