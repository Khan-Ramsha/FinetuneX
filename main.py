from sft_trainer import SFT
from data_collator import DataCollator
from sft_config import SFTConfig
from build_dataset import create_dataset
from finetunex.distributed.setup import distributed_setup
import wandb
import argparse
import torch
import torch.multiprocessing as mp
from torch.distributed import destroy_process_group
from inference import infer
from dpo_config import DPOConfig
from dpo_trainer import DPOTrainer
from dpo_data_preprocessing import PreferenceDPODataset
from dpo_data_collator import DataCollatorForDPO
import pandas as pd
from datasets import load_dataset


def main(rank: int, world_size: int, model_name: str, dataset: str, training_args: SFTConfig, post_training:str):
    try: 
        if training_args.distributed_strategy in ["ddp", "fsdp"]:
            distributed_setup(rank, world_size)
        if post_training == "sft":
            train_dataset, eval_dataset = create_dataset(dataset)
            sft = SFT(model=model_name, pad_token=0, args=training_args, rank=rank)
        
            tokenized_train = sft.prepare_dataset(train_dataset)
            tokenized_eval  = sft.prepare_dataset(eval_dataset)
        
            collator = DataCollator(pad_token_id=sft.tokenizer.pad_token_id, completion_only_loss=True)
        
            if rank == 0 and training_args.report_to_wandb:
                wandb.init(
                    project="finetunex-sft",
                    config=vars(training_args),
                    name=f"{model_name}-{training_args.epochs}-epochs"
                )
        
            sft.train_model(
                tokenized_train, collator,
                eval_dataset=tokenized_eval,
                rank=rank,
                world_size=world_size,
            )
        elif post_training == "dpo":
            trainer = DPOTrainer(training_args)
            df = load_dataset("json",data_files=dataset)
            split = df["train"].train_test_split(
                test_size=0.2,
                seed=42
            )
            train_ds = PreferenceDPODataset(split["train"], trainer.tokenizer)
            eval_ds  = PreferenceDPODataset(split["test"],  trainer.tokenizer)
            collator = DataCollatorForDPO(pad_token_id=trainer.tokenizer.pad_token_id)
            trainer.train(train_ds, eval_ds, collator)
    finally:
        if training_args.distributed_strategy != "single":
            destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen2.5-0.5B")
    parser.add_argument("--dataset", default="openai/gsm8k") #for DPO pass the dataset path (stored in json format)
    parser.add_argument("--epochs", type=int,   default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size",type=int,   default=2)
    parser.add_argument("--grad_accum",type=int,   default=2)
    parser.add_argument("--output_dir",default="./finetuned")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--distributed_strategy", default="single", choices=["single", "ddp","fsdp"])
    parser.add_argument("--post_training", default="sft", choices=["sft","dpo"])
    args = parser.parse_args()
    if args.post_training == "sft":
        training_args = SFTConfig(
            epochs=args.epochs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            output_dir=args.output_dir,
            report_to_wandb=args.wandb,
            distributed_strategy=args.distributed_strategy
        )
    elif args.post_training == "dpo":
        training_args = DPOConfig(model_name="Qwen2.5-0.5B", beta=0.1, epochs=3, distributed_strategy=args.distributed_strategy)
        
    world_size = max(torch.cuda.device_count(), 1)
    if training_args.distributed_strategy in ["ddp", "fsdp"] and world_size > 1:
        mp.spawn(main, args=(world_size, args.model, args.dataset, training_args, args.post_training), nprocs=world_size)    
    else:
        main(rank=0, world_size=1, model_name=args.model, dataset= args.dataset, training_args=training_args, post_training=args.post_training)