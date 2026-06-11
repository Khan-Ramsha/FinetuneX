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

def main(rank: int, world_size: int, model_name: str, dataset: str, training_args: SFTConfig):
    try: 
        if training_args.distributed_strategy in ["ddp", "fsdp"]:
            distributed_setup(rank, world_size)
    
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
    finally:
        if training_args.distributed_strategy != "single":
            destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     default="Qwen2.5-0.5B")
    parser.add_argument("--dataset",     default="openai/gsm8k")
    parser.add_argument("--epochs",    type=int,   default=3)
    parser.add_argument("--lr",        type=float, default=1e-5)
    parser.add_argument("--batch_size",type=int,   default=2)
    parser.add_argument("--grad_accum",type=int,   default=2)
    parser.add_argument("--output_dir",default="./finetuned")
    parser.add_argument("--wandb",  action="store_true")
    parser.add_argument("--distributed_strategy", default="single", choices=["single", "ddp","fsdp"])
    args = parser.parse_args()

    training_args = SFTConfig(
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        output_dir=args.output_dir,
        report_to_wandb=args.wandb,
        distributed_strategy=args.distributed_strategy
    )

    world_size = max(torch.cuda.device_count(), 1)
    if training_args.distributed_strategy in ["ddp", "fsdp"] and world_size > 1:
        mp.spawn(main, args=(world_size, args.model, args.dataset, training_args), nprocs=world_size)
    else:
        main(rank=0, world_size=1, model_name=args.model, dataset= args.dataset, training_args=training_args)
    model_path = f"{training_args.output_dir}/checkpoint_epoch_{training_args.epochs-1}"
    prompt = input("Enter the prompt: ")
    result = infer(prompt, model_path, args.model)
    print(f"\n== Model Response == \n {result}")