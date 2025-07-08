from datasets import load_dataset
from sft_trainer import SFT
from data_collator import DataCollator
from inference import infer

def main():
    dataset = load_dataset("philschmid/dolly-15k-oai-style", split="train")
    dataset = dataset.select(range(1000))

    sft = SFT(model = "Qwen/Qwen2.5-0.5B-Instruct", pad_token=0)
    tokenized_data = sft.prepare_dataset(dataset = dataset)
    collator = DataCollator(pad_token_id=sft.tokenizer.pad_token_id, completion_only_loss= True)
    sft.train_model(tokenized_data, collator, batch_size=1, epochs = 3, learning_rate = 5e-5, eval_dataset=tokenized_data, gradient_accumulation_steps=16)
    
    infer()
    
    print("\n" + "=" * 50)
    print("DATA PREPARED FOR MODEL TRAINING & MODEL TRAINING STARTS")

if __name__ == "__main__":
    main()