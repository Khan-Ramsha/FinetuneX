from datasets import load_dataset
from sft_trainer import SFT
from inference import infer
from data_collator import DataCollator

def main():
    dataset = load_dataset("philschmid/dolly-15k-oai-style", split="train")
    dataset = dataset.select(range(500))
    train_test_split = dataset.train_test_split(test_size=0.2, seed=42)  # 80% train, 20% eval
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]
    sft = SFT(model = "Qwen/Qwen2.5-0.5B-Instruct", pad_token=0)
    tokenized_data = sft.prepare_dataset(dataset = train_dataset)
    eval_dataset = sft.prepare_dataset(dataset = eval_dataset)
    collator = DataCollator(pad_token_id=sft.tokenizer.pad_token_id, completion_only_loss= True)
    sft.train_model(tokenized_data, collator, batch_size=1, epochs = 3, learning_rate = 1e-5, eval_dataset=eval_dataset, gradient_accumulation_steps=8)

    infer()

    print("\n" + "=" * 50)
    print("DATA PREPARED FOR MODEL TRAINING & MODEL TRAINING STARTS")

if __name__ == "__main__":
    main()