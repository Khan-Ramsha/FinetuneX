from sft_trainer import SFT
from data_collator import DataCollator
from datasets import load_dataset
from inference import infer

def main():
    dataset = load_dataset("philschmid/dolly-15k-oai-style", split="train")
    dataset = dataset.select(range(70))

    print("\n" + "=" * 50)
    print("PREPARING TRAINING AND TESTING DATA")

    data = dataset.train_test_split(test_size=0.3, shuffle=True, seed = 42)
    train_data = data["train"]
    test_data = data["test"]
    print(f"Training data length: {len(train_data)}")
    print(f"Training data length: {len(test_data)}")

    sft = SFT(model = "Qwen/Qwen2-0.5B", pad_token=0)
    tokenized_data = sft.prepare_dataset(dataset = train_data, packing = True)
    test_data = sft.prepare_dataset(dataset= test_data, packing= True)
    collator = DataCollator(pad_token_id=sft.tokenizer.pad_token_id, completion_only_loss= True)

    sft.train_model(tokenized_data, collator, batch_size=2, epochs = 3, learning_rate = 5e-5, eval_dataset=test_data, gradient_accumulation_steps=4)

    print("\n" + "=" * 50)
    print("DATA PREPARED FOR MODEL TRAINING & MODEL TRAINING STARTS")
    infer(train_data)

if __name__ == "__main__":
    main()