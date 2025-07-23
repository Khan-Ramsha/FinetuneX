from sft_trainer import SFT
from inference import infer
from data_collator import DataCollator
import pandas as pd
from datasets import Dataset
from data_preprocessing import ChatMLPreprocessor

def main(file):
    data = pd.read_csv(file)
    obj = ChatMLPreprocessor(data)
    data = obj.convert_to_chatml()
    train_test_split = data.train_test_split(test_size=0.2, seed=42)
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]
    sft = SFT(model = "Qwen/Qwen2.5-0.5B-Instruct", pad_token=0)
    tokenized_data = sft.prepare_dataset(dataset = train_dataset)
    eval_dataset = sft.prepare_dataset(dataset = eval_dataset)
    collator = DataCollator(pad_token_id=sft.tokenizer.pad_token_id, completion_only_loss= True)
    sft.train_model(tokenized_data, collator, batch_size=1, epochs = 3, learning_rate = 1e-5, eval_dataset=eval_dataset, gradient_accumulation_steps=8)
    print("\n" + "=" * 50)
    print("DATA PREPARED FOR MODEL TRAINING & MODEL TRAINING STARTS")