from sft_trainer import SFT
from data_collator import DataCollator
import pandas as pd
from data_preprocessing import ChatMLPreprocessor
from sft_config import SFTConfig
import wandb

def main(file, model):
    data = pd.read_csv(file)
    obj = ChatMLPreprocessor(data)
    data = obj.convert_to_chatml()
    train_test_split = data.train_test_split(test_size=0.2, seed=42)
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]
    training_args = SFTConfig()
    sft = SFT(
        model = model, #user selected model
        pad_token=0,
        args = training_args
    )
    tokenized_data = sft.prepare_dataset(dataset = train_dataset)
    eval_dataset = sft.prepare_dataset(dataset = eval_dataset)
    collator = DataCollator(pad_token_id=sft.tokenizer.pad_token_id, completion_only_loss= True)
    if training_args.report_to_wandb:
        wandb.init(
            project="finetunex-sft",
            config=vars(training_args),
            name=f"{model}-{training_args.epochs}-epochs"
        )
    sft.train_model(tokenized_data, collator, eval_dataset=eval_dataset)