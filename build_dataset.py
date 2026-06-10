import pandas as pd
from data_preprocessing import ChatMLPreprocessor
from datasets import load_dataset

def create_dataset(name):
    data = load_dataset(name, "main")
    df = pd.DataFrame(data["train"])
    df = df.select(range(min(300, len(data["train"]))))
    data = ChatMLPreprocessor(df).convert_to_chatml()
    split = data.train_test_split(
        test_size=0.2,
        seed=42
    )

    return split["train"], split["test"]