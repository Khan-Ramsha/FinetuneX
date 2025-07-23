"""Converting user uploaded data into ChatML format!!"""
from datasets import Dataset
import pandas as pd

class ChatMLPreprocessor:
    def __init__(self, data):
        self.data = data

    def convert_to_chatml(self):
        q_col, a_col = list(self.data.columns)
        data = []
        for _,row in self.data.iterrows():
            data.append(
                {
                    "messages":[
                        {
                            "content": row[q_col],
                            "role": "user"
                        },
                        {
                            "content": row[a_col],
                            "role": "assistant"
                        }
                    ]
                }
            )
        data = Dataset.from_list(data)
        print(data)
        print(type(data))
        return data