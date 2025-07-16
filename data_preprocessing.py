"""Converting user uploaded data into ChatML format!!"""
from datasets import Dataset
import pandas as pd

class ChatMLPreprocessor:
    def __init__(self, path: str, ):
        self.path = path

    def convert_to_chatml(self):
        df = pd.read_csv(self.path)
        data = []

        for _,row in df.iterrows():
            data.append(
                {
                    "messages":[
                        {
                            "content": row["question"],
                            "role": "user"
                        },
                        {
                            "content": row["answer"],
                            "role": "assistant"
                        }
                    ]
                }
            )
        data = Dataset.from_list(data)
        print(data)
        print(type(data))
        return data
    
obj = ChatMLPreprocessor("data/chatbot_qna.csv")
obj.convert_to_chatml()