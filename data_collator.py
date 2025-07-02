
import torch
from torch.nn import functional as F

class DataCollator:
    def __init__(self, pad_token_id: int, completion_only_loss: bool = False):
        self.pad_token_id = pad_token_id
        self.completion_only_loss = completion_only_loss
    
    def __call__(self, examples):
       input_ids =[torch.tensor(e['input_ids'], dtype= torch.long) for e in examples]
       max_len = max(len(ids) for ids in input_ids)
       padded_ids = [F.pad(ids, (0, max_len - len(ids)), value = self.pad_token_id) for ids in input_ids]
       attention_mask = [torch.ones(len(ids), dtype=torch.long) for ids in input_ids]
       attention_mask = [F.pad(mask, (0, max_len - len(mask)), value = 0) for mask in attention_mask] #putting zeros where attention not needed!
       
       padded_labels = [ids.clone() for ids in padded_ids]

       #calculate loss for only completion
       if self.completion_only_loss and "completion_mask" in examples[0]:
           for i, e in enumerate(examples):
               mask = torch.tensor(e["completion_mask"], dtype = torch.bool)
               mask = F.pad(mask, (0, max_len - len(mask)), value = 0) #zeros for padding
               padded_labels[i][~mask] = -100
               
       for i, ids in enumerate(padded_labels):
            pad_mask = (padded_ids[i] == self.pad_token_id)
            padded_labels[i][pad_mask] = -100

       # Final batch dict
       output = {
          "input_ids": torch.stack(padded_ids),
          "labels": torch.stack(padded_labels),
          "attention_mask": torch.stack(attention_mask),
       }
       return output
       #returns inputs ids, labels, attention masks (here completion only loss and completion mask is used)

collator = DataCollator(pad_token_id = 0, completion_only_loss = True)
example = [
    {
        "input_ids": [101, 2023, 2003, 2307],  
        "completion_mask": [0, 0, 1, 1],  
    },
    {
        "input_ids": [101, 2009, 2003, 1037, 2204, 2154],
        "completion_mask": [0, 0, 0, 1, 1, 1]
    }
]
result = collator(example)
print(result)