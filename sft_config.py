from dataclasses import dataclass, field
from torch.optim import AdamW
@dataclass
class SFTConfig:
    learning_rate : float = field(
        default = 1e-5,
        metadata={"help":"base learning rate for optimizer"}
    )
    epochs: int = field(
        default=5, 
        metadata={"help":"number of times model sees training data"}
    )
    weight_decay:float = field(
        default= 0.001,
        metadata={"help":"prevents overfitting"}  
    ) 