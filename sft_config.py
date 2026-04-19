from dataclasses import dataclass, field
from torch.optim import AdamW
@dataclass
class SFTConfig:
    seed: int = field(
        default=42,
        metadata={"help":"seed for reproducibility"}
    )
    learning_rate : float = field(
        default = 1e-5,
        metadata={"help":"base learning rate for optimizer"}
    )
    epochs: int = field(
        default=3, 
        metadata={"help":"number of times model sees training data"}
    )
    weight_decay:float = field(
        default= 0.001,
        metadata={"help":"prevents overfitting"}  
    ) 
    gradient_accumulation_steps: int = field(
        default=8,
        metadata={"help":"number of steps to accumulate gradients before updating parameters"}
    )
    batch_size: int = field(
        default=8,
        metadata={"help":"number of samples to process in each batch"}
    )
    use_flash_attn: bool = field(
        default=False,
        metadata={"help":"use flash attention triton kernels"}
    )
    output_dir: str = field(
        default="./finetuned",
        metadata={"help":"directory to save the model and training artifacts"}
    )
