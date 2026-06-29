# dpo_config.py
# ---------------------------------------------------------------------------
# DPO training configuration for FinetuneX.
# Mirrors the style of sft_config.py so it plugs in without surprises.
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field


@dataclass
class DPOConfig:
    # ── Model ─────────────────────────────────────────────────────────────
    model_name: str = field(
        default="Qwen2.5-0.5B",
        metadata={"help": "Architecture to load; one of 'Qwen2.5-0.5B' | 'Llama-3.2-1B'"},
    )
    ref_model_path: str = field(
        default=None,
        metadata={
            "help": (
                "Path to a separately saved reference model checkpoint "
                "(output of save_pretrained / model.bin + config.json). "
                "When None the policy model is deep-copied and frozen at "
                "init time, giving an identical β-reference."
            )
        },
    )

    beta: float = field(
        default=0.1,
        metadata={
            "help": (
                "KL-penalty coefficient β from the DPO objective. "
                "Larger β → stronger regularisation toward the reference. "
                "Typical range: 0.05 – 0.5."
            )
        },
    )
    label_smoothing: float = field(
        default=0.0,
        metadata={
            "help": (
                "Adds a conservative regulariser by mixing a small amount of "
                "the *wrong* label into the cross-entropy loss. "
                "0.0 = standard DPO. 0.1 recommended for noisy preference data."
            )
        },
    )
    loss_type: str = field(
        default="sigmoid",
        metadata={
            "help": (
                "DPO loss variant: 'sigmoid' (original paper) or "
                "'ipo' (IPO, Bradley-Terry-free variant)."
            )
        },
    )

    # ── Dataset ────────────────────────────────────────────────────────────
    max_length: int = field(
        default=512,
        metadata={"help": "Maximum token length for both chosen & rejected sequences."},
    )

    # ── Optimisation ───────────────────────────────────────────────────────
    use_flash_attn: bool = field(
        default=False,
        metadata={"help":"use flash attention triton kernels"}
    )    
    seed: int = field(
        default=42,
        metadata={"help": "Random seed for reproducibility."},
    )
    learning_rate: float = field(
        default=1e-5,
        metadata={"help": "Peak learning rate for AdamW (cosine schedule with warmup)."},
    )
    weight_decay: float = field(
        default=0.0,
        metadata={"help": "L2 weight-decay for AdamW."},
    )
    epochs: int = field(
        default=3,
        metadata={"help": "Number of full passes through the preference dataset."},
    )
    batch_size: int = field(
        default=4,
        metadata={"help": "Per-device micro-batch size."},
    )
    gradient_accumulation_steps: int = field(
        default=8,
        metadata={
            "help": "Accumulate gradients over N micro-batches before an optimiser step."
        },
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "Gradient clipping max-norm."},
    )

    # ── I/O ────────────────────────────────────────────────────────────────
    output_dir: str = field(
        default="./dpo_results",
        metadata={"help": "Directory to save model checkpoints and final weights."},
    )
    report_to_wandb: bool = field(
        default=False,
        metadata={"help": "Stream training metrics to Weights & Biases."},
    )
    