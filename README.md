# FinetuneX

FinetuneX is a ground-up framework with a platform interface designed to democratize LLM fine-tuning by giving users **complete control** over their model training process with an  interface for fine-tuning models on your custom datasets, choosing custom training methods, interacting with your fine-tuned models. 

Built entirely from scratch, FinetuneX implements core architecture from scratch with support for stochastic sampling (top-p, top-k) text generation techniques. It integrates post-training algorithms (SFT, more to come) while keeping the **codebase modular and extensible** for experimentation and future model support.

## Key Features include:

- Fine-tune models on custom datasets
- Interface to interact with the fine-tuned model
- Use multiple LLM architectures
- Experiment with multiple post-training algorithms (SFT, DPO, PPO, RLOO)
- Choose extensive fine-tuning capabilities including Full Fine-Tuning, LoRA and QLoRA
  
It’s designed for researchers, ML enthusiasts, and developers who want full control over the training process while keeping things modular and extensible.

> While FinetuneX references research papers and open-source code from Hugging Face, all core training and architecture modules are implemented independently for transparency and learning.

## Features Roadmap (Work in Progress)

- [ ] **LLM Architecture Support** 
  - [x] Qwen2.5 (0.5B & 1.2B)
  - [x] Llama-3.2-1B
  - [ ] Mistral 7B
  - [ ] Gemma2
  - [ ] GPT2

- [ ] **Fine-tuning Capabilities**
  - [x] Full Fine-tuning
  - [ ] LoRA
  - [ ] QLoRA
        
- [ ] **Post-Training Algorithms**
  - [x] Supervised Fine-Tuning (SFT)
  - [ ] Direct Preference Optimization (DPO)
  - [ ] Proximal Policy Optimization (PPO)
  - [ ] REINFORCE Leave-One-Out (RLOO)
        
- [x] **Interface for Inference**
  - [x] Upload your own dataset
  - [x] Select training configs
  - [x] Train & monitor progress
  - [x] Chat with your fine-tuned model
  - [x] Compare base and tuned model responses
  - [ ] Download the trained weights

## Architecture Details
<img width="1030" height="550" alt="image" src="https://github.com/user-attachments/assets/d7edd961-ed4e-4943-92b2-b5360b6d7906" />


FinetuneX implements the architecture from scratch for Qwen2 0.5B and Llama3.2 1B. Qwen2 follows a similar architecture to Qwen3, but with denser configuration (24 layers, wider hidden dimensions). Have a look at  `finetunex/base/config.py` 

The architecture for Qwen2 incorporates several key components: Group Query Attention (GQA), Root Mean Square Normalization (You will find RMSNorm applied at: Pre-Attention Normalization and Post-Attention Normalization), Rotary Positional Encodings for encoding position information, QKV bias in attention mechanism and SwiGLU activation in FeedForward Network. 

### Unified Architecture Design
Both Qwen2 and LLaMA3 models share the same core transformer architecture (with slight difference in Attention Layer: for Llama no QKV bias) allows to:
- Core components can be reused across similar architecture.
- A shared base class inherited by all models makes it easy to extend support for new architectures.

## Dataset Support
#### For (SFT)
| Format | Required Fields |
| :------ | :---------: |
| CSV | `question`, `answer` columns |
> Note:
> `question` -> user query,
> `answer` -> assistant response.

> User-uploaded data is automatically converted to **ChatML** format - same as the [Dolly](https://huggingface.co/datasets/philschmid/dolly-15k-oai-style) style dataset from huggingface

## Quickstart Guide:
- Install dependencies

  ``` bash
     pip install -r requirements.txt
  ```
- Run `app.py`

    ``` bash
          python app.py
    ```
- Configure your training hyperparameters (defaults shown below)
    - `learning_rate` = 1e-5 , `epochs` = 3, `weight_decay` = 0.001   
    - Edit `main.py` ->  SFT_Config() to customize
        
- Model Variants:
  - Add config for more variants of Qwen or LLama models in `base/config.py`

- Training Details
  - Cosine decay Learning Rate schedular with short warmup phase
  - Gradient clipping enabled
  - Gradient accumulation steps = 8 with batch size = 1 (avoid CUDA OOM)
  - Add `self.model.gradient_checkpointing_enable()` (reduces CUDA memory usage)
  - Uses Accelerator for Distributed Training
    - [Launch distributed training from Jupyter notebook](https://huggingface.co/docs/accelerate/en/basic_tutorials/notebook). [Checkout Notebook](https://github.com/huggingface/notebooks/blob/main/examples/accelerate_examples/simple_cv_example.ipynb)
    - [Launch accelerator script](https://huggingface.co/docs/accelerate/en/basic_tutorials/launch)

> FinetuneX is **Research-Oriented** - built with experimentation in mind

## References
- [HuggingFace TRL](https://github.com/huggingface/trl)
- [HuggingFace Transformer](https://github.com/huggingface/transformers)
- [Qwen 2.5 Technical Report](https://arxiv.org/abs/2412.15115)
- [SFT Memorizes, RL Generalizes](https://arxiv.org/abs/2501.17161)
- [Rotary Position Embedding](https://arxiv.org/abs/2104.09864)

Demo:

https://github.com/user-attachments/assets/e8c26301-79b9-41a6-9808-bd3bc27a3426

Check out the blogs I wrote while building FinetuneX:
- https://www.notion.so/The-Math-behind-Rotating-Attention-23d71454a834808a879df30519ded1df?source=copy_link
- https://www.notion.so/How-transpose-pretends-to-transpose-in-PyTorch-1aae2160433a41438a79771c83a620a1?source=copy_link
- https://www.notion.so/Memory-layout-23d71454a8348001bdbed261efd2746c?source=copy_link
