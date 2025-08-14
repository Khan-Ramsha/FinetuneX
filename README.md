# FinetuneX
FinetuneX is a ground-up framework with a platform interface designed to democratize LLM fine-tuning - built entirely from scratch, including core architecture implementations (Rotary Position Embeddings, RMSNorm, custom attention layers) for models like Qwen, LLaMA, and more to come.

It’s designed for researchers, ML enthusiasts, and developers who want full control over the training process while keeping things modular and extensible.

> While FinetuneX references research papers and open-source code from Hugging Face, all core training and architecture modules are implemented independently for transparency and learning.

## Features Roadmap (Work in Progress)

- [ ] **LLM Architecture Support** 
  - [x] Qwen2.5-0.5B-Instruct (adding support for more variants soon)
  - [ ] Llama 3.1
  - [ ] Mistral 7B
  - [ ] Gemma 2

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

## Dataset Support
| Format | Required Fields |
| :------ | :---------: |
| CSV | `question`, `answer` columns |
> Note:
> `question` -> user query
> `answer` -> assistant response.
> User-uploaded data is automatically converted to **ChatML** format - same as the [Dolly](https://huggingface.co/datasets/philschmid/dolly-15k-oai-style) style dataset from huggingface

## Quickstart Guide:
- Install dependencies
  ``` bash
     pip install requirements.txt
  ```
- Run `app.py`
``` bash
     python app.py
```
- Configure Hyperparameter for training: (below are default values)
    - `learning_rate` = 1e-5 , `epochs` = 5, `weight_decay` = 0.001
  Edit `main.py` ->  SFT_Config() to customize
- Model Variants:
  - Add config for more variants of Qwen or LLama models in `base/config.py`
- Training Details
  - Cosine decay Learning Rate schedular with short warmup phase
  - Gradient clipping enabled
  - Gradient accumulation steps = 8 with batch size = 1 (avoid CUDA OOM)
  - `self.model.gradient_checkpointing_enable()` (reduces CUDA memory usage)
  - Uses Accelerator for Distributed Training
    - [Launch distributed training from Jupyter notebook](https://huggingface.co/docs/accelerate/en/basic_tutorials/notebook) [Notebook](https://github.com/huggingface/notebooks/blob/main/examples/accelerate_examples/simple_cv_example.ipynb)
    - [Launch accelerator script](https://huggingface.co/docs/accelerate/en/basic_tutorials/launch)

> FinetuneX is **Research-Oriented** - built with experimentation in mind

## References
- [HuggingFace TRL](https://github.com/huggingface/trl)
- [Qwen 2.5 Technical Report](https://arxiv.org/abs/2412.15115)
- [SFT Memorizes, RL Generalizes](https://arxiv.org/abs/2501.17161)
- [Rotary Position Embedding](https://arxiv.org/abs/2104.09864)

Demo:

https://github.com/user-attachments/assets/e8c26301-79b9-41a6-9808-bd3bc27a3426

