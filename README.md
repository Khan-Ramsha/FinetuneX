# FinetuneX
FinetuneX is a powerful, ground-up framework with a platform interface designed to democratize LLM fine-tuning by giving users **complete control** over their model training process. FinetuneX provides an intuitive interface for fine-tuning models on your custom datasets, choosing custom training methods, interacting with your trained models, and downloading them.

FinetuneX allows you to:
- Fine-tune models on custom datasets
- Interface to interact with the fine-tuned model
- Use multiple LLM architecture 
- Apply custom training methods with different post-training algorithms (SFT, DPO, PPO, RLOO)
- Choose extensive fine-tuning capabilities including Full Fine-Tuning, LoRA and QLoRA
- MultiModal Vision Support

FinetuneX is a built-from-scratch framework that provides a platform to ML enthusiasts, developers, and researchers or anyone to easily experiment with and fine-tune models.

Implementation Note:
While building FinetuneX, I’ve referred to open-source implementations from Hugging Face. However, all core components have been re-implemented from scratch in a simplified way. 

## Features Roadmap (Work in Progress)

- [ ] **LLM Architecture Support**
  - [x] Qwen2.5-0.5B-Instruct (adding support for more variants soon)
  - [ ] Llama 3.1
  - [ ] Mistral 7B
  - [ ] Gemma 2
        
- [ ] **Interface for Inference**
- [ ] **Download the Model**
- [ ] **Fine-tuning Capabilities**
  - [x] Full Fine-tuning
  - [ ] LoRA
  - [ ] QLoRA
        
- [ ] **Training Algorithms**
  - [x] Supervised Fine-Tuning (SFT)
  - [ ] Direct Preference Optimization (DPO)
  - [ ] Proximal Policy Optimization (PPO)
  - [ ] REINFORCE Leave-One-Out (RLOO)

Demo:


https://github.com/user-attachments/assets/e8c26301-79b9-41a6-9808-bd3bc27a3426

