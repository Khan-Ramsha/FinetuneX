# ====================
# INFERENCE
# ====================

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def infer(prompt, model_path):
    print("\n" + "="*50)
    print("STARTING INFERENCE")
    print("="*50)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    print(f"Model loaded on device: {device}")

    print("\n" + "="*50)
    print("GENERATING RESPONSE")
    print("="*50)

    print(f"User prompt: {prompt}")

    # Prepare messages for chat template
    messages = [
        {"role": "user", "content": prompt}
    ]

    chat_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    print(f"Chat input: {repr(chat_input)}")

    inputs = tokenizer(chat_input, return_tensors="pt").to(device) #tokenize the prompt
    max_new_tokens = model.config.max_position_embeddings - inputs["input_ids"].shape[1]
    print(f"Max tokens by qwen2 0.5B: {max_new_tokens}")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens = max_new_tokens,
            temperature = 0.5,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    outputs = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)] #generate() returns prompt tokens and generated tokens

    # Decode the full response
    full_response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    print(f"\nFull response:\n{full_response}")
    return full_response

    print("\n" + "="*50)
    print("INFERENCE COMPLETE!")
    print("="*50)