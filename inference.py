# ====================
# INFERENCE
# ====================

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def infer(dataset):
    print("\n" + "="*50)
    print("STARTING INFERENCE")
    print("="*50)

    # Load the fine-tuned model and tokenizer
    output_dir = "/kaggle/working/finetuned_qwen"  
    model = AutoModelForCausalLM.from_pretrained(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(output_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()  

    print(f"Model loaded on device: {device}")

    print("\nSample training data:")
    for i in range(5):
        print(f"{i+1}. {dataset[i]['messages'][0]['content']}")

    print("\n" + "="*50)
    print("GENERATING RESPONSE")
    print("="*50)

    prompt = "What are the 4 oceans on earth"
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

    inputs = tokenizer(chat_input, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,        
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    outputs = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)]

    # Decode the full response
    full_response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    print(f"\nFull response:\n{full_response}")

    # Extract only the assistant's response
    if chat_input in full_response:
        assistant_response = full_response.replace(chat_input, "").strip()
        print(f"\nAssistant response only:\n{assistant_response}")
    else:
        print(f"\nCouldn't extract assistant response. Full output:\n{full_response}")

    print("\n" + "="*50)
    print("INFERENCE COMPLETE!")
    print("="*50)