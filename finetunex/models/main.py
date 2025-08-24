
from transformers import AutoModelForCausalLM, AutoTokenizer
from finetunex.text_generation.generate import generate
from finetunex.base.config import Config
from finetunex.models.qwen2.model import Qwen2Model
import torch


hf_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B") #from hf

hf_state = hf_model.state_dict()
print(list(hf_state.keys())[:50]) 
config = Config.config_from_model("Qwen2.5-0.5B")
model = Qwen2Model(config=config) #self implemented architecture

def mapping(name):
    hf_name = f"model.{name}" #make it same as hf naming conventions
    hf_name = hf_name.replace(".attn.", ".self_attn.")
    return hf_name

def load_pretrained_weights(model, hf_model):
    model_state_dict = model.state_dict()
    hfmodel_state_dict = hf_model.state_dict()
    for name, params in list(model_state_dict.items()):
        if "rotary.inv_freq" in name:
            continue #skip it
        hf_name = mapping(name)
        if hf_name in hfmodel_state_dict:
            model_state_dict[name] = hfmodel_state_dict[hf_name] #manually assigning weights
        else:
            print(f"Skipping, {hf_name} is not in hf state dict")
    return model_state_dict

model_state_dict = load_pretrained_weights(model, hf_model)
model.load_state_dict(model_state_dict, strict = False)
model.eval()
print("GENERATING RESPONSE")
prompt="What is capital of France?"
print(f"User prompt: {prompt}")

messages = [
    {"role": "user", "content": prompt}
]
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
chat_input = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
print(f"Chat input: {repr(chat_input)}")
inputs = tokenizer(chat_input, return_tensors = "pt") #tokenize the prompt
inputs= inputs["input_ids"]
print(inputs)
max_new_tokens = model.config.max_position_embeddings 
print(f"Max tokens by qwen2 0.5B: {max_new_tokens}")
with torch.no_grad():
    outputs = generate(
        model,
        inputs,
        max_new_tokens = max_new_tokens,
        top_p = 0.6, 
        top_k = 100,
        temperature = 0.5,
        stop_tokens=([tokenizer.eos_token_id],)
    )
output = list(outputs)
token_ids = [token.item() for token in output] 
full_response = tokenizer.decode(token_ids)
print(f"\nFull response:\n{full_response}")