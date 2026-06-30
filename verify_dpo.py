import torch
from transformers import AutoTokenizer
from dpo_data_preprocessing import PreferenceDPODataset
from dpo_data_collator import DataCollatorForDPO

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Tiny dummy data
dummy = [
    {"prompt": "cancel my order", "chosen": "Sure, I'll cancel it right away.", "rejected": "Whatever."},
    {"prompt": "where is my package", "chosen": "Your package arrives tomorrow.", "rejected": "No idea."},
]

ds = PreferenceDPODataset(dummy, tokenizer, max_length=512)
collator = DataCollatorForDPO(pad_token_id=tokenizer.pad_token_id)
batch = collator([ds[0], ds[1]])

# CHECK 1: Mask correctness 
# Decode only the tokens where mask=1 (response tokens)
# Should show ONLY the assistant response, not the prompt
for i in range(2):
    chosen_response_tokens = batch["chosen"][i][batch["chosen_mask"][i].bool()]
    print(f"[Sample {i}] Chosen response tokens decoded:")
    print(repr(tokenizer.decode(chosen_response_tokens)))
    # Expected: "Sure, I'll cancel it right away.<eos>" (no user prompt)
# CHECK 2: Attention mask covers prompt + response, not padding ──
print("\nChosen attention mask (last 10 positions):", batch["chosen_attention_mask"][0][-10:])
# Padding positions should be 0

# CHECK 3: Initial loss should be ~0.693 
# When policy == ref, log-ratios cancel -> loss = -log(sigmoid(0)) = log(2) -> 0.693
from dpo_config import DPOConfig
from dpo_trainer import DPOTrainer, _response_log_probs, _dpo_loss
import copy

args = DPOConfig(model_name="Qwen2.5-0.5B", beta=0.1, epochs=1)
trainer = DPOTrainer(args)  # ref = deep copy of policy (identical weights)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trainer.model.to(device)
trainer.ref_model.to(device)

batch_d = {k: v.to(device) for k, v in batch.items()}

with torch.no_grad():
    p_c  = _response_log_probs(trainer.model,batch_d["chosen"],   batch_d["chosen_attention_mask"],   batch_d["chosen_mask"].float())
    p_r  = _response_log_probs(trainer.model,batch_d["rejected"], batch_d["rejected_attention_mask"], batch_d["rejected_mask"].float())
    r_c  = _response_log_probs(trainer.ref_model, batch_d["chosen"],   batch_d["chosen_attention_mask"],   batch_d["chosen_mask"].float())
    r_r  = _response_log_probs(trainer.ref_model, batch_d["rejected"], batch_d["rejected_attention_mask"], batch_d["rejected_mask"].float())

loss, c_rew, r_rew = _dpo_loss(p_c, p_r, r_c, r_r, beta=0.1, label_smoothing=0.0, loss_type="sigmoid")

print(f"\nInitial loss: {loss.item():.4f}  (expected ~0.6931)")
print(f"Chosen reward: {c_rew.mean().item():.4f}  (expected ~0.0)")
print(f"Rejected reward: {r_rew.mean().item():.4f}  (expected ~0.0)")