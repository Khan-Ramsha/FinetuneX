import torch
from torch.utils.data import Dataset


class PreferenceDPODataset(Dataset):
    """
    Tokenizes using apply_chat_template — same as SFT's prepare_dataset.
    """
    def __init__(self, data, tokenizer, max_length=512):
        self.encoded = []
        def _apply_chat_template_ids(tokenizer, messages, **kwargs):
            """Reuse the exact same helper from sft_trainer.py"""
            out = tokenizer.apply_chat_template(messages, **kwargs)
            if isinstance(out, list):
                return out
            ids = out["input_ids"] if hasattr(out, "input_ids") else out
            if hasattr(ids, "tolist"):
                ids = ids.tolist()
            return list(ids)

        for entry in data:
            prompt_text = entry["prompt"]
            chosen_text = entry["chosen"]
            rejected_text = entry["rejected"]

            prompt_messages = [
                {"role": "user", "content": prompt_text}
            ]
            chosen_messages = [
                {"role": "user", "content": prompt_text},
                {"role": "assistant", "content": chosen_text},
            ]
            rejected_messages = [
                {"role": "user",      "content": prompt_text},
                {"role": "assistant", "content": rejected_text},
            ]

            prompt_ids = _apply_chat_template_ids(
                tokenizer, prompt_messages,
                tokenize=True, add_generation_prompt=True, return_tensors=None,
            )
            chosen_ids = _apply_chat_template_ids(
                tokenizer, chosen_messages,
                tokenize=True, add_generation_prompt=False, return_tensors=None,
            )
            rejected_ids = _apply_chat_template_ids(
                tokenizer, rejected_messages,
                tokenize=True, add_generation_prompt=False, return_tensors=None,
            )

            if chosen_ids[-1] != tokenizer.eos_token_id:
                chosen_ids.append(tokenizer.eos_token_id)
            if rejected_ids[-1] != tokenizer.eos_token_id:
                rejected_ids.append(tokenizer.eos_token_id)

            if len(chosen_ids) > max_length or len(rejected_ids) > max_length:
                continue

            chosen_mask   = [0] * len(prompt_ids) + [1] * (len(chosen_ids)   - len(prompt_ids))
            rejected_mask = [0] * len(prompt_ids) + [1] * (len(rejected_ids) - len(prompt_ids))

            if sum(chosen_mask) == 0 or sum(rejected_mask) == 0:
                continue  # skip bad examples

            self.encoded.append({
                "chosen": chosen_ids,    # full sequence: prompt + chosen response
                "rejected": rejected_ids,  # full sequence: prompt + rejected response
                "chosen_mask": chosen_mask,   # 0=prompt, 1=response
                "rejected_mask": rejected_mask,
            })

    def __getitem__(self, index):
        return self.encoded[index]

    def __len__(self):
        return len(self.encoded)

    @torch.no_grad()
    def attach_ref_logps(self, ref_model, collator, batch_size, device):
        from torch.utils.data import DataLoader
        from dpo_trainer import _response_log_probs

        loader = DataLoader(self, batch_size=batch_size, shuffle=False, collate_fn=collator)

        ref_model.to(device)
        ref_model.eval()

        all_chosen_logps = []
        all_rejected_logps = []

        for batch in loader:
            chosen = batch["chosen"].to(device)
            rejected = batch["rejected"].to(device)
            chosen_mask = batch["chosen_mask"].to(device)
            rejected_mask = batch["rejected_mask"].to(device)
            chosen_attn = batch["chosen_attention_mask"].to(device)
            rejected_attn = batch["rejected_attention_mask"].to(device)

            chosen_logp = _response_log_probs(ref_model, chosen, chosen_attn, chosen_mask)
            rejected_logp = _response_log_probs(ref_model, rejected, rejected_attn, rejected_mask)

            all_chosen_logps.append(chosen_logp.cpu())
            all_rejected_logps.append(rejected_logp.cpu())

        all_chosen_logps = torch.cat(all_chosen_logps)
        all_rejected_logps = torch.cat(all_rejected_logps)

        for i, entry in enumerate(self.encoded):
            entry["ref_chosen_logps"] = all_chosen_logps[i].item()
            entry["ref_rejected_logps"] = all_rejected_logps[i].item()