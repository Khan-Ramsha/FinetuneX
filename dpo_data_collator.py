import torch

class DataCollatorForDPO:
    def __init__(self, pad_token_id, pad_to_multiple_of=8):
        self.pad_token_id = pad_token_id
        self.pad_to_multiple = pad_to_multiple_of

    def __call__(self, batch):
        chosen_ids    = [torch.tensor(e["chosen"],dtype=torch.long) for e in batch]
        rejected_ids  = [torch.tensor(e["rejected"],dtype=torch.long) for e in batch]
        chosen_masks  = [torch.tensor(e["chosen_mask"], dtype=torch.long) for e in batch]
        rejected_masks= [torch.tensor(e["rejected_mask"],dtype=torch.long) for e in batch]

        max_len = max(
            max(t.size(0) for t in chosen_ids),
            max(t.size(0) for t in rejected_ids),
        )

        if self.pad_to_multiple is not None:
            remainder = max_len % self.pad_to_multiple
            if remainder != 0:
                max_len += self.pad_to_multiple - remainder

        def pad_seq(seqs, pad_val):
            out = torch.full((len(seqs), max_len), pad_val, dtype=torch.long)
            for i, s in enumerate(seqs):
                out[i, :s.size(0)] = s
            return out

        def make_attn_mask(seqs):
            mask = torch.zeros(len(seqs), max_len, dtype=torch.long)
            for i, s in enumerate(seqs):
                mask[i, :s.size(0)] = 1
            return mask

        result = {
            "chosen": pad_seq(chosen_ids, self.pad_token_id),
            "rejected": pad_seq(rejected_ids,  self.pad_token_id),
            "chosen_mask": pad_seq(chosen_masks, 0),
            "rejected_mask":  pad_seq(rejected_masks, 0),
            "chosen_attention_mask": make_attn_mask(chosen_ids),    
            "rejected_attention_mask": make_attn_mask(rejected_ids),
        }
        if "ref_chosen_logps" in batch[0]:
            result["ref_chosen_logps"] = torch.tensor(
                [e["ref_chosen_logps"] for e in batch], dtype=torch.float32
            )
            result["ref_rejected_logps"] = torch.tensor(
                [e["ref_rejected_logps"] for e in batch], dtype=torch.float32
            )

        return result