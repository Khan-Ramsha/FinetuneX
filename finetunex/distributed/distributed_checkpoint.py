import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful

class AppState(Stateful):
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def state_dict(self):
        model_sd, optim_sd = get_state_dict(self.model, self.optimizer)
        return {"model": model_sd, "optim": optim_sd}

    def load_state_dict(self, state_dict):
        set_state_dict(
            self.model, self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"]
        )