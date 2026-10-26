import math
import tempfile

import torch

from permutect import constants


class Checkpoint:
    def __init__(self, device, model: torch.nn.Module, optimizer: torch.optim.Optimizer):
        self.checkpoint_file = tempfile.NamedTemporaryFile(suffix=".pt")
        self.best_checkpoint = None
        self.device = device
        self.model = model
        self.optimizer = optimizer

    def save_checkpoint_if_needed(self, epoch, loss: float):
        # a NaN loss is never the best: once recorded, no later loss compares as smaller and no checkpoint is ever saved
        if math.isnan(loss):
            return
        if self.best_checkpoint is None or loss < self.best_checkpoint["loss"]:
            print(f"New best mean loss: {loss:.1f}, saving new checkpoint.")
            save_data = {
                constants.STATE_DICT_NAME: self.model.state_dict(),
                constants.OPTIMIZER_STATE_DICT_NAME: self.optimizer.state_dict(),
            }
            torch.save(save_data, self.checkpoint_file.name)
            self.best_checkpoint = {"epoch": epoch, "loss": loss}

    def load_checkpoint_if_needed(self, loss: float):
        if self.best_checkpoint is not None and (math.isnan(loss) or loss > 2 * self.best_checkpoint["loss"]):
            print(f"Anomalously large mean loss: {loss:.1f}, loading checkpoint.")
            saved = torch.load(self.checkpoint_file.name, map_location=self.device)
            self.model.load_state_dict(saved[constants.STATE_DICT_NAME])
            self.optimizer.load_state_dict(saved[constants.OPTIMIZER_STATE_DICT_NAME])
