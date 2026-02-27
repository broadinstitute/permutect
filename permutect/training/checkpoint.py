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

    def save_checkpoint_if_needed(self, epoch, loss: torch.Tensor):
        # loss is a length-1 tensor
        if self.best_checkpoint is None or (loss.item() < self.best_checkpoint['loss'] and not loss.isnan().item()):
            print(f"New best mean loss: {loss.item():.1f}, saving new checkpoint.")
            save_data = {constants.STATE_DICT_NAME: self.model.state_dict(),
                         constants.OPTIMIZER_STATE_DICT_NAME: self.optimizer.state_dict()}
            torch.save(save_data, self.checkpoint_file.name)
            self.best_checkpoint = {'epoch': epoch, 'loss': loss.item()}

    def load_checkpoint_if_needed(self, loss: torch.Tensor):
        if self.best_checkpoint is not None and (loss.isnan().item() or loss.item() > 2 * self.best_checkpoint['loss']):
            print(f"Anomalously large mean loss: {loss.item():.1f}, loading checkpoint.")
            saved = torch.load(self.checkpoint_file.name, map_location=self.device)
            self.model.load_state_dict(saved[constants.STATE_DICT_NAME])
            self.optimizer.load_state_dict(saved[constants.OPTIMIZER_STATE_DICT_NAME])