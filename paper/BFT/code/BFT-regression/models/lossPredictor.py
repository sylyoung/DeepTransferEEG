import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class LossPredictor(nn.Module):
    def __init__(self, inputdim):
        super().__init__()
        self.inputdim = inputdim
        self.losspre_block = nn.Sequential(
            nn.Linear(in_features=self.inputdim,
                      out_features=self.inputdim // 2,
                      bias=True),
            nn.ELU(),
            nn.Linear(in_features=self.inputdim // 2,
                    out_features=self.inputdim // 4,
                    bias=True),  
            nn.ELU(),      
            nn.Linear(in_features=self.inputdim // 4,
                    out_features=1,
                    bias=True))

    def forward(self, x):
        output = self.losspre_block(x)
        return output
    

def compute_real_losses(augmented_inputs, model_target):
    loss_fn = nn.MSELoss()
    real_losses = []
    model_target.eval()
    with torch.no_grad():
        for x_aug, label in augmented_inputs:
            label = label.unsqueeze(0)
            pred = model_target(x_aug)
            loss = loss_fn(pred, label)
            real_losses.append(loss.item())
    return torch.tensor(real_losses)  # dim = 12