import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()

        self.fc1 = nn.Linear(d_model, d_model * 4)
        self.fc2 = nn.Linear(d_model * 4, d_model)

    def forward(self, inputs: torch.FloatTensor):
        outputs = self.fc1(inputs)
        outputs = F.relu(outputs)
        outputs = self.fc2(outputs)
        return outputs
