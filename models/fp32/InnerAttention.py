import torch
import torch.nn as nn
import math


class ScaledDotProductAttention(nn.Module):
    def __init__(self, nhead: int, d_model: int):
        super().__init__()

        self.scale = 1.0 / math.sqrt(d_model // nhead)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, q: torch.FloatTensor, k: torch.FloatTensor, v: torch.FloatTensor):

        scores = torch.einsum("blhe,bshe->bhls", q, k)
        attn_weights = self.dropout(torch.softmax(self.scale * scores, dim=-1))
        context = torch.einsum("bhls,bshd->blhd", attn_weights, v)

        return context.contiguous(), attn_weights
