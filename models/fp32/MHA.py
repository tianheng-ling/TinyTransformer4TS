import torch
import torch.nn as nn
from .InnerAttention import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int):

        super().__init__()
        self.nhead = nhead
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.inner_attn = ScaledDotProductAttention(d_model=d_model, nhead=self.nhead)
        self.output_layer = nn.Linear(d_model, d_model)

    def forward(self, q: torch.FloatTensor, k: torch.FloatTensor, v: torch.FloatTensor):
        B = q.shape[0]
        L = q.shape[1]
        H = self.nhead

        # linear projection and split heads
        q = self.linear_q(q).view(B, L, H, -1)  # B,L,E -> B,L,H,E/H
        k = self.linear_k(k).view(B, L, H, -1)
        v = self.linear_v(v).view(B, L, H, -1)

        # execute scaled dot product attention
        context, attn_weights = self.inner_attn(q, k, v)

        # concat heads
        context = context.view(B, L, -1)  # B,L,H,E -> B,L,D

        outputs = self.output_layer(context)

        return outputs, attn_weights
