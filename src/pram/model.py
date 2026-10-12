"""The notebook's main model: a fixed 0.3 gate and sparsemax memory."""

import torch
from torch import nn
from entmax import sparsemax


class RawRejectMemory(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, head_num=1):
        super().__init__()
        if hidden_dim % head_num:
            raise ValueError("hidden_dim must be divisible by head_num")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.head_dim = hidden_dim // head_num
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.reject_proj = nn.Linear(input_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        # Unused in forward, retained to preserve notebook initialization order.
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 3, 1), nn.Sigmoid())
        self.clf = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x, rej_tensor):
        batch, count = x.size(0), rej_tensor.size(0)
        if count == 0:
            raise ValueError("The reject memory cannot be empty")
        q = self.input_proj(x)
        r = self.reject_proj(rej_tensor)
        qh = q.view(batch, self.head_num, self.head_dim)
        k = self.W_k(r).view(count, self.head_num, self.head_dim).transpose(0, 1)
        v = self.W_v(r).view(count, self.head_num, self.head_dim).transpose(0, 1)
        scores = torch.einsum("bhd,hnd->bhn", qh, k) / self.head_dim**0.5
        attention = sparsemax(scores, dim=-1)
        memory = torch.einsum("bhn,hnd->bhd", attention, v).reshape(batch, self.hidden_dim)
        fused = 0.7 * q + 0.3 * memory
        return self.clf(fused).squeeze(1), attention, scores.max(dim=-1).values

