"""GPT-style pre-norm Transformer with RoPE and whole-stack weight sharing."""
import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(),
                                 nn.Linear(4 * width, width))
        inv = 1 / (10000 ** (torch.arange(0, width // heads, 2).float() / (width // heads)))
        self.register_buffer("inv_freq", inv, persistent=False)

    def rotate(self, x):
        positions = torch.arange(x.shape[-2], device=x.device).float()
        phase = torch.outer(positions, self.inv_freq.float())
        cos, sin = phase.cos().to(x.dtype), phase.sin().to(x.dtype)
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1).flatten(-2)

    def forward(self, x):
        b, s, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(b, s, 3, self.heads, d // self.heads).unbind(2)
        q, k, v = [z.transpose(1, 2) for z in (q, k, v)]
        h = F.scaled_dot_product_attention(self.rotate(q), self.rotate(k), v,
                                           dropout_p=0.0, is_causal=True)
        x = x + self.proj(h.transpose(1, 2).reshape(b, s, d))
        return x + self.mlp(self.norm2(x))


class Model(nn.Module):
    def __init__(self, layers=3, loops=1, width=256, heads=4, vocab=40):
        super().__init__()
        self.loops = loops
        self.embedding = nn.Embedding(vocab, width)
        self.blocks = nn.ModuleList([Block(width, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.apply(self.initialize)

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens, positions):
        x = self.embedding(tokens)
        for _ in range(self.loops):
            for block in self.blocks:
                x = block(x)
        x = self.norm(x[torch.arange(x.shape[0], device=x.device), positions])
        return F.linear(x, self.embedding.weight)
