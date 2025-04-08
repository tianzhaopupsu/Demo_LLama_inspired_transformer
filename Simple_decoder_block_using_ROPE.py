import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def apply_rope(x: torch.Tensor, base_theta=10000.0) -> torch.Tensor:

    seq_len, _, dim = x.shape
    assert dim % 2 == 0
    dim_half = dim // 2

    idx = torch.arange(0, dim_half, device=x.device)
    inv_freq = 1.0 / (base_theta ** (idx / dim_half))
    positions = torch.arange(seq_len, device=x.device).type_as(inv_freq)

    sinusoid = torch.einsum("i,j->ij", positions, inv_freq)
    sin = torch.sin(sinusoid).repeat_interleave(2, dim=-1)
    cos = torch.cos(sinusoid).repeat_interleave(2, dim=-1)

    x1 = x[..., ::2]
    x2 = x[..., 1::2]

    x_rope = torch.stack([
        x1 * cos[:, None, ::2] - x2 * sin[:, None, ::2],
        x1 * sin[:, None, ::2] + x2 * cos[:, None, ::2]
    ], dim=-1)

    return x_rope.flatten(-2)

class ToyRoPEDecoderLayer(nn.Module):
    def __init__(self, dim, num_heads, ff_mult=4, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        self.ff = nn.Sequential(
            nn.Linear(dim, ff_mult * dim),
            nn.ReLU(),
            nn.Linear(ff_mult * dim, dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):

        b, t, d = x.shape

        # qkv: (batch, seq_len, 3*dim)
        qkv = self.qkv_proj(x)
        qkv = qkv.view(b, t, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (batch, heads, seq_len, head_dim)

        # Apply RoPE to q and k
        q = q.permute(2, 0, 1, 3).reshape(t, b * self.num_heads, self.head_dim)
        k = k.permute(2, 0, 1, 3).reshape(t, b * self.num_heads, self.head_dim)
        q = apply_rope(q)
        k = apply_rope(k)
        q = q.view(t, b, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        k = k.view(t, b, self.num_heads, self.head_dim).permute(1, 2, 0, 3)

        # Attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_scores = attn_scores.masked_fill(torch.triu(torch.ones(t, t, device=x.device), 1).bool(), float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)  # (batch, heads, seq_len, head_dim)

        # Merge heads
        attn_output = attn_output.transpose(1, 2).reshape(b, t, d)
        x = x + self.dropout(self.out_proj(attn_output))
        x = self.norm1(x)

        # Feedforward
        x = x + self.ff(x)
        x = self.norm2(x)
        print(x)
        return x
if __name__ == "__main__":
    layer = ToyRoPEDecoderLayer(dim=32, num_heads=4)
    x = torch.randn(2, 10, 32)  # (batch, seq_len, dim)
    y = layer(x)
    print("Output shape:", y.shape)  # should be (2, 10, 32)
