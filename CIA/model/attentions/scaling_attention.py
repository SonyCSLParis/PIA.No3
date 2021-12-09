import torch
import torch.nn as nn
from einops.einops import rearrange


class ScalingAttention(nn.Module):
    def __init__(self, dim, num_heads, downscaling, dropout, norm_out):
        super().__init__()
        self.downscaling = downscaling
        self.num_heads = num_heads
        # Bias?!
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm_out = nn.LayerNorm(dim) if norm_out else None

    def forward(self, q, kv):
        """ """
        _, num_tok_q, dim = q.shape
        _, num_tok_kv, dim = kv.shape
        assert dim % self.num_heads == 0
        dim = dim // self.num_heads
        if num_tok_q > num_tok_kv:
            assert num_tok_q == num_tok_kv * self.downscaling
            upscaled_dim = 0
            num_token_downscaled = num_tok_kv
        elif num_tok_kv > num_tok_q:
            assert num_tok_kv == num_tok_q * self.downscaling
            upscaled_dim = 1
            num_token_downscaled = num_tok_q
        else:
            num_token_downscaled = num_tok_q
        q = self.to_q(q)
        k = self.to_k(kv)
        v = self.to_v(kv)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), (q, k, v)
        )
        qk = torch.einsum("bhid,bhjd->bhij", q, k) * (dim ** -0.5)
        causal_mask = torch.triu(
            -float("inf") * torch.ones(num_token_downscaled, num_token_downscaled),
            diagonal=1,
        ).to(qk.device)
        if self.downscaling > 1:
            causal_mask = causal_mask.repeat_interleave(
                self.downscaling, dim=upscaled_dim
            )
        qk_masked = causal_mask[None, None, :, :] + qk
        attn = qk_masked.softmax(dim=-1)
        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        if self.norm_out:
            out = self.norm_out(out)
        out = self.to_out(out)
        out = self.dropout(out)
        return out
