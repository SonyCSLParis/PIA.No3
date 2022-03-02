import torch
import torch.nn as nn
from einops.einops import rearrange

from CIA.model.attentions.subsampled_relative_attention import (
    SubsampledRelativeAttention,
)


class ScalingAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        seq_len_src,
        seq_len_tgt,
        dropout,
        norm_out,
        relative_pos_bias,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert (seq_len_src % seq_len_tgt == 0) or (seq_len_tgt % seq_len_src == 0)
        self.seq_len_src = seq_len_src
        self.seq_len_tgt = seq_len_tgt

        # Bias?!
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm_out = nn.LayerNorm(dim) if norm_out else None

        if relative_pos_bias:
            self.attn_bias = SubsampledRelativeAttention(
                head_dim=self.head_dim,
                num_heads=num_heads,
                seq_len_src=seq_len_src,
                seq_len_tgt=seq_len_tgt,
            )
        else:
            self.attn_bias = None

    def forward(self, q, kv):
        """ """
        _, num_tok_q, dim_q = q.shape
        _, num_tok_kv, dim_k = kv.shape
        assert (
            (num_tok_q == self.seq_len_tgt)
            and (num_tok_kv == self.seq_len_src)
            and (dim_k == self.dim)
            and (dim_q == self.dim)
        )
        if num_tok_q > num_tok_kv:
            assert num_tok_q % num_tok_kv == 0
            upscaled_dim = 0
            downscaling = num_tok_q // num_tok_kv
            num_token_downscaled = num_tok_kv
        elif num_tok_kv > num_tok_q:
            assert num_tok_kv % num_tok_q == 0
            upscaled_dim = 1
            downscaling = num_tok_kv // num_tok_q
            num_token_downscaled = num_tok_q
        else:
            downscaling = 1
            num_token_downscaled = num_tok_q
        q = self.to_q(q)
        k = self.to_k(kv)
        v = self.to_v(kv)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), (q, k, v)
        )
        qk = torch.einsum("bhid,bhjd->bhij", q, k) * (self.head_dim ** -0.5)
        # NOTES: we can set diagonal to 1 for every cases since latents are time-shifted before being read back, so no info leak
        causal_mask = torch.triu(
            -float("inf") * torch.ones(num_token_downscaled, num_token_downscaled),
            diagonal=1,
        ).to(qk.device)
        if downscaling > 1:
            causal_mask = causal_mask.repeat_interleave(downscaling, dim=upscaled_dim)
        qk_masked = causal_mask[None, None, :, :] + qk

        # relative attention
        if self.attn_bias is not None:
            attn_bias = self.attn_bias(q)
            qk_masked += attn_bias

        attn = qk_masked.softmax(dim=-1)
        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        if self.norm_out:
            out = self.norm_out(out)
        out = self.to_out(out)
        out = self.dropout(out)
        return out
