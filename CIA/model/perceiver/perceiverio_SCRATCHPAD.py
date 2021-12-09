from performer_pytorch.performer_pytorch import SelfAttention
from CIA.model.causal_events_model_full_cat import GEGLU, GeneralSWIGLU
import torch.nn as nn
import torch.nn.functional as F
import torch
from einops import rearrange
import math


class Encode(nn.Module):
    def __init__(self, dim, num_heads, downscaling, hidden_dim, dropout):
        super().__init__()
        self.dim = dim
        self.downscaling = downscaling

        assert dim % num_heads == 0
        self.dim_heads = dim // num_heads
        # Bias?!
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)

        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, l):
        """Put x into l

        Args:
            x (M, C): [description]
            l (N, ): [description]
        """

        dim = self.dim_heads
        num_latents = l.size(1)
        num_tokens = x.size(1)

        assert num_tokens == num_latents * self.downscaling

        q = self.to_q(l)
        k = self.to_k(x)
        v = self.to_v(x)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), (q, k, v)
        )

        qk = torch.einsum("bhid,bhjd->bhij", q, k) * (dim ** -0.5)
        causal_mask = torch.triu(
            -float("inf") * torch.ones(num_latents, num_latents), diagonal=1
        ).to(qk.device)

        causal_mask = causal_mask.repeat_interleave(self.downscaling, dim=1)
        qv_masked = causal_mask[None, None, :, :] + qk

        attn = qv_masked.softmax(dim=-1)

        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return self.dropout(out)


class Process(nn.Module):
    def __init__(self, dim, num_heads, hidden_dim, dropout):
        super().__init__()
        self.self_attn = Encode(
            dim=dim,
            num_heads=num_heads,
            downscaling=1,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        # self.norm = lambda x: x
        # self.norm2 = lambda x: x

        # self.mlp = SWIGLU(dim, dropout) ? Was this used?
        self.mlp = GEGLU(dim, hidden_dim=hidden_dim, output_dim=dim, dropout=dropout)

        self.rezero_1 = nn.Parameter(torch.zeros(1))
        self.rezero_2 = nn.Parameter(torch.zeros(1))

    def forward(self, l):
        out = l
        out = self.norm(out)
        out = self.self_attn(out, out)
        if self.rezero_1 is not None:
            out = self.rezero_1 * out
        l = l + out

        out = self.norm2(l)
        out = self.mlp(out)
        if self.rezero_2 is not None:
            out = self.rezero_2 * out
        out = l + out
        return out


class Remixer(nn.Module):
    def __init__(self, dim, length, dropout):
        super().__init__()
        self.w1 = nn.Linear(dim, dim * 4 * 2)
        self.w2 = nn.Linear(dim * 4, dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        self.H = torch.nn.Parameter(torch.randn(length, length))
        self.causal_mask = torch.nn.Parameter(
            torch.triu(-float("inf") * torch.ones(length, length), diagonal=1),
            requires_grad=False,
        )
        self.alpha = torch.nn.Parameter(torch.randn(1))
        self.w3 = nn.Linear(dim, dim)

    def forward(self, x):
        x, v = self.w1(x).chunk(2, dim=-1)
        x = self.gelu(x) * v
        x = self.dropout(x)
        x_prime = self.w2(x)

        H = torch.softmax(self.H + self.causal_mask, dim=-1)

        x_s_l = torch.einsum("bnd,mn->bmd", x_prime, H)
        alpha = torch.sigmoid(self.alpha)

        x_c_l = alpha * x_s_l * x_prime + (1 - alpha) * (x_prime - x_s_l)

        y = x_prime + self.w3(x_c_l)
        return y


class ProcessRemixer(nn.Module):
    def __init__(self, dim, num_heads, hidden_dim, length, dropout):
        super().__init__()
        self.self_attn = Encode(
            dim=dim,
            num_heads=num_heads,
            downscaling=1,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.remixer = Remixer(dim, length, dropout)

        self.rezero_1 = nn.Parameter(torch.zeros(1))
        self.rezero_2 = nn.Parameter(torch.zeros(1))

    def forward(self, l):
        out = l
        out = self.norm(out)
        out = self.self_attn(out, out)
        if self.rezero_1 is not None:
            out = self.rezero_1 * out
        l = l + out

        out = self.norm2(l)
        out = self.remixer(out)
        if self.rezero_2 is not None:
            out = self.rezero_2 * out
        out = l + out
        return out


class ProcessGW(nn.Module):
    def __init__(self, dim, num_heads, hidden_dim, length, dropout):
        super().__init__()
        self.self_attn = Encode(
            dim=dim,
            num_heads=num_heads,
            downscaling=1,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.remixer = Remixer(dim, length, dropout)

        self.rezero_1 = nn.Parameter(torch.zeros(1))
        self.rezero_2 = nn.Parameter(torch.zeros(1))

    def forward(self, x, l):
        out = l
        out = self.norm(out)
        out = self.self_attn(out, out)
        if self.rezero_1 is not None:
            out = self.rezero_1 * out
        l = l + out

        out = self.norm2(l)
        out = self.remixer(out)
        if self.rezero_2 is not None:
            out = self.rezero_2 * out
        out = l + out
        return out


class CrossAttentionDecoder(nn.Module):
    def __init__(self, dim, num_heads, downscaling, hidden_dim, dropout):
        super().__init__()
        self.dim = dim
        self.downscaling = downscaling

        assert dim % num_heads == 0
        self.dim_heads = dim // num_heads
        # Bias?!
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)

        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, l):
        """[summary]

        Args:
            x (batch_size * num_latents, downscaling, d):
            l (batch_size * num_latents, num_latents, d ):
        """

        dim = self.dim_heads
        num_latents = l.size(1)
        num_tokens = x.size(1)
        true_batch_size = l.size(0) // num_latents

        q = self.to_q(x)
        k = self.to_k(l)
        v = self.to_v(l)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), (q, k, v)
        )

        qk = torch.einsum("bhid,bhjd->bhij", q, k) * (dim ** -0.5)
        causal_mask = torch.triu(
            -float("inf") * torch.ones(num_latents, num_latents), diagonal=1
        ).to(qk.device)

        causal_mask = causal_mask.repeat(true_batch_size, 1)
        qv_masked = causal_mask[:, None, None, :] + qk

        attn = qv_masked.softmax(dim=-1)

        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return self.dropout(out)


class SWIGLU(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.w1 = nn.Linear(dim, dim * 4 * 2)
        self.w2 = nn.Linear(dim * 4, dim)
        self.silu = lambda x: x * torch.sigmoid(x)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x, v = self.w1(x).chunk(2, dim=-1)
        x = self.silu(x) * v
        x = self.dropout(x)
        x = self.w2(x)
        return x


class DecodeConcat(nn.Module):
    def __init__(self, dim, num_heads, hidden_dim, downscaling, dropout):
        super().__init__()
        self.downscaling = downscaling
        self.self_attn = Encode(
            dim=dim,
            num_heads=num_heads,
            downscaling=1,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.norm = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)

        self.mlp = SWIGLU(dim, dropout)

        self.dropout = nn.Dropout(dropout)
        self.rezero_1 = nn.Parameter(torch.zeros(1))
        self.rezero_2 = nn.Parameter(torch.zeros(1))
        self.rezero_3 = nn.Parameter(torch.zeros(1))

    def forward(self, x, l):
        out = self.cross_attn(self.norm3(x), self.norm4(l))
        if self.rezero_1:
            out = self.rezero_1 * out
        x = x + out

        out = self.norm(x)
        out = self.self_attn(out, out)
        if self.rezero_2:
            out = self.rezero_2 * out
        x = x + out

        out = self.norm2(x)
        out = self.mlp(out)
        if self.rezero_3:
            out = self.rezero_3 * out
        out = x + out
        return out


# PerceiverIO Remixer
# class PerceiverIO(nn.Module):
#     def __init__(self, dim, num_layers, dropout, **kwargs):
#         super().__init__()
#         self.dim_last_layer = dim
#         self.downscaling = 16

#         self.l_init = nn.Parameter(torch.randn(1, 64, 512))
#         self.encode = Encode(dim=512,
#                              num_heads=8,
#                              downscaling=self.downscaling,
#                              hidden_dim=512,
#                              dropout=dropout)
#         self.process = nn.Sequential(*[
#             ProcessRemixer(dim, num_heads=8, hidden_dim=512,
#                            length=64,
#                            dropout=dropout)
#             for _ in range(num_layers)
#         ])

#         self.process_query = nn.Sequential(*[
#             ProcessRemixer(dim, num_heads=8,
#                            hidden_dim=512,
#                            length=self.downscaling,
#                            dropout=dropout)
#             for _ in range(10)
#         ])

#         # Small causal encoder transformer
#         self.decode = CrossAttentionDecoder(dim=dim,
#                                             num_heads=8,
#                                             downscaling=self.downscaling,
#                                             hidden_dim=None,
#                                             dropout=dropout)

#         self.dummy_latent = nn.Parameter(torch.randn(1, 1, 512))

#         # self.last_layer_norm = nn.LayerNorm(dim)
#         self.last_layer_norm = None

#     def forward(self, x, **kwargs):
#         batch_size, num_tokens, feature_dim = x.size()
#         _, num_latents, latent_dim = self.l_init.size()

#         # intialise the tower of latents
#         l = self.l_init.expand(batch_size, num_latents, latent_dim)
#         l = self.encode(x, l)

#         # offset on l
#         l = torch.cat([self.dummy_latent.repeat(batch_size, 1, 1), l],
#                       dim=1)[:, :-1]

#         l = self.process(l)

#         x = x.reshape(batch_size * num_tokens // self.downscaling,
#                       self.downscaling, feature_dim)
#         x = self.process_query(x)
#         l = l.repeat_interleave(num_tokens // self.downscaling, dim=0)

#         y = self.decode(x, l)

#         y = y.reshape(batch_size, num_tokens, latent_dim)

#         if self.last_layer_norm is not None:
#             y = self.last_layer_norm(y)
#         return dict(x=y)


# Not a PerceiverIO, just Globalworkspace
# Gros rouge 17:22:49 6 couches
# class PerceiverIO(nn.Module):
#     def __init__(self, dim, num_layers, dropout, **kwargs):
#         super().__init__()
#         self.dim_last_layer = dim
#         self.downscaling = 1

#         # which init?
#         self.l_init = torch.zeros(1024, 512)
#         position = torch.arange(0, 1024, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, 512, 2).float() * (
#             -math.log(10000.0) / 512))
#         self.l_init[:, 0::2] = torch.sin(position * div_term)
#         self.l_init[:, 1::2] = torch.cos(position * div_term)
#         self.l_init = nn.Parameter(self.l_init.unsqueeze(0), requires_grad=False)

#         # self.l_init = nn.Parameter(torch.randn(1, 1024, 512))

#         self.write = nn.ModuleList([
#             QKV_Write(dim, num_heads=8, downscaling=self.downscaling,
#                       hidden_dim=dim, dropout=dropout)
#             for _ in range(num_layers)
#         ])
#         self.read = nn.ModuleList([
#             QKV_Read(dim, num_heads=8, downscaling=self.downscaling,
#                       hidden_dim=dim, dropout=dropout)
#             for _ in range(num_layers)
#         ])

#         self.ffn = nn.ModuleList([
#             GEGLU(dim, hidden_dim=2 * dim,
#                   output_dim=dim,
#                   dropout=dropout)
#             for _ in range(num_layers)
#         ])

#         self.norm1 = nn.ModuleList([
#             nn.LayerNorm(dim)
#             for _ in range(num_layers)
#         ])
#         self.norm2 = nn.ModuleList([
#             nn.LayerNorm(dim)
#             for _ in range(num_layers)
#         ])
#         # self.norm2 = [
#         #     None
#         #     for _ in range(num_layers)
#         # ]
#         self.norm3 = nn.ModuleList([
#             nn.LayerNorm(dim)
#             for _ in range(num_layers)
#         ])

#         # self.last_layer_norm = nn.LayerNorm(dim)
#         self.last_layer_norm = None

#     def forward(self, x, **kwargs):
#         batch_size, num_tokens, feature_dim = x.size()

#         # intialise the memory
#         _, num_latents, latent_dim = self.l_init.size()
#         l = self.l_init.expand(batch_size, num_latents, latent_dim)

#         for write, read, process, norm1, norm2, norm3 in zip(self.write,
#                                         self.read,
#                                         self.ffn,
#                                         self.norm1,
#                                         self.norm2,
#                                         self.norm3
#                                         ):
#             # residual connections?
#             x_norm = norm1(x)
#             # l_norm = norm2(l)

#             # always start with new memories
#             l_norm = self.l_init.expand(batch_size, num_latents, latent_dim)
#             l = l + write(x_norm, l_norm)
#             x = x + read(x_norm, l)
#             x = x + process(norm3(x))

#         if self.last_layer_norm is not None:
#             x = self.last_layer_norm(x)
#         return dict(x=x)


# piano_event_performer_2021-09-30_17:52:57
# nouveau petit bleu 16 couches!

# class PerceiverIO(nn.Module):
#     def __init__(self, dim, num_layers, dropout, **kwargs):
#         super().__init__()
#         self.dim_last_layer = dim
#         self.downscaling = 16

#         # which init?
#         self.l_init = torch.zeros(1024 // self.downscaling, 512)
#         position = torch.arange(0, 1024 // self.downscaling,
#                                 dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(
#             torch.arange(0, 512, 2).float() * (-math.log(10000.0) / 512))
#         self.l_init[:, 0::2] = torch.sin(position * div_term)
#         self.l_init[:, 1::2] = torch.cos(position * div_term)
#         self.l_init = nn.Parameter(self.l_init.unsqueeze(0),
#                                    requires_grad=False)

#         self.dummy_l = nn.ParameterList([nn.Parameter(torch.randn(1, 1, 512))
#                                      for _ in range(num_layers)])

#         # self.l_init = nn.Parameter(torch.randn(1, 1024, 512))

#         self.write = nn.ModuleList([
#             QKV_Write(dim,
#                       num_heads=8,
#                       downscaling=self.downscaling,
#                       hidden_dim=dim,
#                       dropout=dropout) for _ in range(num_layers)
#         ])
#         self.read = nn.ModuleList([
#             QKV_Read(dim,
#                      num_heads=8,
#                      downscaling=self.downscaling,
#                      hidden_dim=dim,
#                      dropout=dropout) for _ in range(num_layers)
#         ])

#         self.ffn = nn.ModuleList([
#             GEGLU(dim, hidden_dim=2 * dim, output_dim=dim, dropout=dropout)
#             for _ in range(num_layers)
#         ])

#         self.self_local_attention = nn.ModuleList([
#             SelfAttention(dim=dim,
#                           local_heads=8,
#                           heads=8,
#                           causal=True,
#                           local_window_size=64,
#                           dropout=dropout) for _ in range(num_layers)
#         ])

#         self.norm1 = nn.ModuleList(
#             [nn.LayerNorm(dim) for _ in range(num_layers)])
#         self.norm2 = nn.ModuleList(
#             [nn.LayerNorm(dim) for _ in range(num_layers)])

#         self.norm3 = nn.ModuleList(
#             [nn.LayerNorm(dim) for _ in range(num_layers)])

#         self.last_layer_norm = nn.LayerNorm(dim)

#     def forward(self, x, **kwargs):
#         batch_size, num_tokens, feature_dim = x.size()

#         # intialise the memory
#         _, num_latents, latent_dim = self.l_init.size()
#         l = self.l_init.expand(batch_size, num_latents, latent_dim)

#         for write, read, process, atn, norm1, norm2, norm3, dummy_l in zip(
#                 self.write, self.read, self.ffn, self.self_local_attention,
#                 self.norm1, self.norm2, self.norm3, self.dummy_l):
#             # residual connections?
#             # l_norm = norm2(l)

#             x = x + atn(norm1(x))

#             # always start with new memories
#             l_norm = self.l_init.expand(batch_size, num_latents, latent_dim)
#             x_norm = norm2(x)
#             l = l + write(x_norm, l_norm)

#             # we must shift by one
#             dummy_l = dummy_l.repeat(batch_size, 1, 1)
#             l = torch.cat([dummy_l, l[:, :-1]], dim=1)
#             x = x + read(x_norm, l)

#             x = x + process(norm3(x))

#         if self.last_layer_norm is not None:
#             x = self.last_layer_norm(x)
#         return dict(x=x)


# Tried with concatenating l_i, encoder-decoder on all ls

# class PerceiverIO(nn.Module):
#     def __init__(self, dim, num_layers, dropout, **kwargs):
#         super().__init__()
#         self.dim_last_layer = dim
#         self.downscaling = 16

#         self.l_init = nn.Parameter(torch.randn(1, 64, 512))
#         self.encode = Encode(dim=512,
#                              num_heads=8,
#                              downscaling=self.downscaling,
#                              hidden_dim=512,
#                              dropout=dropout)
#         self.process = nn.Sequential(*[
#             Process(dim, num_heads=8, hidden_dim=512, dropout=dropout)
#             for _ in range(num_layers)
#         ])

#         # Small encoder-decoder transformer
#         self.decode = nn.Sequential(*[
#             Process(dim, num_heads=8, hidden_dim=512, dropout=dropout)
#             for _ in range(8)
#         ])

#         self.dummy_latent = nn.Parameter(torch.randn(1, 1, 512))

#         self.last_layer_norm = nn.LayerNorm(dim)

#     def forward(self, x, **kwargs):
#         batch_size, num_tokens, feature_dim = x.size()
#         _, num_latents, latent_dim = self.l_init.size()

#         # intialise the tower of latents
#         l = self.l_init.expand(batch_size, num_latents, latent_dim)
#         l = self.encode(x, l)

#         l = self.process(l)

#         # offset on l
#         l = torch.cat([self.dummy_latent.repeat(batch_size, 1, 1), l],
#                       dim=1)[:, :-1]

#         l = l.reshape(batch_size * num_latents, 1, feature_dim)
#         x = x.reshape(batch_size * num_tokens // self.downscaling,
#                       self.downscaling, feature_dim)

#         y = torch.cat([l, x], dim=1)
#         y = self.decode(y)
#         y = y[:, :-1]
#         y = y.reshape(batch_size, num_tokens, latent_dim)

#         if self.last_layer_norm is not None:
#             y = self.last_layer_norm(y)
#         return dict(x=y)

# Try with processings
# 14 couches piano_event_performer_2021-10-01_16:03:06


# class PerceiverIO(nn.Module):
#     def __init__(self, dim, num_layers, dropout, **kwargs):
#         super().__init__()
#         self.dim_last_layer = dim
#         self.downscaling = 16

#         self.l_init = nn.Parameter(torch.randn(1, 64, 512))
#         self.encode = Encode(dim=512,
#                              num_heads=8,
#                              downscaling=self.downscaling,
#                              hidden_dim=512,
#                              dropout=dropout)
#         self.process = nn.Sequential(*[
#             Process(dim, num_heads=8, hidden_dim=512, dropout=dropout)
#             for _ in range(num_layers)
#         ])

#         # Small encoder-decoder transformer
#         self.decode = nn.Sequential(*[
#             Process(dim, num_heads=8, hidden_dim=512, dropout=dropout)
#             for _ in range(8)
#         ])

#         self.dummy_latent = nn.Parameter(torch.randn(1, 1, 512))

#         self.last_layer_norm = nn.LayerNorm(dim)

#     def forward(self, x, **kwargs):
#         batch_size, num_tokens, feature_dim = x.size()
#         _, num_latents, latent_dim = self.l_init.size()

#         # intialise the tower of latents
#         l = self.l_init.expand(batch_size, num_latents, latent_dim)
#         l = self.encode(x, l)

#         l = self.process(l)

#         # offset on l
#         l = torch.cat([self.dummy_latent.repeat(batch_size, 1, 1), l],
#                       dim=1)[:, :-1]

#         l = l.reshape(batch_size * num_latents, 1, feature_dim)
#         x = x.reshape(batch_size * num_tokens // self.downscaling,
#                       self.downscaling, feature_dim)

#         y = torch.cat([l, x], dim=1)
#         y = self.decode(y)
#         y = y[:, :-1]
#         y = y.reshape(batch_size, num_tokens, latent_dim)

#         if self.last_layer_norm is not None:
#             y = self.last_layer_norm(y)
#         return dict(x=y)


# Tried with concatenating l_i, encoder-decoder on all ls
class Decode(nn.Module):
    def __init__(self, dim, num_heads, hidden_dim, downscaling, dropout):
        super().__init__()
        self.downscaling = downscaling
        self.self_attn = Encode(
            dim=dim,
            num_heads=num_heads,
            downscaling=1,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.cross_attn = CrossAttentionDecoder(
            dim=dim,
            num_heads=num_heads,
            downscaling=downscaling,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)

        self.rezero_1 = nn.Parameter(torch.zeros(1))
        self.rezero_2 = nn.Parameter(torch.zeros(1))
        self.rezero_3 = nn.Parameter(torch.zeros(1))

        self.mlp = SWIGLU(dim, dropout=dropout)

    def forward(self, x, l):
        out = self.norm(x)
        out = self.self_attn(out, out)
        if self.rezero_1 is not None:
            out = out * self.rezero_1
        x = x + out

        out = self.cross_attn(self.norm3(x), self.norm4(l))
        if self.rezero_2 is not None:
            out = out * self.rezero_2
        x = x + out

        out = self.norm2(x)
        out = self.mlp(out)
        if self.rezero_3 is not None:
            out = self.rezero_3 * out
        out = x + out
        return out


# class PerceiverIO(nn.Module):
#     def __init__(self, dim, num_layers, dropout, **kwargs):
#         super().__init__()
#         self.dim_last_layer = dim
#         self.downscaling = 16

#         num_latents = 64

#         # Fixed sinusoidal embeddings?
#         self.l_init = torch.zeros(num_latents, 512)
#         position = torch.arange(0, num_latents, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, 512, 2).float() * (
#             -math.log(10000.0) / 512))
#         self.l_init[:, 0::2] = torch.sin(position * div_term)
#         self.l_init[:, 1::2] = torch.cos(position * div_term)
#         self.l_init = nn.Parameter(self.l_init.unsqueeze(0), requires_grad=False)

#         # self.l_init = nn.Parameter(torch.randn(1, 64, 512))
#         self.encode = Encode(dim=512,
#                              num_heads=8,
#                              downscaling=self.downscaling,
#                              hidden_dim=512,
#                              dropout=dropout)
#         self.process_layers = nn.ModuleList(
#             nn.Sequential(*[
#                 Process(dim, num_heads=8, hidden_dim=512, dropout=dropout)
#                 for _ in range(num_layers)
#             ]) for _ in range(4))

#         # Small encoder-decoder transformer
#         self.decoder_layers = nn.ModuleList([
#             Decode(dim,
#                    num_heads=8,
#                    hidden_dim=dim,
#                    downscaling=self.downscaling,
#                    dropout=dropout) for _ in range(4)
#         ])

#         self.dummy_latent = nn.Parameter(torch.randn(1, 1, 512))

#     def forward(self, x, **kwargs):
#         batch_size, num_tokens, feature_dim = x.size()
#         _, num_latents, latent_dim = self.l_init.size()

#         # intialise the tower of latents
#         l = self.l_init.expand(batch_size, num_latents, latent_dim)
#         l = self.encode(x, l)

#         # and offset
#         l = torch.cat([self.dummy_latent.repeat(batch_size, 1, 1), l],
#                         dim=1)[:, :-1]

#         y = x
#         y = y.reshape(batch_size * num_tokens // self.downscaling,
#                         self.downscaling, feature_dim)
#         for decoder_layer, process_layer in zip(self.decoder_layers,
#                                                 self.process_layers):
#             l = process_layer(l)
#             l_cond = l.repeat_interleave(num_tokens // self.downscaling, dim=0)
#             y = decoder_layer(y, l_cond)

#         y = y.reshape(batch_size, num_tokens, latent_dim)
#         return dict(x=x)
