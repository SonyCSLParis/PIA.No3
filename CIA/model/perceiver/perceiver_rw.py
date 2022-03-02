import math

import torch
import torch.nn as nn
from CIA.model.attentions.scaling_attention import ScalingAttention
from CIA.model.attentions.sliding_causal_local_self_attention import (
    SlidingCausalLocalSelfAttention,
)
from CIA.model.causal_events_model_full_cat import GEGLU
from CIA.model.perceiver.perceiver import Perceiver


class PerceiverReadWrite(Perceiver):
    def __init__(
        self,
        dim,
        num_layers,
        num_heads,
        dropout,
        local_window_size,
        num_events,
        downscaling,
        relative_pos_bias,
    ):
        self.dim = dim
        self.dim_last_layer = dim  # needed by handler
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_layers = num_layers
        self.local_window_size = local_window_size
        self.relative_pos_bias = relative_pos_bias
        # latents
        self.num_events = num_events
        self.num_events_latent = num_events // downscaling
        self.latent_dim = dim  # same dim for downscaled transformers
        super(PerceiverReadWrite, self).__init__(dim=dim)

    # def forward(self, x, **kwargs):

    def _get_last_layer_norm(self):
        return nn.LayerNorm(self.dim)

    def _get_latents_init(self):
        # which init?
        latents_init = torch.zeros(self.num_events_latent, self.latent_dim)
        position = torch.arange(0, self.num_events_latent, dtype=torch.float).unsqueeze(
            1
        )
        div_term = torch.exp(
            torch.arange(0, self.latent_dim, 2).float()
            * (-math.log(10000.0) / self.latent_dim)
        )
        latents_init[:, 0::2] = torch.sin(position * div_term)
        latents_init[:, 1::2] = torch.cos(position * div_term)
        latents_init = nn.Parameter(latents_init.unsqueeze(0), requires_grad=False)
        dummy_l = nn.ParameterList(
            [
                nn.Parameter(torch.randn(1, 1, self.latent_dim))
                for _ in range(self.num_layers)
            ]
        )
        return latents_init, dummy_l

    def _get_write(self):
        # TODO: Residual here?
        return nn.ModuleList(
            [
                QKV_Write(
                    self.dim,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                    seq_len_src=self.num_events,
                    seq_len_tgt=self.num_events_latent,
                    residual=False,
                    relative_pos_bias=self.relative_pos_bias,
                )
                for _ in range(self.num_layers)
            ]
        )

    def _get_read(self):
        return nn.ModuleList(
            [
                QKV_Read(
                    self.dim,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                    seq_len_src=self.num_events_latent,
                    seq_len_tgt=self.num_events,
                    residual=True,
                    relative_pos_bias=self.relative_pos_bias,
                )
                for _ in range(self.num_layers)
            ]
        )

    def _get_process_l(self):
        return nn.ModuleList(
            [
                Process_l(
                    dim=self.dim,
                    hidden_dim=self.dim * 2,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                    seq_len=self.num_events_latent,
                    relative_pos_bias=self.relative_pos_bias,
                )
                for _ in range(self.num_layers)
            ]
        )

    def _get_process_x(self):
        return nn.ModuleList(
            [
                Process_x(
                    dim=self.dim,
                    num_heads=self.num_heads,
                    local_window_size=self.local_window_size,
                    dropout=self.dropout,
                    relative_pos_bias=self.relative_pos_bias,
                )
                for _ in range(self.num_layers)
            ]
        )


class QKV_Write(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        dropout,
        residual,
        seq_len_src,
        seq_len_tgt,
        relative_pos_bias,
    ):
        super().__init__()

        self.scaling_atn = ScalingAttention(
            dim=dim,
            num_heads=num_heads,
            seq_len_src=seq_len_src,
            seq_len_tgt=seq_len_tgt,
            dropout=dropout,
            norm_out=True,
            relative_pos_bias=relative_pos_bias,
        )
        self.norm_x = nn.LayerNorm(dim)
        self.norm_l = nn.LayerNorm(dim)
        self.residual = residual

    def forward(self, x, latents):
        """[summary]

        Args:
            x (batch_size * num_latents, downscaling, d):
            l (batch_size * num_latents, num_latents, d ):
        """
        latents_norm = self.norm_l(latents)
        x_norm = self.norm_x(x)
        out = self.scaling_atn(q=latents_norm, kv=x_norm)
        if self.residual:
            out = latents + out
        return out


class QKV_Read(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        dropout,
        residual,
        seq_len_src,
        seq_len_tgt,
        relative_pos_bias,
    ):
        super().__init__()
        self.scaling_atn = ScalingAttention(
            dim=dim,
            num_heads=num_heads,
            seq_len_src=seq_len_src,
            seq_len_tgt=seq_len_tgt,
            dropout=dropout,
            norm_out=False,
            relative_pos_bias=relative_pos_bias,
        )
        self.norm_x = nn.LayerNorm(dim)
        self.residual = residual

    def forward(self, x, latents):
        """[summary]

        Args:
            x (batch_size * num_latents, downscaling, d):
            l (batch_size * num_latents, num_latents, d ):
        """
        x_norm = self.norm_x(x)
        out = self.scaling_atn(q=x_norm, kv=latents)
        if self.residual:
            out = x + out
        return out


class Process_l(nn.Module):
    def __init__(self, dim, num_heads, hidden_dim, dropout, seq_len, relative_pos_bias):
        super().__init__()

        self.self_attn = ScalingAttention(
            dim=dim,
            num_heads=num_heads,
            seq_len_src=seq_len,
            seq_len_tgt=seq_len,
            dropout=dropout,
            norm_out=False,
            relative_pos_bias=relative_pos_bias,
        )
        self.norm = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        # self.norm = lambda x: x
        # self.norm2 = lambda x: x

        # self.mlp = SWIGLU(dim, dropout) ? Was this used?
        self.mlp = GEGLU(dim, hidden_dim=hidden_dim, output_dim=dim, dropout=dropout)

        self.rezero_1 = nn.Parameter(torch.zeros(1))
        self.rezero_2 = nn.Parameter(torch.zeros(1))

    def forward(self, latent):
        latent_norm = self.norm(latent)
        out = self.self_attn(q=latent_norm, kv=latent_norm)
        if self.rezero_1 is not None:
            out = self.rezero_1 * out
        latent = latent + out

        latent_norm = self.norm2(latent)
        out = self.mlp(latent_norm)
        if self.rezero_2 is not None:
            out = self.rezero_2 * out
        out = latent + out
        return out


class Process_x(nn.Module):
    def __init__(self, dim, num_heads, local_window_size, dropout, relative_pos_bias):
        super().__init__()
        self.atn = SlidingCausalLocalSelfAttention(
            dim=dim,
            heads=num_heads,
            local_window_size=local_window_size,
            dropout=dropout,
            qkv_bias=False,
            attn_out_bias=True,
            relative_pos_bias=relative_pos_bias,
        )
        self.ff = GEGLU(dim, hidden_dim=2 * dim, output_dim=dim, dropout=dropout)
        self.norm_atn = nn.LayerNorm(dim)
        self.norm_ff = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.atn(self.norm_atn(x))
        x = x + self.ff(self.norm_ff(x))
        return x
