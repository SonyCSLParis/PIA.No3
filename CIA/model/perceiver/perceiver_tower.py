import math

import torch
import torch.nn as nn
from CIA.model.attentions.scaling_attention import ScalingAttention
from CIA.model.attentions.sliding_causal_local_self_attention import (
    SlidingCausalLocalSelfAttention,
)
from CIA.model.causal_events_model_full_cat import GEGLU
from CIA.model.perceiver.perceiver import Perceiver


class PerceiverTower(Perceiver):
    def __init__(
        self,
        dim,
        num_layers,
        tower_depth,
        num_heads,
        dropout,
        local_window_size,
        num_events,
        downscaling,
    ):
        self.dim = dim
        self.dim_last_layer = dim  # needed by handler
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_layers = num_layers
        self.local_window_size = local_window_size
        # latents
        self.downscaling = downscaling
        self.num_events_latent = num_events // downscaling
        self.latent_dim = dim  # same dim for downscaled transformers
        self.tower_depth = tower_depth
        super(PerceiverTower, self).__init__(dim=dim)

    # def forward(self, x, **kwargs):
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
                    downscaling=self.downscaling,
                    residual=False,
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
                    downscaling=self.downscaling,
                    residual=True,
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
                    num_layers=self.tower_depth,
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
                )
                for _ in range(self.num_layers)
            ]
        )


class QKV_Write(nn.Module):
    def __init__(self, dim, num_heads, dropout, residual, downscaling):
        super().__init__()
        self.scaling_atn = ScalingAttention(
            dim=dim,
            num_heads=num_heads,
            downscaling=downscaling,
            dropout=dropout,
            norm_out=True,
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
    def __init__(self, dim, num_heads, dropout, residual, downscaling):
        super().__init__()
        self.scaling_atn = ScalingAttention(
            dim=dim,
            num_heads=num_heads,
            downscaling=downscaling,
            dropout=dropout,
            norm_out=False,
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
    def __init__(self, dim, num_heads, hidden_dim, dropout, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.self_attn = nn.ModuleList(
            [
                ScalingAttention(
                    dim=dim,
                    num_heads=num_heads,
                    downscaling=1,
                    dropout=dropout,
                    norm_out=False,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
        # self.norm = lambda x: x
        # self.norm2 = lambda x: x

        # self.mlp = SWIGLU(dim, dropout) ? Was this used?
        self.mlp = nn.ModuleList(
            [
                GEGLU(dim, hidden_dim=hidden_dim, output_dim=dim, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

        self.rezero_1 = nn.Parameter(torch.zeros(num_layers))
        self.rezero_2 = nn.Parameter(torch.zeros(num_layers))

    def forward(self, latent):
        for layer_ind in range(self.num_layers):
            latent_norm = self.norm[layer_ind](latent)
            out = self.self_attn[layer_ind](q=latent_norm, kv=latent_norm)
            if self.rezero_1 is not None:
                out = self.rezero_1[layer_ind] * out
            latent = latent + out

            latent_norm = self.norm2[layer_ind](latent)
            out = self.mlp[layer_ind](latent_norm)
            if self.rezero_2 is not None:
                out = self.rezero_2[layer_ind] * out
            latent = latent + out
        return latent


class Process_x(nn.Module):
    def __init__(self, dim, num_heads, local_window_size, dropout):
        super().__init__()
        self.atn = SlidingCausalLocalSelfAttention(
            dim=dim,
            local_heads=num_heads,
            heads=num_heads,
            causal=True,
            local_window_size=local_window_size,
            dropout=dropout,
        )
        self.ff = GEGLU(dim, hidden_dim=2 * dim, output_dim=dim, dropout=dropout)
        self.norm_atn = nn.LayerNorm(dim)
        self.norm_ff = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.atn(self.norm_atn(x))
        x = x + self.ff(self.norm_ff(x))
        return x
