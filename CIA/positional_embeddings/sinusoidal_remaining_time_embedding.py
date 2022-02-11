from CIA.positional_embeddings.positional_embedding import BasePositionalEmbedding
from torch import nn
from CIA.utils import flatten
import torch
import math


class SinusoidalRemainingTimeEmbedding(BasePositionalEmbedding):
    def __init__(
        self,
        positional_embedding_size,
        num_channels,
        dataloader_generator,
        data_processor,
        dropout,
        expand_channels,
        **kwargs,
    ):
        super(SinusoidalRemainingTimeEmbedding, self).__init__(
            expand_channels=expand_channels
        )
        assert positional_embedding_size % 2 == 0
        self.data_processor = data_processor
        self.dataloader_generator = dataloader_generator
        self.positional_embedding_size = positional_embedding_size

        self.dropout = torch.nn.Dropout(p=dropout)
        self.num_channels = num_channels

    def forward(self, x_embed, i, metadata_dict):
        assert i == 0
        batch_size, num_events, _ = x_embed.size()

        # add embedding_dim to elapsed time
        elapsed_time = self.data_processor.compute_elapsed_time(metadata_dict)
        remaining_time = metadata_dict["remaining_time"].unsqueeze(1) - elapsed_time
        ################################################
        # zero remaining_time in prefix
        remaining_time[:, : self.data_processor.num_events_context] = 0
        # zero negative values
        remaining_time = torch.where(
            remaining_time < 0.0, torch.zeros_like(remaining_time), remaining_time
        )
        ################################################
        remaining_time = remaining_time.unsqueeze(2)
        # scaling
        remaining_time = remaining_time * 100
        if self.expand_channels:
            remaining_time = remaining_time.repeat_interleave(self.num_channels, dim=1)
        else:
            remaining_time = remaining_time

        # sinusoid
        pe = torch.zeros(batch_size, num_events, self.positional_embedding_size)
        pos_embedding = pe.to(device=x_embed.device)
        div_term = torch.exp(
            torch.arange(0, self.positional_embedding_size, 2).float()
            * (-math.log(10000.0) / self.positional_embedding_size)
        )
        div_term = div_term.to(device=x_embed.device)
        div_term = div_term.unsqueeze(0).unsqueeze(0)
        pos_embedding[:, :, 0::2] = torch.sin(remaining_time * div_term)
        pos_embedding[:, :, 1::2] = torch.cos(remaining_time * div_term)

        pos_embedding = self.dropout(pos_embedding)
        x_embed = torch.cat([x_embed, pos_embedding], dim=2)
        return x_embed
