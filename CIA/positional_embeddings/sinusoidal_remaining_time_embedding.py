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
        # make sure negative values are very small and only due to rounding/quantization errors
        assert torch.all(
            remaining_time >= -9e-3
        ), f"negative remaining_time values: {torch.min(remaining_time)}"
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

    def forward_step(self, x, i=0, metadata_dict={}):
        if not self.expand_channels:
            raise NotImplementedError

        # time_shift must be the last feature
        assert (
            self.dataloader_generator.features.index("time_shift")
            == len(self.dataloader_generator.features) - 1
        )

        assert (
            "original_token" in metadata_dict
        ), 'Dictionnary metadata_dict must contain entry "original_token" in order to compute the elapsed time'

        batch_size = x.size(0)
        # h represents the elapsed time
        if h is None:
            h = torch.zeros((batch_size,)).to(x.device)

        elapsed_time = h.unsqueeze(1)

        pe = torch.zeros(batch_size, self.positional_embedding_size)
        pe = pe.to(device=x.device)

        div_term = torch.exp(
            torch.arange(0, self.positional_embedding_size, 2).float()
            * (-math.log(10000.0) / self.positional_embedding_size)
        )
        div_term = div_term.to(device=x.device)
        div_term = div_term.unsqueeze(0)
        pe[:, 0::2] = torch.sin(elapsed_time * div_term)
        pe[:, 1::2] = torch.cos(elapsed_time * div_term)

        # dropout only on pe
        pe = self.dropout(pe)
        x_embed = torch.cat([x, pe], dim=1)

        # update h if the current token is a time_shift:
        if i % self.num_channels == self.num_channels - 1:
            # add fake features so that we can call get_elapsed_time
            target = metadata_dict["original_token"]
            target = target.unsqueeze(1).unsqueeze(1)
            target = target.repeat(1, 1, self.num_channels)
            elapsed_time = self.dataloader_generator.get_elapsed_time(target)
            elapsed_time = elapsed_time.squeeze(1)

            # TODO scale?!
            elapsed_time = elapsed_time * 100

            h = h + elapsed_time

        # TODO check this
        if "decoding_start" in metadata_dict:
            if i == self.num_channels * metadata_dict["decoding_start"] - 1:
                h = torch.zeros_like(h)

        return x_embed
