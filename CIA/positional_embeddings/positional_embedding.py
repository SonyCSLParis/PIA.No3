from torch import nn


class BasePositionalEmbedding(nn.Module):
    def __init__(self, expand_channels) -> None:
        super().__init__()
        # BasePositionalEmbedding must define
        # positional_embedding_size

        self.expand_channels = expand_channels
        # expand_channels is True if the BasePositionalEmbeddings are the one used in PIAv1 (one token is one channel)

    def forward(self, x_embed, i=0, h=None, metadata_dict={}):
        return x_embed, h

    def forward_step(self, x_embed, i=0, h=None, metadata_dict={}):
        return x_embed, h


class PositionalEmbedding(nn.Module):
    """Positional embeddings built from a list of
    "base" positional embeddings like
    - ChannelEmbedding,
    - SinusoidalEmbedding
    - etc.
    """

    def __init__(self, base_positional_embedding_list) -> None:
        super().__init__()
        self.base_positional_embeddings = nn.ModuleList(base_positional_embedding_list)
        self.positional_embedding_size = sum(
            [pe.positional_embedding_size for pe in base_positional_embedding_list]
        )

    def forward(self, x_embed, i=0, metadata_dict={}):
        """Concatenates all the simple_positional_embeddings
        on the last dim of x_embed

        Args:
            x_embed (batch_size, num_tokens, embedding_dim): embedded_sequence
            i (int, optional): index of the first token. Defaults to 0.
            target (batch_size, num_events_num_channels, optional):
            The target tensor (not embedded), can be used compute some quantities. Defaults to None.

        Output:
            x_embed_with
        """
        for positional_embedding in self.base_positional_embeddings:
            x_embed = positional_embedding.forward(
                x_embed, i=i, metadata_dict=metadata_dict
            )
        return x_embed

    def forward_step(self, x_embed, i=0, metadata_dict={}):
        """Concatenates all the simple_positional_embeddings
        on the last dim of x_embed

        Args:
            x_embed (batch_size, embedding_dim): embedded_sequence
            i (int, optional): index of the token in the whole sequence. Defaults to 0.
            h (list of tensors, optional): cached values, one for each embedding. Defaults to None.
            target (batch_size, num_events_num_channels, optional):
            The target tensor (not embedded), can be used compute some quantities. Defaults to None.
        """
        for positional_embedding in zip(self.base_positional_embeddings):
            x_embed = positional_embedding.forward_step(
                x_embed, i=i, metadata_dict=metadata_dict
            )
        return x_embed
