from CIA.start_of_sequence_embeddings.start_of_sequence_embedding import BaseSOSEmbedding


from torch import nn
import torch


class LearntSOSEmbedding(BaseSOSEmbedding):
    def __init__(self, embedding_size):
        super(LearntSOSEmbedding, self).__init__()
        self.sos = nn.Parameter(
            torch.randn((embedding_size,))
        )

    def forward(self, metadata_dict={}):
        # get batch_size
        batch_size = metadata_dict['original_sequence'].size(0)
        sos = self.sos.unsqueeze(0).repeat(batch_size, 1)
        return sos
