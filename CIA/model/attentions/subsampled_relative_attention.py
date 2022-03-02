import time

import torch
import torch.nn as nn

from CIA.utils import cuda_variable


class SubsampledRelativeAttention(nn.Module):
    def __init__(self, head_dim, num_heads, seq_len_src, seq_len_tgt):
        super(SubsampledRelativeAttention, self).__init__()

        self.head_dim = head_dim
        self.seq_len_src = seq_len_src
        self.seq_len_tgt = seq_len_tgt
        self.num_heads = num_heads
        if seq_len_src > seq_len_tgt:
            self.mode = "downsampling"
            self.subsampling_ratio = int(seq_len_src // seq_len_tgt)
            # e = torch.randn(num_heads, seq_len_tgt, self.head_dim)
            # self.e = nn.Parameter(
            #     torch.repeat_interleave(e, self.subsampling_ratio, dim=1)
            # )
            self.e = nn.Parameter(torch.randn(num_heads, seq_len_tgt, self.head_dim))
        elif seq_len_src < seq_len_tgt:
            self.mode = "upsampling"
            self.subsampling_ratio = int(seq_len_tgt // seq_len_src)
            self.e = nn.Parameter(torch.randn(num_heads, seq_len_src, self.head_dim))
        elif seq_len_src == seq_len_tgt:
            self.mode = "fix"
            self.e = nn.Parameter(torch.randn(num_heads, seq_len_src, self.head_dim))

        # # NOTES: test values
        # print("!!!USING TEST VALUES!!!")
        # import numpy as np

        # if seq_len_src > seq_len_tgt:
        #     # downsampling
        #     self.subsampling_ratio = int(seq_len_src // seq_len_tgt)
        #     aa = np.arange(seq_len_tgt)
        #     e = cuda_variable(
        #         torch.tensor(aa)
        #         .unsqueeze(0)
        #         .unsqueeze(2)
        #         .repeat(num_heads, 1, head_dim)
        #         .float()
        #     )
        #     self.e = torch.repeat_interleave(e, self.subsampling_ratio, dim=1)
        # elif seq_len_src <= seq_len_tgt:
        #     # upsampling
        #     self.subsampling_ratio = int(seq_len_tgt // seq_len_src)
        #     aa = np.arange(seq_len_src)
        #     self.e = cuda_variable(
        #         torch.tensor(aa)
        #         .unsqueeze(0)
        #         .unsqueeze(2)
        #         .repeat(num_heads, 1, head_dim)
        #         .float()
        #     )

    def forward(self, q):
        """

        :param q: (batch_size * num_heads, len_q_tgt, d)
        :return:
        """
        b, h, l, _ = q.size()
        b_h = b * h

        ################################
        # Causal
        e = self.e.unsqueeze(0).repeat(b, 1, 1, 1)
        rel_attn = torch.einsum("bhld,bhmd->bhlm", (q, e))
        if self.mode == "upsampling":
            len_src = self.seq_len_src
            len_tgt = self.seq_len_tgt
        else:
            len_src = self.seq_len_tgt
            len_tgt = self.seq_len_tgt

        # group head and batch, and transpose tgt * src -> src * tgt. Does nothing if len_src and len_tgt are the same
        rel_attn = rel_attn.view(b_h, len_src, len_tgt)
        # one column padding on dim 2
        rel_attn = torch.cat(
            [
                cuda_variable(torch.ones(b_h, len_src, 1) * -100),
                rel_attn,
            ],
            dim=2,
        )
        #  fill in with lines (ensure view can be done)
        bottom_extension = len_tgt - len_src
        if bottom_extension != 0:
            rel_attn = torch.cat(
                [
                    rel_attn,
                    cuda_variable(
                        torch.ones(b_h, bottom_extension, len_tgt + 1) * -100
                    ),
                ],
                dim=1,
            )
        #  skewing
        rel_attn = rel_attn.view(b_h, -1, len_src)
        #  need to remove first line here
        rel_attn = rel_attn[:, 1:]
        rel_attn = rel_attn[:, :len_tgt, :]
        if self.mode == "downsampling":
            rel_attn = torch.repeat_interleave(rel_attn, self.subsampling_ratio, dim=-1)
        ################################

        #  mask causal
        masks_up = (
            torch.triu(
                torch.ones_like(rel_attn[0, :len_src, :len_src]).byte(),
                diagonal=1,
            )
            .unsqueeze(0)
            .repeat(b_h, 1, 1)
            .type(torch.bool)
        )
        if self.mode == "upsampling":
            masks_up = torch.repeat_interleave(masks_up, self.subsampling_ratio, dim=1)
        elif self.mode == "downsampling":
            masks_up = torch.repeat_interleave(masks_up, self.subsampling_ratio, dim=2)

        rel_attn = rel_attn.masked_fill(masks_up, 0)
        return rel_attn.view(b, h, self.seq_len_tgt, self.seq_len_src)


# if __name__ == "__main__":
#     batch_size = 1
#     head_dim = 2
#     num_heads = 1
#     seq_len_src = 6
#     seq_len_tgt = 6
#     aa = SubsampledRelativeAttention(head_dim, num_heads, seq_len_src, seq_len_tgt)
#     aa.to("cuda")
#     q = cuda_variable(torch.ones((batch_size * num_heads, seq_len_tgt, head_dim)))
#     ret = aa.forward(q)
#     exit()
