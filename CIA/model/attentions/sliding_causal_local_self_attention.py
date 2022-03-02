import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from einops.einops import rearrange


class SlidingCausalLocalSelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        local_window_size,
        dropout,
        qkv_bias,  # False
        attn_out_bias,  # True
        relative_pos_bias,
    ):
        super().__init__()
        assert dim % heads == 0, "dimension must be divisible by number of heads"
        self.heads = heads

        # local self attn
        self.window_size = local_window_size
        self.look_backward = 1
        self.look_forward = 0
        self.exact_windowsize = False
        self.autopad = True
        self.dropout_attn = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_out = nn.Linear(dim, dim, bias=attn_out_bias)
        self.dropout = nn.Dropout(dropout)

        # if relative_pos_bias:
        #     raise NotImplementedError

    def forward(self, x):
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (q, k, v)
        )
        out = self.local_attn(q, k, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return self.dropout(out)

    def local_attn(self, q, k, v):
        shape = q.shape

        merge_into_batch = lambda t: t.reshape(-1, *t.shape[-2:])
        q, k, v = map(merge_into_batch, (q, k, v))

        if self.autopad:
            orig_t = q.shape[1]
            q, k, v = map(
                lambda t: pad_to_multiple(t, self.window_size, dim=-2),
                (q, k, v),
            )

        window_size, look_backward, look_forward = (
            self.window_size,
            self.look_backward,
            self.look_forward,
        )
        b, t, e, device, dtype = *q.shape, q.device, q.dtype
        assert (
            t % window_size
        ) == 0, f"sequence length {t} must be divisible by window size {window_size} for local attention"

        windows = t // window_size

        ticker = torch.arange(t, device=device, dtype=dtype)[None, :]
        b_t = ticker.reshape(1, windows, window_size)

        bucket_fn = lambda t: t.reshape(b, windows, window_size, -1)
        bq, bk, bv = map(bucket_fn, (q, k, v))

        look_around_kwargs = {"backward": look_backward, "forward": look_forward}
        bk = look_around(bk, **look_around_kwargs)
        bv = look_around(bv, **look_around_kwargs)

        bq_t = b_t
        bq_k = look_around(b_t, **look_around_kwargs)

        dots = torch.einsum("bhie,bhje->bhij", bq, bk) * (e ** -0.5)

        mask_value = max_neg_value(dots)

        mask = bq_t[:, :, :, None] < bq_k[:, :, None, :]
        if self.exact_windowsize:
            max_causal_window_size = self.window_size * self.look_backward
            mask = mask | (
                bq_t[:, :, :, None] > (bq_k[:, :, None, :] + max_causal_window_size)
            )
        dots.masked_fill_(mask, mask_value)
        del mask

        mask = bq_k[:, :, None, :] == -1
        dots.masked_fill_(mask, mask_value)
        del mask

        attn = (dots).softmax(dim=-1)
        attn = self.dropout_attn(attn)

        out = torch.einsum("bhij,bhje->bhie", attn, bv)
        out = out.reshape(-1, t, e)

        if self.autopad:
            out = out[:, :orig_t, :]

        return out.reshape(*shape)


def pad_to_multiple(tensor, multiple, dim=-1, value=0):
    seqlen = tensor.shape[dim]
    m = seqlen / multiple
    if m.is_integer():
        return tensor
    remainder = math.ceil(m) * multiple - seqlen
    pad_offset = (0,) * (-1 - dim) * 2
    return F.pad(tensor, (*pad_offset, 0, remainder), value=value)


def look_around(x, backward=1, forward=0, pad_value=-1, dim=2):
    t = x.shape[1]
    dims = (len(x.shape) - dim) * (0, 0)
    padded_x = F.pad(x, (*dims, backward, forward), value=pad_value)
    tensors = [
        padded_x[:, ind : (ind + t), ...] for ind in range(forward + backward + 1)
    ]
    return torch.cat(tensors, dim=dim)


def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max
