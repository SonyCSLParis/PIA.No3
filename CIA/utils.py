import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import NullFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import nn
import torch.distributed as dist


def cuda_variable(tensor, non_blocking=False):
    if torch.cuda.is_available():
        # return tensor.to('cuda', non_blocking=non_blocking)
        return tensor.to(f"cuda:{dist.get_rank()}", non_blocking=non_blocking)
    else:
        return tensor


def is_main_process():
    return dist.get_rank() == 0


def get_free_port():
    import socketserver

    with socketserver.TCPServer(("localhost", 0), None) as s:
        return s.server_address[1]


def all_reduce_scalar(scalar, average=True):
    t = torch.Tensor([scalar]).to(f"cuda:{dist.get_rank()}")
    dist.all_reduce(t)
    scalar = t[0].detach().item()
    if average:
        scalar = scalar / dist.get_world_size()
    return scalar


def display_monitored_quantities(
    epoch_id, monitored_quantities_train, monitored_quantities_val
) -> None:
    if is_main_process():
        print(f"======= Epoch {epoch_id} =======")
        print(f"---Train---")
        dict_pretty_print(monitored_quantities_train, endstr=" " * 5)
        print()
        print(f"---Val---")
        dict_pretty_print(monitored_quantities_val, endstr=" " * 5)
        print("\n")


def to_numpy(tensor):
    return tensor.detach().to("cpu").numpy()


def dict_pretty_print(d, endstr="\n"):
    for key, value in d.items():
        if type(value) == list:
            print(f"{key.capitalize()}: [%s]" % ", ".join(map(str, value)))
        else:
            print(f"{key.capitalize()}: {value:.6}", end=endstr)


def chorale_accuracy(value, target):
    """
    :param value: list of (batch_size, chorale_length, num_notes)
    :param target: (batch_size, num_voices, chorale_length)
    :return:
    """
    batch_size, num_voices, chorale_length = target.size()
    batch_size, chorale_length, _ = value[0].size()
    num_voices = len(value)

    # put num_voices first
    target = target.transpose(0, 1)

    sum = 0
    for voice, voice_target in zip(value, target):
        max_values, max_indexes = torch.max(voice, dim=2, keepdim=False)
        num_correct = (max_indexes == voice_target).float().mean().item()
        sum += num_correct

    return sum / num_voices


def log_normal(x, mean, log_var, eps=0.00001):
    return -((x - mean) ** 2) / (2.0 * torch.exp(log_var) + eps) - log_var / 2.0 + c


def categorical_crossentropy(value, target, mask=None, label_smoothing=False):
    """

    :param value: list of (batch_size, num_events, num_tokens_of_corresponding_channel)
    :param target: (batch_size, num_events, num_channels)
    :param mask: (batch_size, num_events, num_channels)
    :return:
    """
    cross_entropy = nn.CrossEntropyLoss(reduction="none")
    sum = 0

    for channel_probs, target_channel, mask_channel in zip(
        value, target.split(1, dim=2), mask.split(1, dim=2)
    ):
        # select relevent indices
        batch_size, num_events, num_tokens_of_channel = channel_probs.size()

        probs = channel_probs[mask_channel.bool().repeat(1, 1, num_tokens_of_channel)]
        target = target_channel[mask_channel.bool()]

        probs = probs.view(-1, num_tokens_of_channel)
        tgt = target.view(-1)

        if not label_smoothing:
            ce = cross_entropy(probs, tgt)
        else:
            eps = 0.02
            one_hot = torch.zeros_like(probs).scatter(1, tgt.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (
                num_tokens_of_channel - 1
            )
            log_prb = nn.functional.log_softmax(probs, dim=1)
            ce = -(one_hot * log_prb).sum(dim=1)
        sum = sum + ce.sum()

    # divide by the total number of tokens
    if mask.sum() > 0:
        sum = sum / mask.sum()
    return sum


def distilled_categorical_crossentropy(value, target, mask=None):
    """

    :param value: list of (batch_size, num_events, num_notes)
    :param target: list of (batch_size, num_events, num_notes)
    :return:
    """

    def cross_entropy_from_logits(p, q):
        """
        sum softmax(p) log softmax(q)
        :param p:
        :param q:
        :return:
        """
        p = torch.softmax(p, dim=1)
        log_term = q - torch.logsumexp(q, dim=1, keepdim=True)
        return -torch.sum(p * log_term, dim=1)

    def cross_entropy_from_logits_check(p, q):
        """
        sum softmax(p) log softmax(q)
        :param p:
        :param q:
        :return:
        """
        p = torch.softmax(p, dim=1)
        q = torch.softmax(q, dim=1)
        return -torch.sum(p * torch.log(q), dim=1)

    sum = 0
    # TODO better mask horrible
    for channel, channel_target, channel_mask in zip(
        value, target, mask.split(1, dim=2)
    ):
        # channel is (batch_size, num_events, num_tokens_of_corresponding_channel)
        # channel_target is (batch_size, num_events)
        # TODO remove this loop
        for probs, label, m in zip(
            channel.split(1, dim=1),
            channel_target.split(1, dim=1),
            channel_mask.split(1, dim=1),
        ):
            if m.squeeze(2).squeeze(1).float().mean().item() > 0.5:
                ce = cross_entropy_from_logits(label.squeeze(1), probs.squeeze(1))
                sum = sum + ce
    return sum


def quantization_loss(
    loss_quantization_left, loss_quantization_negative, loss_quantization_right
):
    loss_quantization = torch.cat(
        (
            loss_quantization_left.sum(2).sum(1),
            loss_quantization_right.sum(2).sum(1),
            loss_quantization_negative.sum(4).sum(3).sum(2).sum(1),
        ),
        dim=0,
    ).mean()
    return loss_quantization


def quantization_loss_no_negative(loss_quantization_left, loss_quantization_right):
    loss_quantization = torch.cat(
        (
            loss_quantization_left.sum(2).sum(1),
            loss_quantization_right.sum(2).sum(1),
        ),
        dim=0,
    ).mean()
    return loss_quantization


def to_sphere(t):
    return t / t.norm(dim=-1, keepdim=True)


def flatten(x):
    """

    :param x:(batch, num_events, num_channels, ...)
    :return: (batch, num_events * num_channels, ...) with num_channels varying faster
    """
    size = x.size()
    assert len(size) >= 3
    batch_size, num_events, num_channels = size[:3]
    remaining_dims = list(size[3:])
    x = x.view(batch_size, num_events * num_channels, *remaining_dims)
    return x


def unflatten(sequence, num_channels):
    """

    :param sequence: (batch_size, num_events * num_channels, ...)
    where num_channels is varying faster
    :return: (batch_size, num_events, num_channels, ...)
    """
    size = sequence.size()
    assert len(size) >= 2
    batch_size, sequence_length = size[:2]
    assert sequence_length % num_channels == 0
    num_events = sequence_length // num_channels
    remaining_dims = list(size[2:])

    chorale = sequence.view(batch_size, num_events, num_channels, *remaining_dims)
    return chorale


def timing_gpu():
    """
    Just to remember how to time gpus operation
    :return:
    """

    # notes ##################################
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    end_event.record()
    torch.cuda.synchronize()  # Wait for the events to be recorded!
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"T: {elapsed_time_ms}")
    # notes ##################################


def plot_mi_marginals(px, py, mi_matrix, save_path):
    nullfmt = NullFormatter()  # no labels
    dim_x = len(px)
    dim_y = len(py)
    dim_max = max(dim_x, dim_y)

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    # start with a rectangular Figure
    # plt.figure(1, figsize=(dim_max, dim_max))
    # Or fixed size perhaps ??
    plt.figure(1, figsize=(8, 8))

    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    # Plot MI
    im = axScatter.imshow(mi_matrix, cmap="RdBu")
    divider = make_axes_locatable(axScatter)
    # create an axes on the right side of ax. The width of cax will be 1%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    cax = divider.append_axes("left", size="5%", pad=0.05)
    axScatter.figure.colorbar(im, cax=cax)
    axScatter.set_xlim((-0.5, dim_max - 0.5))
    axScatter.set_ylim((-0.5, dim_max - 0.5))

    # Plot marginals
    colorbar_width = dim_max * 0.05
    axHistx.bar(x=range(dim_y), height=py)
    axHisty.barh(y=range(dim_x), width=np.flip(px))
    axHistx.set_xlim((-0.5 - colorbar_width, dim_max - 0.5))
    axHisty.set_ylim((-0.5, dim_max - 0.5))

    plt.savefig(save_path)
    return


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (vocabulary size)
        top_k >0: keep only top k tokens with highest probability (top-k filtering).
        top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    # batch size 1 for now - could be updated for more but the code would be less clear
    assert logits.dim() == 1
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def concat_elu(x):
    """like concatenated ReLU (http://arxiv.org/abs/1603.05201), but then with ELU"""
    # Pytorch ordering
    axis = len(x.size()) - 3
    return torch.nn.ELU()(torch.cat([x, -x], dim=axis))


def log_prob_from_logits(x):
    """numerically stable log_softmax implementation that prevents overflow"""
    # TF ordering
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis, keepdim=True)
    return x - m - torch.log(torch.sum(torch.exp(x - m), dim=axis, keepdim=True))


def discretized_mix_logistic_loss(x, l):
    """log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval"""
    # Pytorch ordering
    x = x.permute(0, 2, 3, 1)
    l = l.permute(0, 2, 3, 1)
    xs = [int(y) for y in x.size()]
    ls = [int(y) for y in l.size()]

    # here and below: unpacking the params of the mixture of logistics
    nr_mix = int(ls[-1] / 10)
    logit_probs = l[:, :, :, :nr_mix]
    l = (
        l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 3])
    )  # 3 for mean, scale, coef
    means = l[:, :, :, :, :nr_mix]
    # log_scales = torch.max(l[:, :, :, :, nr_mix:2 * nr_mix], -7.)
    log_scales = torch.clamp(l[:, :, :, :, nr_mix : 2 * nr_mix], min=-7.0)

    coeffs = torch.tanh(l[:, :, :, :, 2 * nr_mix : 3 * nr_mix])
    # here and below: getting the means and adjusting them based on preceding
    # sub-pixels
    x = x.contiguous()
    x = x.unsqueeze(-1) + cuda_variable(torch.zeros(xs + [nr_mix]))
    m2 = (means[:, :, :, 1, :] + coeffs[:, :, :, 0, :] * x[:, :, :, 0, :]).view(
        xs[0], xs[1], xs[2], 1, nr_mix
    )

    m3 = (
        means[:, :, :, 2, :]
        + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :]
        + coeffs[:, :, :, 2, :] * x[:, :, :, 1, :]
    ).view(xs[0], xs[1], xs[2], 1, nr_mix)

    means = torch.cat((means[:, :, :, 0, :].unsqueeze(3), m2, m3), dim=3)
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = torch.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = torch.sigmoid(min_in)
    # log probability for edge case of 0 (before scaling)
    log_cdf_plus = plus_in - torch.softplus(plus_in)
    # log probability for edge case of 255 (before scaling)
    log_one_minus_cdf_min = -torch.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * centered_x
    # log probability in the center of the bin, to be used in extreme cases
    # (not actually used in our code)
    log_pdf_mid = mid_in - log_scales - 2.0 * torch.softplus(mid_in)

    # now select the right output: left edge case, right edge case, normal
    # case, extremely low prob case (doesn't actually happen for us)

    # this is what we are really doing, but using the robust version below for extreme cases in other applications and to avoid NaN issue with tf.select()
    # log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999, log_one_minus_cdf_min, tf.log(cdf_delta)))

    # robust version, that still works if probabilities are below 1e-5 (which never happens in our code)
    # tensorflow backpropagates through tf.select() by multiplying with zero instead of selecting: this requires use to use some ugly tricks to avoid potential NaNs
    # the 1e-12 in tf.maximum(cdf_delta, 1e-12) is never actually used as output, it's purely there to get around the tf.select() gradient issue
    # if the probability on a sub-pixel is below 1e-5, we use an approximation
    # based on the assumption that the log-density is constant in the bin of
    # the observed sub-pixel value

    inner_inner_cond = (cdf_delta > 1e-5).float()
    inner_inner_out = inner_inner_cond * torch.log(
        torch.clamp(cdf_delta, min=1e-12)
    ) + (1.0 - inner_inner_cond) * (log_pdf_mid - np.log(127.5))
    inner_cond = (x > 0.999).float()
    inner_out = (
        inner_cond * log_one_minus_cdf_min + (1.0 - inner_cond) * inner_inner_out
    )
    cond = (x < -0.999).float()
    log_probs = cond * log_cdf_plus + (1.0 - cond) * inner_out
    log_probs = torch.sum(log_probs, dim=3) + log_prob_from_logits(logit_probs)

    return -torch.sum(torch.logsumexp(log_probs))


def discretized_mix_logistic_loss_1d(x, l):
    """log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval"""
    # Pytorch ordering
    x = x.permute(0, 2, 3, 1)
    l = l.permute(0, 2, 3, 1)
    xs = [int(y) for y in x.size()]
    ls = [int(y) for y in l.size()]

    # here and below: unpacking the params of the mixture of logistics
    nr_mix = int(ls[-1] / 3)
    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 2])  # 2 for mean, scale
    means = l[:, :, :, :, :nr_mix]
    log_scales = torch.clamp(l[:, :, :, :, nr_mix : 2 * nr_mix], min=-7.0)
    # here and below: getting the means and adjusting them based on preceding
    # sub-pixels
    x = x.contiguous()
    x = x.unsqueeze(-1) + cuda_variable(torch.zeros(xs + [nr_mix]).cuda())

    # means = torch.cat((means[:, :, :, 0, :].unsqueeze(3), m2, m3), dim=3)
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = torch.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = torch.sigmoid(min_in)
    # log probability for edge case of 0 (before scaling)
    log_cdf_plus = plus_in - torch.softplus(plus_in)
    # log probability for edge case of 255 (before scaling)
    log_one_minus_cdf_min = -torch.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * centered_x
    # log probability in the center of the bin, to be used in extreme cases
    # (not actually used in our code)
    log_pdf_mid = mid_in - log_scales - 2.0 * torch.softplus(mid_in)

    inner_inner_cond = (cdf_delta > 1e-5).float()
    inner_inner_out = inner_inner_cond * torch.log(
        torch.clamp(cdf_delta, min=1e-12)
    ) + (1.0 - inner_inner_cond) * (log_pdf_mid - np.log(127.5))
    inner_cond = (x > 0.999).float()
    inner_out = (
        inner_cond * log_one_minus_cdf_min + (1.0 - inner_cond) * inner_inner_out
    )
    cond = (x < -0.999).float()
    log_probs = cond * log_cdf_plus + (1.0 - cond) * inner_out
    log_probs = torch.sum(log_probs, dim=3) + log_prob_from_logits(logit_probs)

    return -torch.sum(torch.logsumexp(log_probs))
