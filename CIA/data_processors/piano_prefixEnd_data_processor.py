import random

import torch
from CIA.dataset_managers.piano_midi_dataset import END_SYMBOL, PAD_SYMBOL, START_SYMBOL
from CIA.utils import cuda_variable
from torch import nn

from .data_processor import DataProcessor


class PianoPrefixEndDataProcessor(DataProcessor):
    def __init__(
        self,
        dataloader_generator,
        embedding_size,
        num_events,
        num_tokens_per_channel,
        num_events_local_window,
        num_events_context,
        reverse_prefix,
    ):
        super(PianoPrefixEndDataProcessor, self).__init__(
            embedding_size=embedding_size,
            num_events=num_events,
            num_tokens_per_channel=num_tokens_per_channel,
            num_additional_tokens=2,  # sod and eod
        )
        # We need full access to the dataset and dataloader_generator
        self.dataloader_generator = dataloader_generator
        self.num_events_local_window = num_events_local_window
        self.num_events_context = num_events_context
        self.num_meta_events = 2  # the 2 accounts for the SOD and EOD tokens

        # Start of decoding
        self.sod_symbols = nn.Parameter(
            torch.LongTensor([nt + 1 for nt in num_tokens_per_channel]),
            requires_grad=False,
        )
        # End of decoding
        self.eod_symbols = nn.Parameter(
            torch.LongTensor(num_tokens_per_channel), requires_grad=False
        )
        # Meta symbols
        self.end_tokens = nn.Parameter(
            torch.LongTensor(
                [
                    self.dataloader_generator.dataset.value2index[feature][END_SYMBOL]
                    for feature in self.dataloader_generator.features
                ]
            ),
            requires_grad=False,
        )
        self.pad_tokens = nn.Parameter(
            torch.LongTensor(
                [
                    self.dataloader_generator.dataset.value2index[feature][PAD_SYMBOL]
                    for feature in self.dataloader_generator.features
                ]
            ),
            requires_grad=False,
        )
        self.start_tokens = nn.Parameter(
            torch.LongTensor(
                [
                    self.dataloader_generator.dataset.value2index[feature][START_SYMBOL]
                    for feature in self.dataloader_generator.features
                ]
            ),
            requires_grad=False,
        )

        self.reverse_prefix = reverse_prefix
        if reverse_prefix:
            raise NotImplementedError

    def reverse(self, x):
        """Reverse midi sequence"""
        # Do more simple: reverse sequence, then shift TS
        # rev_x = x
        # index2ts = self.dataloader_generator.dataset.index2value['time_shift']
        # ts2index = self.dataloader_generator.dataset.value2index['time_shift']
        # ts_channel = self.dataloader_generator.features.index('time_shift')
        # index2d = self.dataloader_generator.dataset.index2value['duration']
        # d_channel = self.dataloader_generator.features.index('duration')
        # for time in range(x.shape[0]):
        #     dt = index2d[int(x[time, d_channel])]
        #     dtm1 = index2d[int(x[time-1, d_channel])] if time != 0 else 0
        #     tstm1 = index2ts[int(x[time-1, ts_channel])] if time != 0 else 0
        #     tst = tstm1 + dt - dtm1
        #     tst_index = ts2index[tst]
        #     rev_x[time, ts_channel] = tst_index
        rev_x = torch.flip(x, [0])
        ts_channel = self.dataloader_generator.features.index("time_shift")
        rev_x[:-1, ts_channel] = rev_x[1:, ts_channel]
        rev_x[-1, ts_channel] = 0
        return rev_x

    def dereverse(self, rev_x):
        ts_channel = self.dataloader_generator.features.index("time_shift")
        rev_x[:, 1:, ts_channel] = rev_x[:, :-1, ts_channel]
        x = torch.flip(rev_x, [1])
        return x

    def preprocess(
        self,
        x,
        num_events_inpainted,
        training,
        non_conditioned_examples,
    ):
        """[summary]

        Args:
            x ([type]):
            decomposes as:

            ======= ======
            before  after

        Returns:
        Sequences of the form
        (parenthesis for optional tokens depending on context):
        ===== ===== ===== === ======= ====== === =====
        after (END) (PAD) SOD (START) before EOD (PAD)

        """
        sequences_size = self.dataloader_generator.sequences_size
        batch_size, num_events, _ = x.size()
        assert num_events == sequences_size
        assert sequences_size > self.num_events_context + self.num_events_local_window

        if not training:
            assert batch_size == 1

        x = cuda_variable(x.long())

        max_num_events_suffix = (
            sequences_size - self.num_events_context - self.num_meta_events
        )

        # === Find end and start tokens in x
        is_start_token = x[:, :, 0] == self.start_tokens[0].unsqueeze(0).unsqueeze(
            0
        ).repeat(batch_size, num_events)
        is_end_token = x[:, :, 0] == self.end_tokens[0].unsqueeze(0).unsqueeze(
            0
        ).repeat(batch_size, num_events)
        contains_start_token = is_start_token.sum(1) >= 1
        contains_end_token = is_end_token.sum(1) >= 1
        # only one of those per sequence
        # Only valid when contains_end_token!!
        start_token_location = torch.argmax(is_start_token.long(), dim=1)
        end_token_location = torch.argmax(is_end_token.long(), dim=1)
        start_ind = torch.where(
            contains_start_token,
            start_token_location,
            torch.zeros_like(start_token_location),
        )
        end_ind = torch.where(
            contains_end_token,
            end_token_location + 1,
            num_events * torch.ones_like(end_token_location),
        )

        ys = []
        remaining_time_l = []
        suffix_len = []
        prefix_len = []
        for batch_ind in range(batch_size):
            # null conditioning
            if non_conditioned_examples and training:
                rand_float = random.random()
                null_masking_batch = bool(rand_float < 0.1)
            else:
                null_masking_batch = False

            # x_trim may contain start and end token if they were present in the sequence
            x_trim = x[batch_ind, start_ind[batch_ind] : end_ind[batch_ind]]
            x_trim_len = x_trim.size(0)
            # shorten randomly x_trim only if no end or start symbol at the end
            if x_trim_len > sequences_size - 2:
                if training:
                    x_trim_len = random.randint(10, sequences_size - 2)
                    x_trim = x_trim[:x_trim_len]
                else:
                    x_trim_len = sequences_size - 2
                    x_trim = x_trim[:x_trim_len]

            if training:
                # draw randomly the length of the suffix
                max_len_suffix = min(x_trim_len - 1, max_num_events_suffix)
                min_len_suffix = max(1, x_trim_len - self.num_events_context)
                if num_events_inpainted is None:
                    num_events_suffix = random.randint(min_len_suffix, max_len_suffix)

                # overwrite num_events_suffix to cover special cases in Ableton plug-in when there's no end provided by user
                # we don't do the same w start because this case is cover by regular training
                if contains_end_token[batch_ind] and (
                    x_trim_len < max_num_events_suffix
                ):
                    if bool(random.random() < 0.2):
                        num_events_suffix = x_trim_len - 1
            else:
                if num_events_inpainted is None:
                    num_events_prefix = min(
                        x_trim_len // 4 + 1, self.num_events_context
                    )
                    num_events_suffix = x_trim_len - num_events_prefix
                    decoding_start_suffix = min(
                        num_events_suffix // 4, self.num_events_context
                    )
                else:
                    x_trim_len

            # split between prefix (= end) and suffix (= beginning)
            suffix = x_trim[:num_events_suffix]
            prefix = x_trim[num_events_suffix:]
            suffix_len.append(len(suffix))
            prefix_len.append(len(prefix))

            if training:
                decoding_start = None
                inpaint_zone_duration = None
            else:
                mid_part = suffix[decoding_start_suffix:]
                inpaint_zone_duration = self.dataloader_generator.get_elapsed_time(
                    mid_part.unsqueeze(0)
                )[:, -1]
                inpaint_zone_duration = inpaint_zone_duration.item()
                decoding_start = decoding_start_suffix + self.num_events_context

            # compute remaining time the suffix
            remaining_time = self.dataloader_generator.get_elapsed_time(
                suffix.unsqueeze(0)
            )[:, -1]
            remaining_time_l.append(remaining_time.squeeze())

            # Asserts
            assert not torch.any(
                prefix[:, 0] == self.start_tokens[0]
            ), "Start token located in prefix!"
            assert not torch.any(
                suffix[:, 0] == self.end_tokens[0]
            ), "End token located in suffix"

            # prefix
            pad_size_prefix = self.num_events_context - prefix.size(0)
            assert pad_size_prefix >= 0
            if pad_size_prefix != 0:
                prefix = torch.cat(
                    [
                        prefix,
                        self.pad_tokens.unsqueeze(0).repeat(pad_size_prefix, 1),
                    ],
                    dim=0,
                )
            if null_masking_batch:
                null_tokens = self.pad_tokens.unsqueeze(0).repeat(prefix.size(0), 1)
                prefix = null_tokens

            # suffix
            pad_size_suffix = max_num_events_suffix - num_events_suffix
            assert pad_size_suffix >= 0
            if pad_size_suffix == 0:
                suffix = torch.cat(
                    [
                        self.sod_symbols.unsqueeze(0),
                        suffix,
                        self.eod_symbols.unsqueeze(0),
                    ],
                    dim=0,
                )
            else:
                suffix = torch.cat(
                    [
                        self.sod_symbols.unsqueeze(0),
                        suffix,
                        self.eod_symbols.unsqueeze(0),
                        self.pad_tokens.unsqueeze(0).repeat(pad_size_suffix, 1),
                    ],
                    dim=0,
                )

            # creates final sequence
            ys.append(torch.cat([prefix, suffix], dim=0))

        y = torch.stack(ys, dim=0)
        remaining_time = torch.stack(remaining_time_l, dim=0)

        # recompute padding mask
        _, num_events_output, _ = y.size()
        padding_mask = y[:, :, :] == self.pad_tokens.unsqueeze(0).unsqueeze(0).repeat(
            batch_size, num_events_output, 1
        )
        sod_mask = y[:, :, :] == self.sod_symbols.unsqueeze(0).unsqueeze(0).repeat(
            batch_size, num_events_output, 1
        )
        start_mask = y[:, :, :] == self.start_tokens.unsqueeze(0).unsqueeze(0).repeat(
            batch_size, num_events_output, 1
        )
        end_mask = y[:, :, :] == self.end_tokens.unsqueeze(0).unsqueeze(0).repeat(
            batch_size, num_events_output, 1
        )
        final_mask = padding_mask + sod_mask + start_mask + end_mask
        # final_mask[:, : self.num_events_context] = True  # remove prefix

        metadata_dict = {
            "remaining_time": remaining_time,
            "original_sequence": y,
            "loss_mask": final_mask,
            "decoding_start": decoding_start,  # only used for generating
            "inpaint_zone_duration": inpaint_zone_duration,
            "suffix_len": suffix_len,  # for monitoring
            "prefix_len": prefix_len,  # for monitoring
        }
        return y, metadata_dict

    def compute_elapsed_time(self, metadata_dict):
        # original sequence is in prefix order!
        x = metadata_dict["original_sequence"]
        elapsed_time = self.dataloader_generator.get_elapsed_time(x)
        # shift to right so that elapsed_time[0] = 0
        elapsed_time = torch.cat(
            [torch.zeros_like(elapsed_time)[:, :1], elapsed_time[:, :-1]], dim=1
        )
        # offset prefix
        elapsed_time[:, : self.num_events_context] = elapsed_time[
            :, : self.num_events_context
        ] + metadata_dict["remaining_time"].unsqueeze(1)
        # reversed?
        if self.reverse_prefix:
            elapsed_time[:, : self.num_events_context] = (
                elapsed_time[:, self.num_events_context - 1].unsqueeze(1)
                - elapsed_time[:, : self.num_events_context]
            )
        # offset suffix
        elapsed_time[:, self.num_events_context :] = (
            elapsed_time[:, self.num_events_context :]
            - elapsed_time[:, self.num_events_context + 1 : self.num_events_context + 2]
        )
        # assert not negative elapsed time
        assert torch.all(elapsed_time >= -9e-3), "Negative elapsed time"

        return elapsed_time

    def postprocess(self, x, decoding_end):
        # put all pieces in order:
        x_out = []
        # TODO: change this
        if type(decoding_end) == int:
            decoding_end = [decoding_end] * len(x)
        for batch_ind in range(len(x)):
            x_out.append(
                torch.cat(
                    [
                        x[
                            batch_ind,
                            self.num_events_context + 1 : decoding_end[batch_ind],
                        ],
                        x[batch_ind, : self.num_events_context],
                    ],
                    dim=0,
                )
            )
        return x_out
