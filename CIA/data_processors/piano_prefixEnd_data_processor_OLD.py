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
            add_mask_token=True,
            num_additional_tokens=2,
        )
        # We need full access to the dataset and dataloader_generator
        self.dataloader_generator = dataloader_generator
        self.num_events_local_window = num_events_local_window
        self.num_events_context = num_events_context

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

        x = cuda_variable(x.long())

        num_events_suffix = num_events_inpainted
        # the 2 accounts for the SOD and EOD tokens
        num_meta_events = 2
        max_num_events_suffix = (
            sequences_size - (self.num_events_context + num_meta_events) - 1
        )
        if num_events_suffix is None:
            num_events_suffix = random.randint(1, max_num_events_suffix)

        # Slice x
        x = x[:, : self.num_events_context + num_events_suffix]
        batch_size, num_events, _ = x.size()

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
        # Only valid when containes_end_token!!
        start_token_location = torch.argmax(is_start_token.long(), dim=1)
        end_token_location = torch.argmax(is_end_token.long(), dim=1)

        before = x[:, :num_events_suffix]
        after = x[:, num_events_suffix : num_events_suffix + self.num_events_context]
        remaining_time = self.dataloader_generator.get_elapsed_time(before)[:, -1]
        placeholder_duration = self.dataloader_generator.get_elapsed_time(
            before[:, self.num_events_local_window :]
        )[:, -1]

        prefix_list, suffix_list = [], []
        prefix_list_nonull, suffix_list_nonull = [], []
        # TODO batch this?!
        for (b, a, c_start_token, c_end_token, start_token_l, end_token_l) in zip(
            before,
            after,
            contains_start_token,
            contains_end_token,
            start_token_location,
            end_token_location,
        ):
            if non_conditioned_examples:
                if training:
                    rand_float = random.random()
                    null_masking_batch = bool(rand_float < 0.2)
                else:
                    null_masking_batch = True
            else:
                null_masking_batch = False

            # assert START is not in end
            assert not torch.any(
                after[:, :, 0] == self.start_tokens[0]
            ), "Start token located in after!"
            assert not (
                c_start_token and (start_token_l >= num_events_suffix)
            ), "Start token located in after"
            # assert END is not in the first local_window tokens
            assert not (
                c_end_token and (end_token_l < self.num_events_local_window)
            ), "End token located in local_window"

            ########################################################################
            # Construction du prefix
            if c_end_token and (end_token_l < num_events_suffix):
                # END is in before
                if self.reverse_prefix:
                    prefix = torch.cat(
                        [
                            self.pad_tokens.unsqueeze(0).repeat(a.size(0) - 1, 1),
                            self.end_tokens.unsqueeze(0),
                        ],
                        dim=0,
                    )
                else:
                    prefix = torch.cat(
                        [
                            self.end_tokens.unsqueeze(0),
                            self.pad_tokens.unsqueeze(0).repeat(a.size(0) - 1, 1),
                        ],
                        dim=0,
                    )

            else:
                if self.reverse_prefix:
                    prefix = self.reverse(a)
                else:
                    prefix = a
            ########################################################################

            ################################
            # Safeguards, can be removed after a while
            if torch.any(prefix[:, 0] == self.start_tokens[0]):
                # START in after
                raise Exception
            if torch.any(prefix[:, 0] == self.pad_tokens[0]):
                # PADS in after: there needs to be an END
                # and they have to appear after END
                assert torch.any(
                    prefix[:, 0] == self.end_tokens[0]
                ), "after contains PADS, but no END"
                pads_locations = torch.where(
                    prefix[:, 0] == self.pad_tokens[0].unsqueeze(0)
                )[0]
                end_location = torch.where(
                    prefix[:, 0] == self.end_tokens[0].unsqueeze(0)
                )[0]
                assert end_location.shape[0] == 1, "several END in suffix"
                if self.reverse_prefix:
                    assert torch.all(
                        pads_locations < end_location
                    ), "PADS before ENDS in after in reversed prefix"
                else:
                    assert torch.all(
                        pads_locations > end_location
                    ), "PADS before ENDS in after"
            ################################

            if null_masking_batch:
                null_tokens = self.pad_tokens.unsqueeze(0).repeat(prefix.size(0), 1)
                prefix_nonull = prefix
                prefix = null_tokens
            else:
                prefix_nonull = prefix

            ########################################################################
            # Construction du suffix
            if c_start_token and (start_token_l >= self.num_events_local_window):
                # START is in before, but not in the local window.
                # trim until START appears as the last element of the local window
                # (we don't want the model to predict START tokens)
                trim_begin = start_token_l - self.num_events_local_window + 1
                suffix = b[trim_begin:]
            elif c_end_token and (end_token_l < num_events_suffix):
                # END token is in before.
                # Remove END from suffix since it is appended later
                suffix = b[:end_token_l]
            else:
                suffix = b
            ########################################################################

            ################################
            # Safeguards, can be removed after running the code for a while with no exception
            if torch.any(suffix[:, 0] == self.start_tokens[0]):
                # check START position
                start_location = torch.where(
                    suffix[:, 0] == self.start_tokens[0].unsqueeze(0)
                )[0]
                assert start_location.shape[0] == 1, "several STARTS in suffix"
                assert (
                    start_location < self.num_events_local_window
                ), "START appears after local window"
            # No END in suffix (yet)
            is_end_token = torch.any(suffix[:, 0] == self.end_tokens[0])
            assert not is_end_token, "end token in suffix before end token is added"
            ################################

            # Now append end and pads
            num_events_pad_end = sequences_size - (
                self.num_events_context + len(suffix) + num_meta_events
            )
            assert num_events_pad_end > 0
            suffix = torch.cat(
                [
                    suffix,
                    self.end_tokens.unsqueeze(0),
                    self.pad_tokens.unsqueeze(0).repeat(num_events_pad_end, 1),
                ],
                dim=0,
            )

            if null_masking_batch:
                null_tokens = self.pad_tokens.unsqueeze(0).repeat(
                    self.num_events_local_window - 1, 1
                )
                suffix_nonull = suffix
                suffix[: self.num_events_local_window - 1] = null_tokens
            else:
                suffix_nonull = suffix

            assert len(prefix) + len(suffix) == sequences_size - 1
            prefix_list.append(prefix)
            suffix_list.append(suffix)
            prefix_list_nonull.append(prefix_nonull)
            suffix_list_nonull.append(suffix_nonull)

        prefix_tensor = torch.stack(prefix_list, dim=0)
        suffix_tensor = torch.stack(suffix_list, dim=0)
        prefix_tensor_nonull = torch.stack(prefix_list_nonull, dim=0)
        suffix_tensor_nonull = torch.stack(suffix_list_nonull, dim=0)
        sod = self.sod_symbols.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1)
        # creates final sequence
        y = torch.cat([prefix_tensor, sod, suffix_tensor], dim=1)
        y_nonull = torch.cat([prefix_tensor_nonull, sod, suffix_tensor_nonull], dim=1)

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
        final_mask = padding_mask + sod_mask + start_mask
        # add local windows, we only want "valid" local windows
        final_mask[:, : self.num_events_local_window, :] = True
        final_mask[
            :,
            self.num_events_context
            + 1 : self.num_events_context
            + 1
            + self.num_events_local_window,
            :,
        ] = True

        # decoding_start and decoding_end
        decoding_start = self.num_events_context + self.num_events_local_window + 1
        # valid_suffix_len = torch.where(suffix_tensor[:, :, 0] == self.end_tokens[0])[0][0] + 1
        # decoding_end = (self.num_events_context + 1 + valid_suffix_len)

        # self.num_events_before + self.num_events_after + 1 is the location
        # of the SOD symbol (only the placeholder is added)
        metadata_dict = {
            "placeholder_duration": placeholder_duration,
            "remaining_time": remaining_time,
            "decoding_start": decoding_start,
            # "decoding_end": None,
            "original_sequence": y,
            "original_sequence_nonull": y_nonull,
            "loss_mask": final_mask,
        }
        return y, metadata_dict

    def compute_elapsed_time(self, metadata_dict):
        # original sequence is in prefix order!
        x = metadata_dict["original_sequence_nonull"]
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

    def postprocess(self, x, decoding_end, metadata_dict):
        before = x[:, self.num_events_context + 1 :].to(self.end_tokens.device)

        # trim end
        num_events = before.shape[1]
        is_end_token = before[:, :, 0] == self.end_tokens[0].unsqueeze(0).unsqueeze(
            0
        ).repeat(1, num_events)
        contains_end_token = is_end_token.sum(1) == 1
        if contains_end_token:
            end_token_location = torch.argmax(is_end_token.long(), dim=1)
        else:
            if is_end_token.sum(1) > 1:
                raise Exception("more than 1 END token generated in suffix")
            else:
                raise Exception("no END token generated in suffix")
        before = before[:end_token_location]

        # trim start
        num_events = before.shape[1]
        is_start_token = before[:, :, 0] == self.start_tokens[0].unsqueeze(0).unsqueeze(
            0
        ).repeat(1, num_events)
        contains_start_token = is_start_token.sum(1) == 1
        if contains_start_token:
            start_token_location = torch.argmax(is_start_token.long(), dim=1)
            before = before[start_token_location + 1 :]

        after = x[:, : self.num_events_context].to(self.end_tokens.device)
        if self.reverse_prefix:
            after = self.dereverse(after)
        # trim end
        num_events = after.shape[1]
        is_end_token = after[:, :, 0] == self.end_tokens[0].unsqueeze(0).unsqueeze(
            0
        ).repeat(1, num_events)
        contains_end_token = is_end_token.sum(1) == 1
        if contains_end_token:
            end_token_location = torch.argmax(is_end_token.long(), dim=1)
            after = after[:, :end_token_location]

        # put all pieces in order
        x_out = torch.cat([before, after], dim=1)
        return x_out
