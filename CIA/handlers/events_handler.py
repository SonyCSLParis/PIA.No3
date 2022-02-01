from CIA.handlers.handler import Handler
from CIA.dataloaders.dataloader import DataloaderGenerator
from CIA.utils import (
    all_reduce_scalar,
    is_main_process,
    to_numpy,
    top_k_top_p_filtering,
)
import torch
from tqdm import tqdm
from itertools import islice
import numpy as np
from torch.nn.parallel import DistributedDataParallel


# TODO duplicated code with decoder_prefix_handler.py
class EventsHandler(Handler):
    def __init__(
        self,
        model: DistributedDataParallel,
        model_dir: str,
        dataloader_generator: DataloaderGenerator,
    ) -> None:
        super().__init__(
            model=model, model_dir=model_dir, dataloader_generator=dataloader_generator
        )

    # --- EventsHandler-specific wrappers
    def event_state_to_weight_step(self, output, target_embedded, channel_index):
        return self.model.module.event_state_to_weight_step(
            output, target_embedded, channel_index
        )

    def compute_event_state(self, target, metadata_dict):
        return self.model.module.compute_event_state(target, metadata_dict)

    # ==== Training methods

    def epoch(
        self,
        data_loader,
        train,
        num_batches,
        compute_loss_prefix,
        non_conditioned_examples,
    ):
        means = None
        lists = None

        if train:
            self.train()
        else:
            self.eval()

        iterator = enumerate(islice(data_loader, num_batches))
        if is_main_process():
            iterator = tqdm(iterator, ncols=80)

        for sample_id, tensor_dict in iterator:

            # ==========================
            with torch.no_grad():
                x = tensor_dict["x"]
                x, metadata_dict = self.data_processor.preprocess(
                    x,
                    num_events_inpainted=None,
                    training=True,
                    non_conditioned_examples=non_conditioned_examples,
                )

            # ========Train decoder =============
            self.optimizer.zero_grad()
            forward_pass = self.forward(
                target=x,
                metadata_dict=metadata_dict,
                compute_loss_prefix=compute_loss_prefix,
            )
            loss = forward_pass["loss"]

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 5)
                self.optimizer.step()

            # Monitored quantities
            monitored_quantities = forward_pass["monitored_quantities"]

            # average quantities
            if means is None:
                means = {key: 0 for key in monitored_quantities}
            means = {
                key: value + means[key] for key, value in monitored_quantities.items()
            }
            del loss

            if lists is None:
                lists = dict(prefix_len=[], suffix_len=[])
            lists["prefix_len"] += metadata_dict["prefix_len"]
            lists["suffix_len"] += metadata_dict["suffix_len"]

        # renormalize monitored quantities
        for key, value in means.items():
            means[key] = all_reduce_scalar(value, average=True) / (sample_id + 1)

        return means, lists

    def inpaint(
        self,
        x,
        metadata_dict,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        num_max_generated_events=None,
    ):
        # TODO add arguments to preprocess
        print(f'Placeholder duration: {metadata_dict["placeholder_duration"]}')
        self.eval()
        batch_size, num_events, _ = x.size()

        # TODO only works with batch_size=1 at present
        assert x.size(0) == 1

        index2value = self.dataloader_generator.dataset.index2value
        decoding_end = None
        placeholder_duration = metadata_dict["placeholder_duration"].item()
        decoding_start_event = metadata_dict["decoding_start"]
        generated_duration = 0.0
        x[:, decoding_start_event:] = 0

        if num_max_generated_events is not None:
            num_events = min(
                decoding_start_event + num_max_generated_events, num_events
            )
        with torch.no_grad():
            # event_index corresponds to the position of the token BEING generated
            for event_index in range(decoding_start_event, num_events):
                metadata_dict["original_sequence"] = x

                # output is used to generate auto-regressively all
                # channels of an event
                output, target_embedded = self.compute_event_state(
                    target=x,
                    metadata_dict=metadata_dict,
                )

                # extract correct event_step
                output = output[:, event_index]

                for channel_index in range(self.num_channels_target):
                    # target_embedded must be recomputed!
                    # TODO could be optimized
                    target_embedded = self.data_processor.embed(x)[:, event_index]
                    weights = self.event_state_to_weight_step(
                        output, target_embedded, channel_index
                    )
                    logits = weights / temperature

                    filtered_logits = []
                    for logit in logits:
                        filter_logit = top_k_top_p_filtering(
                            logit, top_k=top_k, top_p=top_p
                        )
                        filtered_logits.append(filter_logit)
                    filtered_logits = torch.stack(filtered_logits, dim=0)
                    # Sample from the filtered distribution
                    p = to_numpy(torch.softmax(filtered_logits, dim=-1))

                    # update generated sequence
                    for batch_index in range(batch_size):
                        if event_index >= decoding_start_event:
                            new_pitch_index = np.random.choice(
                                np.arange(
                                    self.num_tokens_per_channel_target[channel_index]
                                ),
                                p=p[batch_index],
                            )
                            x[batch_index, event_index, channel_index] = int(
                                new_pitch_index
                            )

                            end_symbol_index = (
                                self.dataloader_generator.dataset.value2index[
                                    self.dataloader_generator.features[channel_index]
                                ]["END"]
                            )
                            if end_symbol_index == int(new_pitch_index):
                                decoding_end = event_index
                                print("End of decoding due to END symbol generation")

                            ###############################################
                            # Additional check: if the generated duration is > than the placeholder_duration
                            # TODO hardcoded channel index for timeshifts
                            if channel_index == 3:
                                generated_duration += index2value["time_shift"][
                                    new_pitch_index
                                ]
                                # TODO: Change that, placeholder does not exists anyumore
                                raise NotImplementedError(
                                    "Change that, placeholder does not exists anyumore"
                                )
                                if generated_duration > placeholder_duration:
                                    decoding_end = event_index + 1
                                    for channel_index_local in range(
                                        self.num_channels_target
                                    ):
                                        end_symbol_index = self.dataloader_generator.dataset.value2index[
                                            self.dataloader_generator.features[
                                                channel_index_local
                                            ]
                                        ][
                                            "END"
                                        ]
                                        x[
                                            batch_index,
                                            decoding_end,
                                            channel_index_local,
                                        ] = int(end_symbol_index)
                                    print(
                                        "End of decoding due to the generation > than placeholder duration"
                                    )
                                    print(
                                        f"Excess: {generated_duration - placeholder_duration}"
                                    )
                            ###############################################

                    if decoding_end is not None:
                        break
                if decoding_end is not None:
                    break
            if decoding_end is None:
                # replace last event by an end event
                for batch_index in range(batch_size):
                    for channel_index in range(self.num_channels_target):
                        end_symbol_index = (
                            self.dataloader_generator.dataset.value2index[
                                self.dataloader_generator.features[channel_index]
                            ]["END"]
                        )
                        x[batch_index, event_index, channel_index] = int(
                            end_symbol_index
                        )
                done = False
                decoding_end = num_events
            else:
                done = True
        num_event_generated = decoding_end - decoding_start_event
        generated_region = x[:, decoding_start_event:decoding_end]
        return x.cpu(), generated_region, decoding_end, num_event_generated, done
