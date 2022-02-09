import torch

from CIA.utils import cuda_variable


def json_to_tensor(
    handler,
    data_processor,
    notes_before_json,
    notes_after_json,
    seconds_per_beat,
    selected_region=None,
):
    """[summary]
    Args:
        ableton_note_list ([type]): [description]
        note_density ([type]): [description]
        selected_region ([type], optional): [description]. Defaults to None.
    Returns:
        x [type]: x is at least of size (1024, 4), it is padded if necessary
    """
    # Parse before:
    features_before = {
        "pitch": [],
        "time": [],
        "duration": [],
        "velocity": [],
        "muted": [],
    }

    for n in notes_before_json:
        for k, v in n.items():
            features_before[k].append(v)

    # we now have to sort
    before_list_notes = [
        [p, t, d, v]
        for p, t, d, v in zip(
            features_before["pitch"],
            features_before["time"],
            features_before["duration"],
            features_before["velocity"],
        )
    ]
    before_list_notes = sorted(before_list_notes, key=lambda x: (x[1], -x[0]))

    features_before = dict(
        pitch=torch.LongTensor([x[0] for x in before_list_notes]),
        time=torch.FloatTensor([x[1] for x in before_list_notes]),
        duration=torch.FloatTensor([max(float(x[2]), 0.05) for x in before_list_notes]),
        velocity=torch.LongTensor([x[3] for x in before_list_notes]),
    )

    # Parse after
    features_after = {
        "pitch": [],
        "time": [],
        "duration": [],
        "velocity": [],
        "muted": [],
    }

    for n in notes_after_json:
        for k, v in n.items():
            features_after[k].append(v)

    # we now have to sort
    after_list_notes = [
        [p, t, d, v]
        for p, t, d, v in zip(
            features_after["pitch"],
            features_after["time"],
            features_after["duration"],
            features_after["velocity"],
        )
    ]

    after_list_notes = sorted(after_list_notes, key=lambda x: (x[1], -x[0]))

    features_after = dict(
        pitch=torch.LongTensor([x[0] for x in after_list_notes]),
        time=torch.FloatTensor([x[1] for x in after_list_notes]),
        duration=torch.FloatTensor([max(float(x[2]), 0.05) for x in after_list_notes]),
        velocity=torch.LongTensor([x[3] for x in after_list_notes]),
    )

    # time in beats
    region_after_start_time = selected_region["end"]
    last_time_shift_before = (
        max((selected_region["start"] - features_before["time"][-1].item()), 0)
        * seconds_per_beat
    )

    print(f"LAST TIMESHIFT IN {last_time_shift_before}")

    # multiply by tempo
    features_before["time"] = features_before["time"] * seconds_per_beat
    features_before["duration"] = features_before["duration"] * seconds_per_beat

    # compute time_shift
    features_before["time_shift"] = torch.cat(
        [
            features_before["time"][1:] - features_before["time"][:-1],
            torch.tensor([last_time_shift_before]),
        ],
        dim=0,
    )

    # same for "after"
    features_after["time"] = features_after["time"] * seconds_per_beat
    features_after["duration"] = features_after["duration"] * seconds_per_beat

    # compute time_shift
    num_notes_after = len(features_after["pitch"])
    if num_notes_after > 0:
        features_after["time_shift"] = torch.cat(
            [
                features_after["time"][1:] - features_after["time"][:-1],
                torch.zeros(
                    1,
                ),
            ],
            dim=0,
        )
    else:
        features_after["time_shift"] = torch.tensor([])

    # compute placeholder
    # selected region already contains exactly the correct durations

    placeholder_duration = (
        selected_region["end"] - selected_region["start"]
    ) * seconds_per_beat
    placeholder_duration = cuda_variable(torch.Tensor([placeholder_duration]))

    placeholder, _ = data_processor.compute_placeholder(
        placeholder_duration=placeholder_duration, batch_size=1
    )

    event_start = len(features_before["time"])

    # delete unnecessary entries in dict
    del features_before["time"]
    del features_after["time"]
    before = {k: v for k, v in features_before.items()}
    after = {k: v for k, v in features_after.items()}

    # format and pad
    # If we need to pad "before"
    if event_start < data_processor.num_events_before:
        before = {k: t.numpy() for k, t in before.items()}
        before = handler.dataloader_generator.dataset.add_start_end_symbols(
            sequence=before,
            start_time=event_start - data_processor.num_events_before,
            sequence_size=data_processor.num_events_before,
        )

        before = handler.dataloader_generator.dataset.tokenize(before)
        before = {k: torch.LongTensor(t) for k, t in before.items()}
        before = torch.stack(
            [before[e] for e in handler.dataloader_generator.features], dim=-1
        ).long()

        before = cuda_variable(before)
    else:
        before = {k: t.numpy() for k, t in before.items()}
        before = handler.dataloader_generator.dataset.add_start_end_symbols(
            sequence=before, start_time=0, sequence_size=event_start
        )
        before = handler.dataloader_generator.dataset.tokenize(before)
        before = {k: torch.LongTensor(t) for k, t in before.items()}
        before = torch.stack(
            [before[e] for e in handler.dataloader_generator.features], dim=-1
        ).long()
        before = cuda_variable(before)

        before = before[-data_processor.num_events_before :]

    # After cannot contain 'START' symbol
    after = {k: t.numpy() for k, t in after.items()}
    after = handler.dataloader_generator.dataset.add_start_end_symbols(
        sequence=after,
        start_time=0,
        sequence_size=max(num_notes_after, data_processor.num_events_after),
    )

    after = handler.dataloader_generator.dataset.tokenize(after)
    after = {k: torch.LongTensor(t) for k, t in after.items()}
    after = torch.stack(
        [after[e] for e in handler.dataloader_generator.features], dim=-1
    ).long()
    after = cuda_variable(after)

    after = after[: data_processor.num_events_after]

    middle_length = (
        data_processor.dataloader_generator.sequences_size
        - data_processor.num_events_before
        - data_processor.num_events_after
        - 2
    )

    # add batch dim
    before = before.unsqueeze(0)
    after = after.unsqueeze(0)

    # create x:
    x = torch.cat(
        [
            before,
            placeholder,
            after,
            data_processor.sod_symbols.unsqueeze(0).unsqueeze(0),
            cuda_variable(torch.zeros(1, middle_length, data_processor.num_channels)),
        ],
        dim=1,
    ).long()

    # if "before" was padded
    if event_start < data_processor.num_events_before:
        # (then event_start is the size of "before")
        before = before[:, -event_start:]

        # slicing does not work in this case
        if event_start == 0:
            before = before[:, :0]

    # if "after" was padded:
    if num_notes_after < data_processor.num_events_after:
        after = after[:, :num_notes_after]

    metadata_dict = dict(
        original_sequence=x,
        placeholder_duration=placeholder_duration,
        decoding_start=data_processor.num_events_before
        + data_processor.num_events_after
        + 2,
    )

    return (x, metadata_dict, selected_region, region_after_start_time)
