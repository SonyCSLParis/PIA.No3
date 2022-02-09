import torch

from CIA.utils import cuda_variable


def ableton_to_tensor(
    handler,
    data_processor,
    ableton_note_list,
    clip_start,
    seconds_per_beat,
    selected_region=None,
):
    """[summary]
    Args:
        ableton_note_list ([type]): [description]
        note_density ([type]): [description]
        clip_start ([type]): [description]
        selected_region ([type], optional): [description]. Defaults to None.
    Returns:
        x [type]: x is at least of size (1024, 4), it is padded if necessary
    """
    ableton_d = {
        "pitch": [],
        "time": [],
        "duration": [],
        "velocity": [],
        "muted": [],
    }
    mod = -1

    # pitch time duration velocity muted
    ableton_features = ["pitch", "time", "duration", "velocity", "muted"]

    if selected_region is not None:
        start_time = selected_region["start"]
        end_time = selected_region["end"]

    for msg in ableton_note_list:
        if msg == "notes":
            pass
        elif msg == "note":
            mod = 0
        elif msg == "done":
            break
        else:
            if mod >= 0:
                ableton_d[ableton_features[mod]].append(msg)
                mod = (mod + 1) % 5

    # we now have to sort
    features_l = [
        [p, t, d, v]
        for p, t, d, v in zip(
            ableton_d["pitch"],
            ableton_d["time"],
            ableton_d["duration"],
            ableton_d["velocity"],
        )
    ]

    features_l = sorted(features_l, key=lambda x: (x[1], -x[0]))

    features = dict(
        pitch=torch.LongTensor([x[0] for x in features_l]),
        time=torch.FloatTensor([x[1] for x in features_l]),
        duration=torch.FloatTensor([max(float(x[2]), 0.05) for x in features_l]),
        velocity=torch.LongTensor([x[3] for x in features_l]),
    )

    # compute event_start, event_end
    # num_notes is the number of notes in the original sequence
    epsilon = 1e-4
    num_notes = features["time"].size(0)

    event_start, event_end = None, None
    if selected_region is not None:
        i = 0
        flag = True
        while flag:
            if i == num_notes:
                event_start = num_notes
                break
            if features["time"][i].item() >= start_time - epsilon:
                flag = False
                event_start = i
            else:
                i = i + 1

        i = 0
        flag = True
        while flag:
            if i == num_notes:
                event_end = num_notes
                break
            if i > features["time"].size(0):
                flag = False
                event_end = i
            elif features["time"][i].item() >= end_time - epsilon:
                flag = False
                event_end = i
            else:
                i = i + 1

    # if no region after
    if event_end == len(features["time"]):
        region_after_start_time = selected_region["end"]
    else:
        # time in beats
        region_after_start_time = features["time"][event_end].item()
        end_time = features["time"][event_end].item()

    # if there is at least ONE note before the region
    # start_time becomes the start_time of this note
    # its timeshift will have to be recomputed
    if event_start > 0:
        regenerate_first_ts = True
        event_start = event_start - 1
        start_time = features["time"][event_start].item()
        clip_start = features["time"][0].item()  # in beats still
    else:
        # If no notes before, start_time also defines the clip_start as
        # clip_start is incorrect
        regenerate_first_ts = False
        clip_start = start_time

    # update selected region with the newly computed region
    selected_region["end"] = end_time
    selected_region["start"] = start_time

    # keep track of the list of notes
    before_notes = []
    for note in features_l[:event_start]:
        note_d = dict(
            pitch=note[0],
            time=note[1],
            duration=note[2],
            velocity=note[3],
            muted=0,
        )
        before_notes.append(note_d)
    after_notes = []
    for note in features_l[event_end:]:
        note_d = dict(
            pitch=note[0],
            time=note[1],
            duration=note[2],
            velocity=note[3],
            muted=0,
        )
        after_notes.append(note_d)

    # ============== to seconds ==================
    # multiply by tempo
    features["time"] = features["time"] * seconds_per_beat
    features["duration"] = features["duration"] * seconds_per_beat

    # compute time_shift
    features["time_shift"] = torch.cat(
        [
            features["time"][1:] - features["time"][:-1],
            torch.zeros(
                1,
            ),
        ],
        dim=0,
    )

    placeholder_duration = (end_time - start_time) * seconds_per_beat
    placeholder_duration = cuda_variable(torch.Tensor([placeholder_duration]))

    placeholder, _ = data_processor.compute_placeholder(
        placeholder_duration=placeholder_duration, batch_size=1
    )

    # delete unnecessary entries in dict
    del features["time"]
    before = {k: v[:event_start] for k, v in features.items()}
    if regenerate_first_ts:
        first_note = {k: v[event_start : event_start + 1] for k, v in features.items()}
    after = {k: v[event_end:] for k, v in features.items()}

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

    # same for "after"
    num_notes_after = after["pitch"].size(0)

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

    # first note as tensor:
    if regenerate_first_ts:
        first_note = {k: t.numpy() for k, t in first_note.items()}
        first_note = handler.dataloader_generator.dataset.tokenize(first_note)
        first_note = {k: torch.LongTensor(t) for k, t in first_note.items()}
        first_note = torch.stack(
            [first_note[e] for e in handler.dataloader_generator.features], dim=-1
        ).long()
        first_note = cuda_variable(first_note)

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
    # Warning special case where we need to specify the first note
    if regenerate_first_ts:
        x = torch.cat(
            [
                before,
                placeholder,
                after,
                data_processor.sod_symbols.unsqueeze(0).unsqueeze(0),
                first_note.unsqueeze(0),
                cuda_variable(
                    torch.zeros(1, middle_length - 1, data_processor.num_channels)
                ),
            ],
            dim=1,
        ).long()
    else:
        x = torch.cat(
            [
                before,
                placeholder,
                after,
                data_processor.sod_symbols.unsqueeze(0).unsqueeze(0),
                cuda_variable(
                    torch.zeros(1, middle_length, data_processor.num_channels)
                ),
            ],
            dim=1,
        ).long()

    # update clip start if necessary
    if clip_start > start_time:
        clip_start = start_time

    metadata_dict = dict(
        original_sequence=x,
        placeholder_duration=placeholder_duration,
        decoding_start=data_processor.num_events_before
        + data_processor.num_events_after
        + 2,
    )

    return (
        x,
        metadata_dict,
        clip_start,
        selected_region,
        region_after_start_time,
        regenerate_first_ts,
        before_notes,
        after_notes,
    )
