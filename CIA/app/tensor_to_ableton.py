import torch


def tensor_to_ableton(
    handler,
    tensor,
    insert_time,
    beats_per_second,
    notes_before,
    notes_after,
    expected_duration=None,
    rescale=False,
):
    """
    convert back a tensor to ableton format.
    Then shift all notes by clip start
    Args:
        tensor (num_events, num_channels)):
        clip_start
    """
    num_events, _ = tensor.size()
    if num_events == 0:
        return [], 0, insert_time

    notes_middle = []
    if tensor is not None:
        tensor = tensor.detach().cpu()
        index2value = handler.dataloader_generator.dataset.index2value

        timeshifts = torch.FloatTensor(
            [index2value["time_shift"][ts.item()] for ts in tensor[:, 3]]
        )

        # compute rescaling factor
        time = torch.cumsum(timeshifts, dim=0)
        actual_duration = time[-1].item()
        if rescale and actual_duration > 0:
            rescaling_factor = expected_duration / actual_duration
        else:
            rescaling_factor = 1

        # update time: convert to beats, takes into acount start_time, rescale
        time = (
            torch.cat([torch.zeros((1,)), time[:-1]], dim=0)
            * rescaling_factor
            * beats_per_second
            + insert_time
        )

        for i in range(num_events):
            note = dict(
                pitch=index2value["pitch"][tensor[i, 0].item()],
                time=time[i].item(),
                duration=index2value["duration"][tensor[i, 2].item()]
                * beats_per_second
                * rescaling_factor,
                velocity=index2value["velocity"][tensor[i, 1].item()],
                muted=0,
            )
            notes_middle.append(note)

        time_next_note = actual_duration * beats_per_second + insert_time
        print(f"DURATION: {actual_duration}")
        print(f"LAST timeshift out {timeshifts[-1]}")
    else:
        time_next_note = None

    new_before_notes = notes_before + notes_middle
    generated_notes = notes_middle
    if len(notes_after) > 0:
        track_duration = notes_after[-1]["time"] + notes_after[-1]["duration"]
    else:
        track_duration = new_before_notes[-1]["time"] + new_before_notes[-1]["duration"]

    return (
        new_before_notes,
        notes_after,
        generated_notes,
        track_duration,
        time_next_note,
    )
