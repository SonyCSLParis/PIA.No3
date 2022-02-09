import importlib
import json
import os
from datetime import datetime

import click
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from flask import Flask, request
from flask.json import jsonify
from flask_cors import CORS
from torch.nn.parallel import DistributedDataParallel
from CIA.app.ableton_to_tensor import ableton_to_tensor
from CIA.app.json_to_tensor import json_to_tensor
from CIA.app.shorten_durations import shorten_durations

from CIA.app.tensor_to_ableton import tensor_to_ableton
from CIA.getters import (
    get_data_processor,
    get_dataloader_generator,
    get_decoder,
    get_handler,
    get_positional_embedding,
    get_sos_embedding,
)
from CIA.positional_embeddings import PositionalEmbedding
from CIA.utils import get_free_port

app = Flask(__name__)
CORS(app)
"""
@author: Gaetan Hadjeres
"""
DEBUG = False


@click.command()
@click.argument("cmd")
@click.option("-o", "--overfitted", is_flag=True)
@click.option("-c", "--config", type=click.Path(exists=True))
@click.option("-n", "--num_workers", type=int, default=0)
def launcher(cmd, overfitted, config, num_workers):
    # === Init process group
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(get_free_port())
    # === Set shared parameters
    # only use 1 GPU for inference
    print(cmd)
    assert cmd == "serve"
    world_size = 1

    # Load config as dict
    config_path = config
    config_module_name = os.path.splitext(config)[0].replace("/", ".")
    config = importlib.import_module(config_module_name).config

    # Compute time stamp
    if config["timestamp"] is not None:
        timestamp = config["timestamp"]
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        config["timestamp"] = timestamp

    # Create or retreive model_dir
    model_dir = os.path.dirname(config_path)

    print(f"Using {world_size} GPUs")
    mp.spawn(
        main,
        args=(overfitted, config, num_workers, world_size, model_dir),
        nprocs=world_size,
        join=True,
    )


def main(rank, overfitted, config, num_workers, world_size, model_dir):
    dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"

    # === Decoder ====
    # dataloader generator
    dataloader_generator = get_dataloader_generator(
        dataset=config["dataset"],
        dataloader_generator_kwargs=config["dataloader_generator_kwargs"],
    )

    # data processor
    global data_processor
    data_processor = get_data_processor(
        dataloader_generator=dataloader_generator,
        data_processor_type=config["data_processor_type"],
        data_processor_kwargs=config["data_processor_kwargs"],
    )

    # positional embedding
    positional_embedding: PositionalEmbedding = get_positional_embedding(
        dataloader_generator=dataloader_generator,
        positional_embedding_dict=config["positional_embedding_dict"],
    )

    # sos embedding
    sos_embedding = get_sos_embedding(
        dataloader_generator=dataloader_generator,
        sos_embedding_dict=config["sos_embedding_dict"],
    )

    decoder = get_decoder(
        data_processor=data_processor,
        dataloader_generator=dataloader_generator,
        positional_embedding=positional_embedding,
        sos_embedding=sos_embedding,
        decoder_kwargs=config["decoder_kwargs"],
        training_phase=False,
        handler_type=config["handler_type"],
    )

    decoder.to(device)
    decoder = DistributedDataParallel(
        module=decoder, device_ids=[rank], output_device=rank
    )

    global handler
    handler = get_handler(
        handler_type=config["handler_type"],
        decoder=decoder,
        model_dir=model_dir,
        dataloader_generator=dataloader_generator,
    )

    # Load
    if overfitted:
        handler.load(early_stopped=False)
    else:
        handler.load(early_stopped=True)

    local_only = False
    if local_only:
        # accessible only locally:
        app.run(threaded=True)
    else:
        # accessible from outside:
        port = 5000 if DEBUG else 8080

        app.run(
            host="0.0.0.0", port=port, threaded=True, debug=DEBUG, use_reloader=False
        )


@app.route("/ping", methods=["GET"])
def ping():
    return "pong"


@app.route("/invocations", methods=["POST"])
def invocations():
    # === Parse request ===
    # common components
    d = json.loads(request.data)
    case = d["case"]
    assert case in ["start", "continue"]
    if DEBUG:
        print(d)

    ############################################################
    ############################################################
    # REPRODUCE SAMPLING
    # Save or load random seed
    # np_rand_state = np.random.get_state()
    # now = datetime.today().strftime("%Y-%m-%d-%H:%M:%S")
    # with open(f"random_states/{now}.pickle", "wb") as handle:
    #     pkl.dump(np_rand_state, handle)

    # name = "save_random_state"
    # with open(f"random_states/{name}.pickle", "rb") as handle:
    #     np_rand_state = pkl.load(handle)
    # np.random.set_state(np_rand_state)
    ############################################################
    ############################################################

    notes = d["notes"]
    top_p = float(d["top_p"])
    superconditioning = float(d["superconditioning"])
    selected_region = d["selected_region"]
    if "clip_start" not in d:
        clip_start = selected_region["start"]
        d["clip_start"] = clip_start
    else:
        clip_start = d["clip_start"]
    print(selected_region, clip_start)

    tempo = d["tempo"]
    beats_per_second = tempo / 60
    seconds_per_beat = 1 / beats_per_second

    # two different parsing methods
    if case == "start":
        num_max_generated_events = 15
        (
            x,
            metadata_dict,
            clip_start,
            selected_region,
            region_after_start_time,
            regenerate_first_ts,
            before_notes,
            after_notes,
        ) = ableton_to_tensor(
            handler,
            data_processor,
            notes,
            clip_start,
            seconds_per_beat,
            selected_region,
        )
    elif case == "continue":
        num_max_generated_events = 20
        before_notes = d["notes_before_next_region"]
        after_notes = d["notes_after_region"]

        # selected_region MUST contain the right start!
        (x, metadata_dict, selected_region, region_after_start_time,) = json_to_tensor(
            handler,
            data_processor,
            before_notes,
            after_notes,
            seconds_per_beat,
            selected_region,
        )
        regenerate_first_ts = False
    else:
        raise NotImplementedError

    # network forward pass
    S = None if superconditioning == 1.0 else [superconditioning]
    if S is None:
        (_, generated_region, _, _, done,) = handler.inpaint_non_optimized(
            x=x,
            metadata_dict=metadata_dict,
            temperature=1.0,
            top_p=top_p,
            top_k=0,
            num_max_generated_events=num_max_generated_events,
            regenerate_first_ts=regenerate_first_ts,
        )
    else:
        (
            _,
            generated_region,
            _,
            _,
            done,
        ) = handler.inpaint_non_optimized_superconditioning(
            x=x,
            metadata_dict=metadata_dict,
            temperature=1.0,
            top_p=top_p,
            top_k=0,
            num_max_generated_events=num_max_generated_events,
            regenerate_first_ts=regenerate_first_ts,
            null_superconditioning=S,
        )

    # convert to ableton format
    (
        new_before_notes,
        new_after_notes,
        generated_notes,
        track_duration,
        time_next_note_before,
    ) = tensor_to_ableton(
        handler=handler,
        tensor=generated_region[0],
        notes_before=before_notes,
        notes_after=after_notes,
        insert_time=selected_region["start"],
        beats_per_second=beats_per_second,
        rescale=False,
    )

    # shorten notes which encroach on notes in the "after" segment
    ableton_notes_region = shorten_durations(generated_notes, new_after_notes)
    selected_region["start"] = time_next_note_before
    notes = new_before_notes + new_after_notes

    # format before sending back
    d = {
        "id": d["id"],
        "notes": notes,
        "track_duration": track_duration,
        "done": done,
        "top_p": top_p,
        "superconditioning": superconditioning,
        "selected_region": selected_region,
        "notes_before_next_region": new_before_notes,
        "notes_region": ableton_notes_region,
        "notes_after_region": new_after_notes,
        "clip_start": clip_start,
        "clip_id": d["clip_id"],
        "clip_end": d["clip_end"],
        "detail_clip_id": d["detail_clip_id"],
        "tempo": d["tempo"],
    }
    return jsonify(d)


if __name__ == "__main__":
    launcher()

# Response format
# {'id': '14', 'notes': ['notes', 10, 'note', 64, 0.5, 0.25, 100, 0, 'note', 64, 0.75, 0.25, 100, 0, 'note', 64, 1, 0.25, 100, 0, 'note', 65, 0.25, 0.25, 100, 0, 'note', 68, 1, 0.25, 100, 0, 'note', 69, 0, 0.25, 100, 0, 'note', 69, 0.75, 0.25, 100, 0, 'note', 69, 1.25, 2, 100, 0, 'note', 70, 0.5, 0.25, 100, 0, 'note', 71, 0.25, 0.25, 100, 0, 'done'], 'duration': 4}
