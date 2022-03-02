from pathlib import Path

sequences_size = 1024
num_events_context = sequences_size // 4
num_event_local_window = 64
downscale = 16
offset_beginning = sequences_size - sequences_size // 4
offset_end = sequences_size - offset_beginning
config = {
    "dataset": "piano",
    # --- Dataloader ---
    "dataloader_generator_kwargs": dict(
        sequences_size=sequences_size,
        transformations={
            "time_dilation": True,
            "velocity_shift": True,
            "transposition": True,
        },
        offset_beginning=-offset_beginning,
        offset_end=-offset_end,
    ),
    # --- DataProcessor ---
    "data_processor_kwargs": dict(
        embedding_size=64,
        num_events_local_window=num_event_local_window,
        num_events_context=num_events_context,
        reverse_prefix=False,  # only for prefixEnd
    ),  # Can be different from the encoder's data processor
    # --- Positional Embedding ---
    "positional_embedding_dict": dict(
        sinusoidal_embedding=dict(
            positional_embedding_size=128,
            num_channels=4,
            dropout=0.0,
            expand_channels=False,
        ),
        sinusoidal_elapsed_time_embedding=dict(
            positional_embedding_size=128,
            num_channels=4,
            dropout=0.0,
            mask_positions=False,
            expand_channels=False,
        ),
        sinusoidal_remaining_time_embedding=dict(
            positional_embedding_size=128,
            num_channels=4,
            dropout=0.0,
            expand_channels=False,
        ),
    ),
    # --- Start Of Sequence embeddings
    "sos_embedding_dict": dict(
        learnt_sos_embedding=dict(
            embedding_size=512  # sum must be equal to d_model_decoder
        )
    ),
    # --- Model ---
    "model_kwargs": dict(
        type="perceiver_rw",
        d_model=512,
        n_head=8,
        num_layers=12,
        dropout=0.1,
        local_window_size=4 * downscale,
        downscaling=downscale,
        relative_pos_bias=True,
    ),
    # ======== Training ========
    "compute_loss_prefix": True,
    "non_conditioned_examples": True,
    "lr": 1e-4,
    "batch_size": 6,
    "num_batches": 64,
    "num_epochs": 1500000,
    # ======== model ID ========
    "timestamp": None,
    "savename": Path(__file__).stem,
}
