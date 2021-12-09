from pathlib import Path

known_suffix_length = 64
downscale = 16
num_events_context = 256
config = {
    "dataset": "piano",
    # --- Dataloader ---
    "dataloader_generator_kwargs": dict(
        sequences_size=1024,
        transformations={
            "time_dilation": True,
            "velocity_shift": True,
            "transposition": True,
        },
        offset_beginning=-(known_suffix_length - 1),
        offset_end=-known_suffix_length,
    ),
    # --- DataProcessor ---
    "data_processor_kwargs": dict(
        embedding_size=64,
        num_events_local_window=known_suffix_length,
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
        type="perceiver_stack",
        autoregressive_decoding="fullcat",  # fullcat | mlp | None
        d_model=512,
        n_head=8,
        num_layers=12,
        dropout=0.1,
        downscaling_l=[
            downscale,
            downscale,
        ],
        local_window_size_l=[
            2 * downscale,
            2 * downscale,
        ],
    ),
    # ======== Training ========
    "compute_loss_prefix": True,
    "lr": 1e-4,
    "batch_size": 7,
    "num_batches": 64,
    "num_epochs": 1500000,
    # ======== model ID ========
    "timestamp": None,
    "savename": Path(__file__).stem,
}
