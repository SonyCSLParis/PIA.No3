from CIA.data_processors.piano_prefixEnd_data_processor import (
    PianoPrefixEndDataProcessor,
)
from CIA.dataloaders.piano_dataloader import PianoDataloaderGenerator
from CIA.handlers.events_handler import EventsHandler
from CIA.model.causal_events_model_full_cat import CausalEventsModelFullCat
from CIA.model.perceiver.perceiver_rw import PerceiverReadWrite
from CIA.model.perceiver.perceiver_stack import PerceiverStack
from CIA.model.perceiver.perceiver_tower import PerceiverTower
from CIA.positional_embeddings.positional_embedding import PositionalEmbedding
from CIA.positional_embeddings.sinusoidal_elapsed_time_embedding import (
    SinusoidalElapsedTimeEmbedding,
)
from CIA.positional_embeddings.sinusoidal_positional_embedding import (
    SinusoidalPositionalEmbedding,
)
from CIA.positional_embeddings.sinusoidal_remaining_time_embedding import (
    SinusoidalRemainingTimeEmbedding,
)
from CIA.start_of_sequence_embeddings import (
    BaseSOSEmbedding,
    LearntSOSEmbedding,
    SOSEmbedding,
)


def get_data_processing(dataset, dataloader_generator_kwargs, data_processor_kwargs):
    if dataset == "piano":
        dataloader_generator = PianoDataloaderGenerator(
            sequences_size=dataloader_generator_kwargs["sequences_size"],
            transformations=dataloader_generator_kwargs["transformations"],
            offset_beginning=dataloader_generator_kwargs["offset_beginning"],
            offset_end=dataloader_generator_kwargs["offset_end"],
            num_elements=None,
        )
        num_events = dataloader_generator.sequences_size
        value2index = dataloader_generator.dataset.value2index
        num_tokens_per_channel = [
            len(value2index[feature]) for feature in dataloader_generator.features
        ]
        data_processor = PianoPrefixEndDataProcessor(
            dataloader_generator=dataloader_generator,
            embedding_size=data_processor_kwargs["embedding_size"],
            num_events=num_events,
            num_events_local_window=data_processor_kwargs["num_events_local_window"],
            num_events_context=data_processor_kwargs["num_events_context"],
            num_tokens_per_channel=num_tokens_per_channel,
            reverse_prefix=data_processor_kwargs["reverse_prefix"],
        )
        return dataloader_generator, data_processor


def get_positional_embedding(
    dataloader_generator, data_processor, positional_embedding_dict
):
    base_positional_embedding_list = []
    for pe_name, pe_kwargs in positional_embedding_dict.items():
        if pe_name == "sinusoidal_embedding":
            # compute num_tokens_max:
            num_tokens_max = dataloader_generator.sequences_size + 1
            base_pe = SinusoidalPositionalEmbedding(
                positional_embedding_size=pe_kwargs["positional_embedding_size"],
                num_tokens_max=num_tokens_max,
                num_channels=pe_kwargs["num_channels"],
                dropout=pe_kwargs["dropout"],
                expand_channels=pe_kwargs["expand_channels"],
            )
        elif pe_name == "sinusoidal_elapsed_time_embedding":
            base_pe = SinusoidalElapsedTimeEmbedding(
                dataloader_generator=dataloader_generator,
                data_processor=data_processor,
                **pe_kwargs
            )
        elif pe_name == "sinusoidal_remaining_time_embedding":
            base_pe = SinusoidalRemainingTimeEmbedding(
                dataloader_generator=dataloader_generator,
                data_processor=data_processor,
                **pe_kwargs
            )
        else:
            raise NotImplementedError
        base_positional_embedding_list.append(base_pe)

    return PositionalEmbedding(
        base_positional_embedding_list=base_positional_embedding_list
    )


# todo write Decoder base class
def get_model(
    data_processor,
    dataloader_generator,
    positional_embedding,
    sos_embedding,
    model_kwargs,
):
    num_channels = data_processor.num_channels
    num_events = data_processor.num_events
    if (
        model_kwargs["type"] == "perceiver_rw"
    ):  # Diff with perceiver_tower is that there processing of latent is shallow compared with tower
        transformer = PerceiverReadWrite(
            dim=model_kwargs["d_model"],
            num_layers=model_kwargs["num_layers"],
            num_heads=model_kwargs["n_head"],
            dropout=model_kwargs["dropout"],
            local_window_size=model_kwargs["local_window_size"],
            num_events=num_events,
            downscaling=model_kwargs["downscaling"],
        )
    elif model_kwargs["type"] == "perceiver_stack":
        num_events_decoder_layer = num_events
        num_events_decoder_l = []
        local_window_size_l = []
        for downscaling in model_kwargs["downscaling_l"]:
            num_events_decoder_l.append(num_events_decoder_layer)
            local_window_size_l.append(downscaling)
            assert num_events_decoder_layer % downscaling == 0
            num_events_decoder_layer = num_events_decoder_layer // downscaling
        transformer = PerceiverStack(
            dim=model_kwargs["d_model"],
            num_layers=model_kwargs["num_layers"],
            num_heads=model_kwargs["n_head"],
            dropout=model_kwargs["dropout"],
            local_window_size_l=local_window_size_l,
            num_events_l=num_events_decoder_l,
            downscaling_l=model_kwargs["downscaling_l"],
        )
    elif model_kwargs["type"] == "perceiver_tower":
        transformer = PerceiverTower(
            dim=model_kwargs["d_model"],
            num_layers=model_kwargs["num_layers"],
            tower_depth=model_kwargs["tower_depth"],
            num_heads=model_kwargs["n_head"],
            dropout=model_kwargs["dropout"],
            local_window_size=model_kwargs["local_window_size"],
            num_events=num_events,
            downscaling=model_kwargs["downscaling"],
        )
    else:
        raise NotImplementedError

    model = CausalEventsModelFullCat(
        data_processor=data_processor,
        dataloader_generator=dataloader_generator,
        positional_embedding=positional_embedding,
        sos_embedding=sos_embedding,
        d_model=model_kwargs["d_model"],
        num_channels=num_channels,
        num_events=num_events,
        transformer=transformer,
    )
    return model


def get_sos_embedding(dataloader_generator, sos_embedding_dict) -> SOSEmbedding:
    base_sos_embedding_list = []
    for sos_name, sos_kwargs in sos_embedding_dict.items():
        if sos_name == "learnt_sos_embedding":
            base_sos: BaseSOSEmbedding = LearntSOSEmbedding(
                embedding_size=sos_kwargs["embedding_size"]
            )
        else:
            raise NotImplementedError
        base_sos_embedding_list.append(base_sos)

    return SOSEmbedding(base_sos_embedding_list=base_sos_embedding_list)


def get_handler(model, model_dir, dataloader_generator):
    return EventsHandler(
        model=model,
        model_dir=model_dir,
        dataloader_generator=dataloader_generator,
    )
