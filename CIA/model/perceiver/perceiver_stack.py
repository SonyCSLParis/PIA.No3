import torch
import torch.nn as nn
from CIA.model.perceiver.perceiver_rw import Process_l
from CIA.model.perceiver.perceiver_rw import PerceiverReadWrite


class PerceiverReadWriteStack(PerceiverReadWrite):
    def __init__(
        self,
        dim,
        num_layers,
        num_heads,
        dropout,
        local_window_size,
        num_events,
        downscaling,
    ):
        super(PerceiverReadWriteStack, self).__init__(
            dim,
            num_layers,
            num_heads,
            dropout,
            local_window_size,
            num_events,
            downscaling,
        )

    # only difference is that you don't process latents at each layer of the stack
    def _get_process_l(self):
        return None

    def _get_last_layer_norm(self):
        return None


class PerceiverStack(nn.Module):
    def __init__(
        self,
        dim,
        num_layers,
        num_heads,
        dropout,
        local_window_size_l,
        num_events_l,
        downscaling_l,
    ):
        super(PerceiverStack, self).__init__()
        perceiver_stack = []
        for local_win_size, num_events, downscaling in zip(
            local_window_size_l, num_events_l, downscaling_l
        ):
            this_perceiver = PerceiverReadWriteStack(
                dim=dim,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
                local_window_size=local_win_size,
                num_events=num_events,
                downscaling=downscaling,
            )
            perceiver_stack.append(this_perceiver)
        self.perceiver_stack = nn.ModuleList(perceiver_stack)
        self.depth = len(self.perceiver_stack)
        self.num_layers = num_layers
        self.dim_last_layer = dim  # needed by handler
        self.process_last = nn.ModuleList(
            [
                Process_l(
                    dim=dim,
                    num_heads=num_heads,
                    hidden_dim=dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.last_layer_norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        batch_size, _, _ = x.size()
        # intialise the memories
        latents = []
        for depth in range(self.depth):
            _, num_latents, latent_dim = self.perceiver_stack[depth].latents_init.size()
            latents_depth = self.perceiver_stack[depth].latents_init.expand(
                batch_size, num_latents, latent_dim
            )
            latents.append(latents_depth)

        for layer in range(self.num_layers):
            # Write
            for depth in range(self.depth):
                perc = self.perceiver_stack[depth]
                if depth == 0:
                    previous_latents = x
                else:
                    previous_latents = latents[depth - 1]
                latents[depth] = perc.write[layer](previous_latents, latents[depth])

            # Process
            for depth in range(self.depth):
                perc = self.perceiver_stack[depth]
                if depth == 0:
                    x = perc.process_x[layer](x)
                else:
                    latents[depth - 1] = perc.process_x[layer](latents[depth - 1])
            latents[-1] = self.process_last[layer](latents[-1])

            # Read
            for depth_ in range(self.depth):
                depth = self.depth - 1 - depth_
                perc = self.perceiver_stack[depth]
                dummy_l = perc.dummy_latents[layer].repeat(batch_size, 1, 1)
                read_l = torch.cat([dummy_l, latents[depth][:, :-1]], dim=1)
                if depth == 0:
                    x = perc.read[layer](x, read_l)
                else:
                    latents[depth - 1] = perc.read[layer](latents[depth - 1], read_l)

        if self.last_layer_norm is not None:
            x = self.last_layer_norm(x)
        return dict(x=x)
