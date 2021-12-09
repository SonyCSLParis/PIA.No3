import torch
import torch.nn as nn


class Perceiver(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.latents_init, self.dummy_latents = self._get_latents_init()
        self.write = self._get_write()
        self.read = self._get_read()
        self.process_x = self._get_process_x()
        self.process_l = self._get_process_l()
        self.last_layer_norm = self._get_last_layer_norm()

    def forward(self, x, **kwargs):
        batch_size, _, _ = x.size()

        # intialise the memory
        _, num_latents, latent_dim = self.latents_init.size()
        latents = self.latents_init.expand(batch_size, num_latents, latent_dim)

        for (write, read, process_x, process_l, dummy_l,) in zip(
            self.write,
            self.read,
            self.process_x,
            self.process_l,
            self.dummy_latents,
        ):
            # write
            latents = write(x, latents)
            # process
            x = process_x(x)
            latents = process_l(latents)
            dummy_l = dummy_l.repeat(batch_size, 1, 1)
            read_l = torch.cat([dummy_l, latents[:, :-1]], dim=1)
            # read
            x = read(x, read_l)
        if self.last_layer_norm is not None:
            x = self.last_layer_norm(x)
        return dict(x=x)
