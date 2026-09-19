"""Small from-scratch image baseline, independent of the external GPE model."""

import torch
from torch import nn

from src.utils.time import batch_time


class FrameAutoencoder(nn.Module):
    def __init__(self, image_size=32, latent_dim=8, width=128):
        super().__init__()
        self.image_size = image_size
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(image_size**2, width),
                                     nn.SiLU(), nn.Linear(width, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, width), nn.SiLU(),
                                     nn.Linear(width, image_size**2), nn.Tanh())

    def encode(self, frames):
        return self.encoder(frames)

    def decode(self, latent):
        return self.decoder(latent).reshape(-1, 1, self.image_size, self.image_size)

    def forward(self, frames):
        return self.decode(self.encode(frames))


class FrameLatentVelocity(nn.Module):
    def __init__(self, seq_length=2, latent_dim=8, width=128, time_scale=30.0):
        super().__init__()
        if time_scale <= 0:
            raise ValueError("time_scale must be positive.")
        self.time_scale = time_scale
        self.network = nn.Sequential(nn.Linear(seq_length * latent_dim + 3, width),
                                     nn.SiLU(), nn.Linear(width, width), nn.SiLU(),
                                     nn.Linear(width, latent_dim))

    def forward(self, sequence, time):
        t = batch_time(time, sequence) / self.time_scale
        features = torch.stack((t, t.sin(), t.cos()), dim=1)
        return self.network(torch.cat((sequence.flatten(1), features), dim=1))


class AcrobotFrameModel(nn.Module):
    def __init__(self, image_size=32, seq_length=2, latent_dim=8, width=128, time_scale=30.0, codec=None):
        super().__init__()
        self.codec = codec if codec is not None else FrameAutoencoder(image_size, latent_dim, width)
        self.velocity = FrameLatentVelocity(seq_length, latent_dim, width, time_scale)

    def encode_sequence(self, frames):
        b, t, c, h, w = frames.shape
        return self.codec.encode(frames.reshape(b*t, c, h, w)).reshape(b, t, -1)

    def rollout(self, context, time, steps, dt):
        if steps < 1 or dt <= 0:
            raise ValueError("steps and dt must be positive.")
        sequence = self.encode_sequence(context)
        time = batch_time(time, sequence)
        future = []
        for _ in range(steps):
            following = sequence[:, -1] + dt * self.velocity(sequence, time)
            future.append(self.codec.decode(following))
            sequence = torch.cat((sequence[:, 1:], following[:, None]), dim=1)
            time = time + dt
        return torch.stack(future, dim=1)
