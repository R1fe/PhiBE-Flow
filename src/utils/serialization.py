"""Compatibility loader for the checksummed, trusted upstream VQ checkpoint.

The upstream file stores a Lightning callback type as training metadata.
Inference does not use it; avoid requiring the entire Lightning dependency tree.
This is NOT a sandbox for untrusted pickle files.
"""

import pickle
from types import SimpleNamespace

import torch


class _UnusedModelCheckpoint:
    pass


class _CheckpointUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "pytorch_lightning.callbacks.model_checkpoint" and name == "ModelCheckpoint":
            return _UnusedModelCheckpoint
        return super().find_class(module, name)


def load_vq_checkpoint(path):
    compatibility = SimpleNamespace(__name__="pickle", Unpickler=_CheckpointUnpickler,
                                     load=pickle.load, loads=pickle.loads)
    return torch.load(path, map_location="cpu", pickle_module=compatibility, weights_only=False)
