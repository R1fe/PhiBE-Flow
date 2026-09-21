# Data 

Follow [DATASETS.md](../DATASETS.md) to download/prepare data. 

- `acrobot_angles/`: trusted benchmark pickle, or generated numeric demo NPZ.
- `acrobot_frames/`: optional loader-only frame sequences; no unified trainer.
- `nse/`: one format only, `.pt` shards or converted float32 `.npy` shards.
- `kth/`: prepared HDF5 videos and the pretrained `vqvae.ckpt`.
- `kth_raw/`: official AVI ZIP archives used by `prepare_kth.py`.
