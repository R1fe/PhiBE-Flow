"""Portable overrides shared by training, evaluation and setup validation."""


def add_runtime_arguments(parser):
    parser.add_argument("--data", default=None, help="Optional dataset path override.")
    parser.add_argument("--device", default=None, help="auto, cpu, cuda or cuda:N.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)


def apply_runtime_arguments(config, args):
    for arg, section, key in (("data", "dataset", "dataset_path"),
                               ("device", "training", "device"),
                               ("batch_size", "dataset", "batch_size"),
                               ("num_workers", "dataset", "num_workers")):
        value = getattr(args, arg, None)
        if value is not None:
            config[section][key] = value
    if int(config.dataset.batch_size) < 1 or int(config.dataset.num_workers) < 0:
        raise ValueError("batch-size must be positive; num-workers must be nonnegative.")
    return config
