"""Portable overrides shared by training, evaluation and setup validation."""


def add_runtime_arguments(parser):
    parser.add_argument("--data", default=None, help="Optional dataset path override.")
    parser.add_argument("--device", default=None, help="auto, cpu, cuda or cuda:N.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--codec", choices=("mlp", "gpe", "torchscript"), default=None,
                        help="Acrobot image codec backend.")
    parser.add_argument("--gpe-source", default=None, help="Reader-owned GPE source directory.")
    parser.add_argument("--trust-external-code", action="store_true",
                        help="Allow execution of the external codec source/TorchScript you have reviewed.")


def apply_runtime_arguments(config, args):
    if getattr(args, "codec", None) or getattr(args, "gpe_source", None) or getattr(args, "trust_external_code", False):
        if config.dataset.name != "acrobot_frames":
            raise ValueError("Codec overrides apply only to acrobot_frames.")
        codec = config.setdefault("codec", {})
        if getattr(args, "codec", None):
            codec["type"] = args.codec
        if getattr(args, "gpe_source", None):
            codec["source_dir"] = args.gpe_source
        if getattr(args, "trust_external_code", False):
            codec["trust_external_code"] = True
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
