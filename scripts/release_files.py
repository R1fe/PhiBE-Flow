"""Positive allowlist for the anonymous source-only artifact."""

from pathlib import Path

ROOT_FILES = (
    "README.md", "DATASETS.md", "REPRODUCIBILITY.md", "RELEASE_CHECKLIST.md",
    "THIRD_PARTY_NOTICES.md", "LICENSE_STATUS.md", "requirements.txt",
    "environment.yml", "assets.json", ".gitignore", ".gitattributes", "FVD.md",
    "EXTERNAL_CODECS.md", "CODE_GUIDE.md", "requirements-gpe.txt",
)
SCRIPT_FILES = (
    "train.py", "eval.py", "download_assets.py", "prepare_kth.py", "prepare_nse.py",
    "generate_acrobot.py", "check_setup.py", "audit_release.py", "export_release.py",
    "release_files.py", "overfit_nse.py", "generate_acrobot_benchmark.py",
)
CONFIG_FILES = ("acrobot_angles.yaml", "acrobot_frames.yaml", "nse.yaml", "kth.yaml",
                "smoke/kth.yaml", "smoke/acrobot_frames.yaml", "acrobot_frames_gpe.yaml")


def release_files(root):
    root = Path(root)
    names = list(ROOT_FILES)
    names += ["scripts/"+name for name in SCRIPT_FILES]
    names += ["configs/"+name for name in CONFIG_FILES]
    names += ["data/README.md", ".github/workflows/tests.yml"]
    for directory, pattern in (("src", "*.py"), ("tests", "test_*.py"), ("licenses", "*.txt")):
        names += [p.relative_to(root).as_posix() for p in sorted((root/directory).rglob(pattern))
                  if "__pycache__" not in p.parts]
    if (root/"LICENSE").is_file():
        names.append("LICENSE")
    files = [root/name for name in sorted(set(names))]
    if any(path.as_posix().endswith("/src/models/gpe.py") for path in files):
        raise ValueError("GPE implementation must not be included in the release.")
    return files
