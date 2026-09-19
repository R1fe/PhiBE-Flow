"""Download declared public assets, with explicit consent for upstream terms."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.download import download_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset", choices=["kth_vqvae", "kth_raw", "nse_dataset", "acrobot_benchmark"])
    parser.add_argument("--accept-terms", action="store_true", help="Confirm you have read the upstream data/model terms.")
    args = parser.parse_args()
    if not args.accept_terms:
        parser.error("Read DATASETS.md and upstream terms, then supply --accept-terms.")
    if args.asset == "kth_raw":
        records = []
        for action in ("walking", "jogging", "running", "boxing", "handwaving", "handclapping"):
            url = f"https://www.csc.kth.se/cvap/actions/{action}.zip"
            destination = ROOT / "data/kth_raw" / f"{action}.zip"
            print(f"Downloading {action} from the official KTH host...", flush=True)
            digest = download_file(url, destination)
            records.append(dict(file=destination.name, sha256=digest, url=url))
        (ROOT / "data/kth_raw/download_manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        print("Downloaded raw AVI archives. Next: python scripts/prepare_kth.py")
        return
    asset = json.loads((ROOT / "assets.json").read_text(encoding="utf-8"))["assets"][args.asset]
    if not asset["url"]:
        parser.error(f"Asset is not public yet: {args.asset} ({asset['status']}). See RELEASE_CHECKLIST.md.")
    print(f"Downloading {args.asset} (large file; please wait)...", flush=True)
    digest = download_file(asset["url"], ROOT / asset["destination"], asset["sha256"])
    print(f"Verified SHA-256: {digest}")


if __name__ == "__main__":
    main()
