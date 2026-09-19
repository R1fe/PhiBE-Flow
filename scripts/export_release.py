"""Export clean source without history, local metadata, assets or experiments."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_release import audit
from scripts.release_files import release_files


def export(root, destination):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    report = audit(root)
    if report["findings"]:
        raise ValueError("Source audit failed; run scripts/audit_release.py for details.")
    archive_path = destination.with_suffix(".zip")
    if destination.exists() or archive_path.exists():
        raise FileExistsError("Export destination already exists; choose a new empty path.")
    # Resolve/capture source list before creating output; never recursively copy the workspace.
    sources = release_files(root)
    destination.mkdir(parents=True)
    entries = {}
    for source in sources:
        name = source.relative_to(root).as_posix()
        content = source.read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8")
        target = destination/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        entries[name] = content
    manifest = {"status": "draft_pending_manual_release_checks", "files": {
        name: hashlib.sha256(content).hexdigest() for name, content in sorted(entries.items())}}
    entries["SOURCE_MANIFEST.json"] = (json.dumps(manifest, indent=2)+"\n").encode()
    (destination/"SOURCE_MANIFEST.json").write_bytes(entries["SOURCE_MANIFEST.json"])
    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(entries.items()):
            # No original creation times, usernames, filesystem paths or executable metadata.
            info = zipfile.ZipInfo("anonymous-code/"+name, date_time=(2000, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, content)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/".release/anonymous-code")
    args = parser.parse_args()
    result = export(ROOT, args.output)
    print(f"Exported {len(result['files'])} source files. This is a draft, not a publication approval.")


if __name__ == "__main__":
    main()
