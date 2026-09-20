"""Scan source files in the public allowlist for private metadata."""

import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.release_files import release_files


def inspect_text(text, deny_tokens=()):
    patterns = {
        "absolute_windows_path": r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]",
        "personal_unix_path": r"/(?:home|Users|root)/[A-Za-z0-9_.-]+",
        "email_address": r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}",
        "private_ssh_endpoint": r"(?:ssh:/{2}|\broot[@]|\bconnect\.[\w.-]+)",
        "credential_assignment": r"(?i)\b(?:password|api_key|access_token|secret_key)\s*[:=]\s*['\"][^'\"]+",
        "author_metadata": r"\b__author__\s*=",
    }
    findings = []
    for line_number, line in enumerate(text.splitlines(), 1):
        for kind, pattern in patterns.items():
            if re.search(pattern, line):
                findings.append(dict(line=line_number, kind=kind))
        if any(token and token.casefold() in line.casefold() for token in deny_tokens):
            findings.append(dict(line=line_number, kind="user_supplied_identity_token"))
    return findings


def audit(root, deny_tokens=()):
    root = Path(root).resolve()
    findings, files = [], release_files(root)
    for file in files:
        relative = file.relative_to(root).as_posix()
        if not file.is_file():
            findings.append(dict(file=relative, kind="missing_release_file"))
            continue
        if file.is_symlink() or root not in file.resolve().parents:
            findings.append(dict(file=relative, kind="symlink_or_external_path"))
            continue
        if file.stat().st_size > 2*1024*1024:
            findings.append(dict(file=relative, kind="unexpected_large_source_file"))
        try:
            text = file.read_text(encoding="utf-8")
        except UnicodeError:
            findings.append(dict(file=relative, kind="non_text_release_file"))
            continue
        # Copyright holders must not be erased from verbatim third-party licenses.
        if not relative.startswith("licenses/") and relative != "LICENSE":
            for finding in inspect_text(text, deny_tokens):
                findings.append(dict(file=relative, **finding))
    assets = json.loads((root/"assets.json").read_text(encoding="utf-8")) if (root/"assets.json").exists() else {"assets": {}}
    missing_assets = [name for name, item in assets["assets"].items() if not item.get("url") or not item.get("sha256")]
    return dict(file_count=len(files), findings=findings, source_scan_passed=not findings,
                assets_without_direct_download=missing_assets,
                project_license_present=(root/"LICENSE").is_file())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--deny-token", action="append", default=[], help="Local names/identifiers to scan; never stored in reports.")
    parser.add_argument("--require-ready", action="store_true", help="Also require direct asset URLs, checksums and a project LICENSE file.")
    args = parser.parse_args()
    report = audit(args.root, args.deny_token)
    print(json.dumps(report, indent=2))
    failed = bool(report["findings"])
    if args.require_ready:
        failed |= bool(report["assets_without_direct_download"]) or not report["project_license_present"]
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
