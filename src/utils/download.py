"""Explicit, checksum-aware downloads; never invoked by training."""

from pathlib import Path
import hashlib
import os
import ssl
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import certifi


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url, destination, sha256=None):
    if urlparse(url).scheme != "https":
        raise ValueError("Downloads require HTTPS.")
    destination = Path(destination)
    if destination.exists():
        actual = sha256_file(destination)
        if sha256 is not None and actual != sha256:
            raise ValueError(f"Checksum mismatch for existing {destination.name}; refusing to overwrite.")
        return actual
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    if partial.exists():
        raise FileExistsError(f"An incomplete download exists: {partial}. Remove it before retrying.")
    try:
        request = Request(url, headers={"User-Agent": "anonymous-research-artifact/1.0"})
        context = ssl.create_default_context(cafile=os.environ.get("SSL_CERT_FILE") or certifi.where())
        with urlopen(request, timeout=120, context=context) as response, partial.open("xb") as handle:
            if urlparse(response.geturl()).scheme != "https":
                raise ValueError("Refusing a redirect to a non-HTTPS endpoint.")
            for chunk in iter(lambda: response.read(1024*1024), b""):
                handle.write(chunk)
        actual = sha256_file(partial)
        if sha256 is not None and actual != sha256:
            raise ValueError(f"Downloaded {destination.name} failed SHA-256 verification.")
        os.replace(partial, destination)
        return actual
    except Exception:
        partial.unlink(missing_ok=True)
        raise
