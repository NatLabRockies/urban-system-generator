"""Deterministic ``.tar.gz`` packing and safe extraction of bundles."""

from __future__ import annotations

import gzip
import hashlib
import io
import os
import shutil
import sys
import tarfile
from pathlib import Path, PurePosixPath

from usg.errors import BundleError


def pack_bundle(bundle_dir: str | Path, out_path: str | Path) -> str:
    """Pack ``bundle_dir`` as ``<name>/...`` into a reproducible tar.gz; returns its sha256 hex digest."""
    bundle_dir = Path(bundle_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    with (
        gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0) as gz,
        tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar,
    ):
        for f in sorted(p for p in bundle_dir.iterdir() if p.is_file() and not p.name.startswith(".")):
            info = tar.gettarinfo(str(f), arcname=f"{bundle_dir.name}/{f.name}")
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mode = 0o644
            with open(f, "rb") as fh:
                tar.addfile(info, fh)
    data = buf.getvalue()
    out_path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _check_members(tar: tarfile.TarFile, expected_top: str | None) -> str:
    tops = set()
    for m in tar.getmembers():
        p = PurePosixPath(m.name)
        if p.is_absolute() or ".." in p.parts:
            raise BundleError(f"unsafe path in bundle archive: {m.name}")
        if not (m.isfile() or m.isdir()):
            raise BundleError(f"unsupported member type in bundle archive: {m.name}")
        tops.add(p.parts[0])
    if len(tops) != 1:
        raise BundleError(f"bundle archive must contain exactly one top-level directory, found {sorted(tops)}")
    top = tops.pop()
    if expected_top is not None and top != expected_top:
        raise BundleError(f"bundle archive contains {top!r}, expected {expected_top!r}")
    return top


def extract_bundle(archive: str | Path, dest_parent: str | Path, expected_name: str | None = None) -> Path:
    """Extract into ``dest_parent/<name>`` atomically (staging dir + rename)."""
    dest_parent = Path(dest_parent)
    dest_parent.mkdir(parents=True, exist_ok=True)
    staging = dest_parent / f".extract-{os.getpid()}-{Path(archive).stem}"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir()
    try:
        try:
            with tarfile.open(archive, "r:gz") as tar:
                top = _check_members(tar, expected_name)
                if sys.version_info >= (3, 12):
                    tar.extractall(staging, filter="data")
                else:
                    tar.extractall(staging)  # members validated above
        except (tarfile.TarError, OSError, EOFError) as exc:
            raise BundleError(f"cannot extract bundle archive {archive}: {exc}") from None
        final = dest_parent / top
        if final.exists():
            shutil.rmtree(final)
        (staging / top).rename(final)
        return final
    finally:
        shutil.rmtree(staging, ignore_errors=True)
