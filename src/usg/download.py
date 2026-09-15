"""Local model cache: checksum-verified download and atomic extraction of registry bundles.

Layout of the cache directory (``$USG_CACHE_DIR`` or ``<user cache>/usg/models``)::

    <model_id>/          extracted, verified bundle
    tmp/<id>.<pid>.part  in-flight downloads
    locks/<id>.lock      inter-process lock per model
"""

from __future__ import annotations

import hashlib
import os
import shutil
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from usg.__about__ import __version__
from usg.bundle.manifest import MANIFEST, sha256_file
from usg.constants import ENV_CACHE_DIR, ENV_OFFLINE, MODEL_ID_RE
from usg.errors import BundleError, DownloadError
from usg.registry import RegistryEntry

LOCK_TIMEOUT_S = 1800
_CHUNK = 1 << 20


def offline_mode(offline: bool = False) -> bool:
    return offline or os.environ.get(ENV_OFFLINE, "").strip().lower() in ("1", "true", "yes", "on")


def default_cache_dir() -> Path:
    env = os.environ.get(ENV_CACHE_DIR)
    if env:
        return Path(env).expanduser()
    import platformdirs

    return Path(platformdirs.user_cache_dir("usg")) / "models"


class Cache:
    def __init__(self, dir: str | Path | None = None):
        self.dir = Path(dir).expanduser() if dir is not None else default_cache_dir()

    def __repr__(self) -> str:
        return f"Cache({str(self.dir)!r})"

    def path(self, model_id: str) -> Path:
        if not MODEL_ID_RE.match(model_id):
            raise DownloadError(f"invalid model_id {model_id!r}")
        return self.dir / model_id

    def is_cached(self, model_id: str, manifest_sha256: str | None = None) -> bool:
        manifest = self.path(model_id) / MANIFEST
        if not manifest.is_file():
            return False
        return manifest_sha256 is None or sha256_file(manifest) == manifest_sha256

    def list(self) -> list[Path]:
        """Cached bundle directories, sorted by model id."""
        if not self.dir.is_dir():
            return []
        return sorted(p for p in self.dir.iterdir() if MODEL_ID_RE.match(p.name) and (p / MANIFEST).is_file())

    def clear(self, model_id: str | None = None) -> int:
        """Remove one cached model (or everything); returns the number of bundles removed."""
        if model_id is not None:
            target = self.path(model_id)
            if not target.exists():
                return 0
            shutil.rmtree(target)
            return 1
        removed = len(self.list())
        if self.dir.is_dir():
            for child in self.dir.iterdir():
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink(missing_ok=True)
        return removed

    def ensure(
        self,
        entry: RegistryEntry,
        offline: bool = False,
        progress: Callable[[int, int | None], None] | None = None,
    ) -> Path:
        """Return the cached bundle for ``entry``, downloading and verifying it first if needed."""
        target = self.path(entry.model_id)
        if self.is_cached(entry.model_id, entry.manifest_sha256):
            return target
        if offline_mode(offline):
            raise DownloadError(
                f"{entry.model_id} is not in the model cache ({self.dir}) and offline mode is on; "
                f"run `usg models download {entry.model_id}` while online or pass --model-dir"
            )
        if not entry.url:
            raise DownloadError(f"{entry.model_id} has no download URL in the registry")
        if not entry.sha256:
            raise DownloadError(f"{entry.model_id} has no archive sha256 in the registry; refusing to download")

        from filelock import FileLock, Timeout

        (self.dir / "locks").mkdir(parents=True, exist_ok=True)
        try:
            with FileLock(str(self.dir / "locks" / f"{entry.model_id}.lock"), timeout=LOCK_TIMEOUT_S):
                if self.is_cached(entry.model_id, entry.manifest_sha256):  # another process finished it
                    return target
                return self._download_and_extract(entry, progress)
        except Timeout:
            raise DownloadError(f"timed out waiting for another process downloading {entry.model_id}") from None

    def _download_and_extract(self, entry: RegistryEntry, progress) -> Path:
        from usg.bundle import Bundle, extract_bundle

        tmp_dir = self.dir / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        part = tmp_dir / f"{entry.model_id}.{os.getpid()}.part"
        try:
            digest = _fetch(entry.url, part, progress)
            if digest != entry.sha256.lower():
                raise DownloadError(
                    f"checksum mismatch for {entry.model_id}: expected {entry.sha256}, got {digest} "
                    f"(corrupted download or tampered archive)"
                )
            try:
                path = extract_bundle(part, self.dir, expected_name=entry.model_id)
                if entry.manifest_sha256 and sha256_file(path / MANIFEST) != entry.manifest_sha256:
                    raise BundleError(f"{entry.model_id}: manifest does not match the registry manifest_sha256")
                Bundle.open(path, verify="full")
            except BundleError as exc:
                shutil.rmtree(self.dir / entry.model_id, ignore_errors=True)
                raise DownloadError(f"downloaded bundle {entry.model_id} is invalid: {exc}") from None
            return path
        finally:
            part.unlink(missing_ok=True)


def _fetch(url: str, dest: Path, progress) -> str:
    """Stream ``url`` into ``dest``; returns the sha256 hex digest."""
    request = urllib.request.Request(url, headers={"User-Agent": f"urban-system-generator/{__version__}"})
    h = hashlib.sha256()
    try:
        with urllib.request.urlopen(request, timeout=60) as resp, open(dest, "wb") as out:
            total = resp.headers.get("Content-Length")
            total = int(total) if total and total.isdigit() else None
            done = 0
            for block in iter(lambda: resp.read(_CHUNK), b""):
                out.write(block)
                h.update(block)
                done += len(block)
                if progress is not None:
                    progress(done, total)
    except urllib.error.HTTPError as exc:
        raise DownloadError(f"download failed for {url}: HTTP {exc.code} {exc.reason}") from None
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise DownloadError(f"download failed for {url}: {exc}") from None
    return h.hexdigest()
