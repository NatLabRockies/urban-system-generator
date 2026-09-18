import pytest

from usg.bundle import Bundle
from usg.download import Cache, default_cache_dir
from usg.errors import DownloadError
from usg.registry import RegistryEntry


def test_download_verify_extract_and_reuse(gnn_bundle, http_server, helpers, tmp_path):
    root, base = http_server
    entry = RegistryEntry.from_json(helpers.publish(gnn_bundle, root, base))
    cache = Cache(tmp_path / "cache")
    progress = []
    path = cache.ensure(entry, progress=lambda done, total: progress.append((done, total)))
    assert path == cache.path(entry.model_id)
    assert Bundle.open(path, verify="fast").model_id == entry.model_id
    assert (path / ".verified").exists()
    assert (cache.dir / "locks").is_dir()  # the download took the locked path; filelock may delete the
    # lock file itself on release (documented as not guaranteed on Windows), so do not assert the file
    assert not list((cache.dir / "tmp").glob("*.part"))
    assert progress and progress[-1][0] == entry.size_bytes

    (root / f"{entry.model_id}.tar.gz").unlink()  # cached copy is reused without the network
    assert cache.ensure(entry) == path
    assert cache.list() == [path]


def test_checksum_mismatch_deletes_partial_download(gnn_bundle, http_server, helpers, tmp_path):
    root, base = http_server
    raw = helpers.publish(gnn_bundle, root, base)
    raw["sha256"] = "f" * 64
    cache = Cache(tmp_path / "cache")
    with pytest.raises(DownloadError, match="checksum mismatch"):
        cache.ensure(RegistryEntry.from_json(raw))
    assert not list((cache.dir / "tmp").glob("*"))
    assert not cache.path(raw["model_id"]).exists()


def test_corrupt_archive_with_matching_checksum(http_server, tmp_path, gnn_bundle, helpers):
    import hashlib

    root, base = http_server
    data = b"not a tarball"
    (root / "broken.tar.gz").write_bytes(data)
    raw = helpers.entry(gnn_bundle, url=f"{base}/broken.tar.gz", sha256=hashlib.sha256(data).hexdigest())
    cache = Cache(tmp_path / "cache")
    with pytest.raises(DownloadError, match="invalid"):
        cache.ensure(RegistryEntry.from_json(raw))
    assert cache.list() == []


def test_offline_and_http_errors(gnn_bundle, http_server, helpers, tmp_path, monkeypatch):
    root, base = http_server
    entry = RegistryEntry.from_json(helpers.publish(gnn_bundle, root, base))
    cache = Cache(tmp_path / "cache")
    with pytest.raises(DownloadError, match="offline"):
        cache.ensure(entry, offline=True)
    monkeypatch.setenv("USG_OFFLINE", "1")
    with pytest.raises(DownloadError, match="offline"):
        cache.ensure(entry)
    monkeypatch.delenv("USG_OFFLINE")

    missing = RegistryEntry.from_json({**helpers.entry(gnn_bundle), "url": f"{base}/missing.tar.gz", "sha256": "0" * 64})
    with pytest.raises(DownloadError, match="HTTP 404") as exc:
        cache.ensure(missing)
    assert exc.value.exit_code == 3

    no_sha = RegistryEntry.from_json({**helpers.entry(gnn_bundle), "url": f"{base}/x.tar.gz"})
    with pytest.raises(DownloadError, match="sha256"):
        cache.ensure(no_sha)

    path = cache.ensure(entry)  # now cached: offline works
    monkeypatch.setenv("USG_OFFLINE", "true")
    assert cache.ensure(entry) == path


def test_clear_and_default_dir(gnn_bundle, tabddpm_bundle, helpers, tmp_path, monkeypatch):
    cache = Cache(tmp_path / "cache")
    for b in (gnn_bundle, tabddpm_bundle):
        helpers.copy(b, cache.dir)
    assert [p.name for p in cache.list()] == sorted([gnn_bundle.name, tabddpm_bundle.name])
    assert cache.clear(gnn_bundle.name) == 1 and cache.clear(gnn_bundle.name) == 0
    assert cache.clear() == 1 and cache.list() == []
    with pytest.raises(DownloadError):
        cache.path("../etc")
    monkeypatch.setenv("USG_CACHE_DIR", str(tmp_path / "elsewhere"))
    assert default_cache_dir() == tmp_path / "elsewhere"
    assert Cache().dir == tmp_path / "elsewhere"
