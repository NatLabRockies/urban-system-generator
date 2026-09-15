#!/usr/bin/env python
"""Copy the registry's embedded bundles into ``src/usg/resources/bundles/`` (run before building a wheel).

Every model marked ``"embedded": true`` (or each id given) is downloaded from its release URL and checked against the
registry's archive and manifest sha256. ``--from-dir`` takes ``<model_id>/`` bundle directories from a local folder
instead, e.g. a folder of exported bundles, and checks them against the registry's manifest sha256. Every
bundle is fully verified before it is copied.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from usg.bundle import Bundle
from usg.bundle.manifest import MANIFEST, sha256_file
from usg.download import Cache
from usg.registry import Registry
from usg.resources import embedded_bundles_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model_ids", nargs="*", help="models to embed (default: every registry model marked embedded)")
    parser.add_argument("--registry", default=None, help="registry file or URL (default: the packaged registry.json)")
    parser.add_argument("--from-dir", type=Path, default=None, help="local directory holding <model_id>/ bundles")
    parser.add_argument("--dest", type=Path, default=None, help="default: usg/resources/bundles")
    args = parser.parse_args(argv)

    registry = Registry.load(args.registry)
    dest = args.dest or embedded_bundles_dir()
    model_ids = args.model_ids or [m.model_id for m in registry.models() if m.embedded]
    if not model_ids:
        print(f"no embedded models in {registry.source}", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory() as tmp:
        cache = Cache(tmp)
        for model_id in model_ids:
            entry = registry.get(model_id)
            if args.from_dir is None:
                src = cache.ensure(entry)
            else:
                src = args.from_dir / model_id
                if entry.manifest_sha256 and sha256_file(src / MANIFEST) != entry.manifest_sha256:
                    print(f"{model_id}: {src} does not match the registry manifest_sha256", file=sys.stderr)
                    return 1
            Bundle.open(src, verify="full")
            target = dest / model_id
            shutil.rmtree(target, ignore_errors=True)
            shutil.copytree(src, target)
            print(f"embedded {model_id} -> {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
