# Releasing

What CI cannot check, and therefore what a person has to.

CI runs `ruff check src tests` and `pytest -q`. It does **not** install URBANopt, OpenStudio or
EnergyPlus, and it should not: they are large installs, and a CI job that needs them is a CI job that
breaks for reasons unrelated to this package. Anything that needs them is a local gate.

That gap is not theoretical. Two defects reached a release because nothing simulated the output:

- **1.1.0** shipped a TabDDPM model that had never reached URBANopt. It produced a vented attic with an
  uninsulated ceiling — a pairing that occurs 0 times in 216,171 vented-attic buildings in ResStock — and
  HPXML refused the resulting R-0 assembly, failing one building in a 77-building district.
- Before that, a value of `None` was written where `options_lookup.tsv` offers `Void`. The test suite
  passed because the conformance check asked the validator to judge its own output.

The step that catches both is the acceptance run below. It takes about 20 minutes.

## Before every release

1. **Tests and lint.** `pytest -q` and `ruff check src tests`, both clean.

2. **URBANopt acceptance, every model.** From the model training repo, with URBANopt on `PATH`:

   ```
   python scripts/urbanopt_acceptance.py --case all -n 4
   ```

   This runs **every model type** (`gnn` and `tabddpm`) across both example districts, through
   EnergyPlus, and checks that each feature actually consumed our mapping CSV. Exit code is non-zero if
   any combination fails. A model nobody runs is a model nobody tests.

   Expect `77/77` for Baltimore and `4/4` for the combined district, per model.

3. **Clean-install check.** Build the wheel, install it into an empty virtualenv with nothing else from
   this repo on the path, and run a workflow offline:

   ```
   python -m build
   pip install dist/urban_system_generator-<version>-py3-none-any.whl
   USG_OFFLINE=1 usg workflow -i examples/example_geojson.json -o out/uo_buildstock_mapping.csv
   ```

   Confirm the output has no missing required columns, no blanks, and no value outside
   `options_lookup.tsv`.

4. **Registry resolves offline for every region.** Every champion must name an embedded bundle, or an
   offline run for that state fails while the national run succeeds:

   ```
   python -c "
   import json,pathlib
   r=json.loads(pathlib.Path('src/usg/resources/registry.json').read_text())
   embedded={m['model_id'] for m in r['models'] if m.get('embedded')}
   missing={i for v in r['champions'].values() for i in v.values()} - embedded
   print('referenced but not embedded:', missing or 'none')"
   ```

5. **Download URLs resolve.** Any model listed but not embedded must be fetchable; check each `url` in
   the registry returns 200. A release that lists a model it cannot serve is worse than not listing it.

6. **Version and changelog.** `src/usg/__about__.py` matches the tag, and `CHANGELOG.md` has an entry.
   If accuracy numbers changed because the *scoring* changed rather than the model, say so explicitly —
   a reader comparing headline numbers across versions will otherwise reach the wrong conclusion.

## Order of operations

Model bundles first, package second. The registry inside the wheel points at model release assets by
URL, so those assets must exist before anyone installs the wheel:

1. `gh release create models-<release>-v<N> <bundles>.tar.gz registry.json`
2. Verify each registry URL returns 200.
3. Commit, tag `vX.Y.Z`, push.
4. Wait for CI.
5. `gh release create vX.Y.Z dist/*`
6. PyPI upload last — it cannot be undone or replaced.
