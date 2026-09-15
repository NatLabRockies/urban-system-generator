import json
import os
import subprocess
import sys


def test_importing_package_does_not_import_torch():
    code = (
        "import sys, usg, usg.schema, usg.bundle, usg.errors;"
        "assert 'torch' not in sys.modules, 'torch was imported'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_light_cli_commands_do_not_import_torch(tmp_path):
    from usg.resources import RESOURCES_DIR

    geojson = RESOURCES_DIR.parents[2] / "examples" / "example_geojson.json"
    args = {
        "help": ["--help"],
        "geojson2csv": ["geojson2csv", "-i", str(geojson), "-o", str(tmp_path / "b.csv")],
        "process": ["process", "-i", str(tmp_path / "b.csv"), "-o", str(tmp_path / "final.csv"), "-g", str(geojson),
                    "--no-reports"],
        "models": ["models", "list"],
    }
    code = f"""
import json, sys
from click.testing import CliRunner
import usg.cli
from usg.cli import cli
assert 'torch' not in sys.modules, 'importing usg.cli imported torch'
for name, argv in json.loads({json.dumps(json.dumps(args))}).items():
    result = CliRunner().invoke(cli, argv, catch_exceptions=False)
    assert result.exit_code == 0, (name, result.output)
    assert 'torch' not in sys.modules, f'usg {{name}} imported torch'
"""
    env = {**os.environ, "USG_CACHE_DIR": str(tmp_path / "cache"), "USG_REGISTRY": ""}
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
