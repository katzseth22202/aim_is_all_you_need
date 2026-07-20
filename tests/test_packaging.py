"""Distribution-level regression tests for the installable project."""

import io
import os
import shutil
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from email.parser import Parser
from pathlib import Path
from zipfile import ZipFile

from packaging.requirements import Requirement
from setuptools.build_meta import build_wheel


def test_wheel_contains_code_dependencies_and_importable_entry_point(
    tmp_path: Path,
) -> None:
    """A built wheel must contain ``src`` and import its console-script target."""
    project_root = Path(__file__).resolve().parents[1]
    checkout = tmp_path / "checkout"
    shutil.copytree(
        project_root,
        checkout,
        ignore=shutil.ignore_patterns(
            ".git", ".pytest_cache", ".mypy_cache", "__pycache__"
        ),
    )
    wheel_dir = tmp_path / "wheel"
    wheel_dir.mkdir()

    previous_cwd = Path.cwd()
    try:
        os.chdir(checkout)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            wheel_name = build_wheel(str(wheel_dir))
    finally:
        os.chdir(previous_cwd)

    wheel = wheel_dir / wheel_name
    with ZipFile(wheel) as archive:
        names = set(archive.namelist())
        assert "src/__init__.py" in names
        assert "src/main.py" in names

        metadata_name = next(
            name for name in names if name.endswith(".dist-info/METADATA")
        )
        metadata = Parser().parsestr(archive.read(metadata_name).decode())
        dependencies = {
            Requirement(value).name.lower()
            for value in metadata.get_all("Requires-Dist", [])
        }
        assert dependencies == {
            "astropy",
            "boinor",
            "numpy",
            "pandas",
            "scipy",
            "tabulate",
        }

        entry_points_name = next(
            name for name in names if name.endswith(".dist-info/entry_points.txt")
        )
        assert "main = src.main:main" in archive.read(entry_points_name).decode()

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(wheel)
    subprocess.run(
        [sys.executable, "-c", "from src.main import main; assert callable(main)"],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
