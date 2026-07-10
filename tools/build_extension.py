"""Build a clean, reproducible Blender extension archive.

Run from the repository root::

    python tools/build_extension.py

The archive intentionally excludes the Git checkout, tests, caches, and other
development files.  Blender expects ``blender_manifest.toml`` and the add-on
entry point at the root of the ZIP, not inside a repository-named directory.
"""

from __future__ import annotations

import argparse
import tomllib
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOP_LEVEL_FILES = (
    "__init__.py",
    "blender_manifest.toml",
    "LICENSE",
    "README.md",
)
PACKAGE_DIRS = ("addon", "stflip")


def package_files(root: Path = ROOT) -> list[Path]:
    """Return the allow-listed source files in stable archive order."""
    files = [root / name for name in TOP_LEVEL_FILES]
    for directory in PACKAGE_DIRS:
        files.extend(
            path
            for path in (root / directory).rglob("*")
            if path.is_file() and path.suffix != ".pyc"
            and "__pycache__" not in path.parts
        )
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing package file(s): {missing}")
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def default_output(root: Path = ROOT) -> Path:
    manifest = tomllib.loads((root / "blender_manifest.toml").read_text("utf-8"))
    return root / "dist" / f"{manifest['id']}-{manifest['version']}.zip"


def build(output: Path | None = None, root: Path = ROOT) -> Path:
    output = (output or default_output(root)).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as archive:
        for path in package_files(root):
            relative = path.relative_to(root).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes(), compresslevel=9)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="archive path")
    args = parser.parse_args()
    print(build(args.output))


if __name__ == "__main__":
    main()
