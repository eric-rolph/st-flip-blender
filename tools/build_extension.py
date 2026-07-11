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
NORMALIZED_TEXT_SUFFIXES = {".md", ".py", ".toml", ".txt"}
NORMALIZED_TEXT_NAMES = {"LICENSE"}


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


def package_payload(path: Path) -> bytes:
    """Return platform-neutral bytes for one allow-listed package file.

    Git may materialize text files with CRLF on Windows and LF on Linux. The
    extension archive is a release artifact, so normalize known text payloads
    explicitly instead of letting checkout settings change its digest.
    """
    payload = path.read_bytes()
    if (path.suffix.lower() in NORMALIZED_TEXT_SUFFIXES
            or path.name in NORMALIZED_TEXT_NAMES):
        payload = payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return payload


def build(output: Path | None = None, root: Path = ROOT) -> Path:
    output = (output or default_output(root)).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Source-only extensions are small. Storing entries verbatim avoids zlib
    # implementation/version variance, making the complete ZIP reproducible
    # once text payloads and metadata are normalized.
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in package_files(root):
            relative = path.relative_to(root).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(2026, 1, 1, 0, 0, 0))
            info.create_system = 3  # fixed Unix metadata on every build host
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o644 << 16
            archive.writestr(info, package_payload(path))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="archive path")
    args = parser.parse_args()
    print(build(args.output))


if __name__ == "__main__":
    main()
