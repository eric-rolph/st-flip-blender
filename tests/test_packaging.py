import re
import tomllib
import zipfile

from stflip import __version__
from tools.build_extension import ROOT, build, package_files


def test_package_allowlist_excludes_development_files():
    relative = {path.relative_to(ROOT).as_posix()
                for path in package_files()}
    assert "blender_manifest.toml" in relative
    assert "__init__.py" in relative
    assert "addon/operators.py" in relative
    assert "stflip/solver.py" in relative
    assert not any(name.startswith((".git/", "tests/", "tmp/"))
                   for name in relative)
    assert not any("__pycache__" in name or name.endswith(".pyc")
                   for name in relative)


def test_built_archive_has_blender_extension_layout(tmp_path):
    output = build(tmp_path / "st_flip.zip")
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        assert names == sorted(names)
        assert "blender_manifest.toml" in names
        assert "__init__.py" in names
        assert all(not name.startswith("st-flip-blender/") for name in names)
        assert archive.testzip() is None


def test_release_version_is_consistent_across_package_surfaces():
    manifest = tomllib.loads(
        (ROOT / "blender_manifest.toml").read_text("utf-8"))
    project = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))
    entrypoint = (ROOT / "__init__.py").read_text("utf-8")
    match = re.search(r'"version": \((\d+), (\d+), (\d+)\)', entrypoint)

    assert match is not None
    entrypoint_version = ".".join(match.groups())
    assert manifest["version"] == project["project"]["version"] \
        == entrypoint_version == __version__
