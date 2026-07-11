import re
import tomllib
import zipfile

from stflip import __version__
from tools.build_extension import ROOT, build, package_files, package_payload


def test_package_allowlist_excludes_development_files():
    relative = {path.relative_to(ROOT).as_posix()
                for path in package_files()}
    assert "blender_manifest.toml" in relative
    assert "__init__.py" in relative
    assert "addon/operators.py" in relative
    assert "stflip/handoff.py" in relative
    assert "stflip/solver.py" in relative
    assert "stflip/validation.py" in relative
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
        assert b"\r\n" not in archive.read("addon/operators.py")
        assert all(info.create_system == 3 for info in archive.infolist())
        assert archive.testzip() is None


def test_text_package_payload_is_platform_neutral(tmp_path):
    source = tmp_path / "module.py"
    source.write_bytes(b"first\r\nsecond\rthird\n")

    assert package_payload(source) == b"first\nsecond\nthird\n"


def test_archive_bytes_ignore_checkout_line_endings(tmp_path):
    lf_root = tmp_path / "lf"
    crlf_root = tmp_path / "crlf"
    for source in package_files():
        relative = source.relative_to(ROOT)
        payload = package_payload(source)
        for root, data in (
            (lf_root, payload),
            (crlf_root, payload.replace(b"\n", b"\r\n")),
        ):
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)

    lf_archive = build(tmp_path / "lf.zip", lf_root)
    crlf_archive = build(tmp_path / "crlf.zip", crlf_root)

    assert lf_archive.read_bytes() == crlf_archive.read_bytes()


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
