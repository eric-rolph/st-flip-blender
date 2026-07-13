"""Fluid material library data integrity and selection logic.

The material *construction* (node graphs, socket values, EEVEE flags) is verified
against a live Blender build in the headless harness; these tests lock down the
pure data table and the scene-selection helper, which are what silently drift
when a new fluid is added or the property enum and the spec table fall out of
sync.
"""

import ast
import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]

# Fluids the UI hints as refractive (needs raytracing). This set is repeated in
# properties.py, operators.py, and panels.py; the test below ties it to the
# authoritative spec so the copies cannot silently drift from the shader.
_REFRACTIVE_HINT_KEYS = {"WATER", "CLEAR", "HONEY", "JUICE"}


def _properties_enum_keys(prop_name):
    """Extract an EnumProperty's item keys from addon/properties.py by parsing
    the source, so the test enforces the REAL UI enum rather than a copy."""
    tree = ast.parse((ROOT / "addon" / "properties.py").read_text("utf-8"))
    for node in ast.walk(tree):
        # Blender props use annotation syntax (``name: EnumProperty(...)``), so
        # the call is the AnnAssign annotation, not an assigned value.
        if not (isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == prop_name
                and isinstance(node.annotation, ast.Call)):
            continue
        for kw in node.annotation.keywords:
            if kw.arg == "items" and isinstance(kw.value, ast.List):
                return {elt.elts[0].value for elt in kw.value.elts
                        if isinstance(elt, ast.Tuple) and elt.elts}
    raise AssertionError(f"{prop_name} EnumProperty not found in properties.py")


# The fluid materials the UI actually offers in the selector, read from source.
_SELECTOR_KEYS = _properties_enum_keys("fluid_material")


@pytest.fixture
def mesher(monkeypatch):
    """Import addon/mesher.py with a bare bpy stub (it makes no bpy calls at
    import time, so the data table and pure helpers load cleanly)."""
    bpy = types.ModuleType("bpy")
    bpy.data = types.SimpleNamespace(materials={}, node_groups={})
    bpy.context = types.SimpleNamespace(scene=None)
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    spec = importlib.util.spec_from_file_location(
        "stflip_test_mesher", ROOT / "addon" / "mesher.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_selector_keys_all_have_specs(mesher):
    specs = mesher._FLUID_MATERIAL_SPECS
    missing = _SELECTOR_KEYS - set(specs)
    assert not missing, f"selector keys without a material spec: {missing}"
    assert mesher.DEFAULT_FLUID_MATERIAL in specs


def test_material_item_keys_match_spec_table(mesher):
    item_keys = {key for key, _label, _desc in mesher.FLUID_MATERIAL_ITEMS}
    assert item_keys == set(mesher._FLUID_MATERIAL_SPECS)


def test_properties_enum_is_subset_of_specs(mesher):
    # The selector keys are parsed from properties.py source, so this genuinely
    # fails if the UI enum and the shader spec table drift apart.
    assert _SELECTOR_KEYS <= set(mesher._FLUID_MATERIAL_SPECS)
    assert _SELECTOR_KEYS  # non-empty: the parse actually found the enum


def test_refractive_hint_matches_spec_refractive_flag(mesher):
    # The "needs raytracing" hint (duplicated in properties/operators/panels)
    # must equal the fluids the shader actually marks refractive.
    specs = mesher._FLUID_MATERIAL_SPECS
    refractive = {k for k in _SELECTOR_KEYS if specs[k].get("refractive")}
    assert refractive == _REFRACTIVE_HINT_KEYS


def test_every_spec_is_well_formed(mesher):
    names = set()
    for key, spec in mesher._FLUID_MATERIAL_SPECS.items():
        assert spec["name"].startswith("STFLIP "), key
        names.add(spec["name"])
        assert len(spec["base_color"]) == 4, key
        assert spec["roughness"] >= 0.0, key
        assert spec["ior"] >= 1.0, key
        assert 0.0 <= spec["transmission"] <= 1.0, key
        if spec.get("refractive"):
            # A refractive fluid that does not transmit would render opaque.
            assert spec["transmission"] > 0.0, key
    assert len(names) == len(mesher._FLUID_MATERIAL_SPECS), "duplicate names"


def test_special_fluids_carry_their_defining_fields(mesher):
    specs = mesher._FLUID_MATERIAL_SPECS
    # Milk is opaque but must scatter (subsurface), not refract.
    assert not specs["MILK"].get("refractive")
    assert specs["MILK"]["subsurface"]["weight"] > 0.0
    # Lava glows via a blackbody-driven emission with non-zero strength.
    assert specs["LAVA"]["emission"]["blackbody_k"] > 0.0
    assert specs["LAVA"]["emission"]["strength"] > 0.0
    # Foam has a faint constant emission so it reads in shadow.
    assert specs["FOAM"]["emission"]["strength"] > 0.0


@pytest.mark.parametrize(
    ("selection", "expected"),
    [
        ("HONEY", "HONEY"),
        ("WATER", "WATER"),
        ("BOGUS", None),      # unknown key -> no selection
        (None, None),         # unset -> no selection
    ],
)
def test_selected_fluid_kind(mesher, selection, expected):
    scene = types.SimpleNamespace(
        stflip=types.SimpleNamespace(fluid_material=selection))
    assert mesher._selected_fluid_kind(scene) == expected


def test_selected_fluid_kind_tolerates_missing_scene(mesher):
    assert mesher._selected_fluid_kind(None) is None
    assert mesher._selected_fluid_kind(types.SimpleNamespace()) is None
