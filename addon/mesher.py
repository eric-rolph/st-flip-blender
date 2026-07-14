"""Blender surfacing helpers for preview and paper-style reconstruction.

The fast preview path builds a Geometry Nodes graph that pulls the baked
particle object's vertices, splats them into a volume, and meshes the result.
The paper path accepts an already reconstructed density field or polygon mesh
and writes ordinary Blender mesh geometry without a live Geometry Nodes
dependency.
"""

import importlib
import math

import bpy
import numpy as np

GROUP_NAME = "STFLIP_Surface"
SURFACE_OBJ = "STFLIP Liquid Surface"
PARTICLE_OBJ = "STFLIP Particles"
SURFACE_MODIFIER = "STFLIP Surface"
SMOOTH_MODIFIER = "STFLIP Geometric Smoothing"
GROUP_SCHEMA_KEY = "stflip_schema_version"
GROUP_SCHEMA_VERSION = 3

_INTERFACE_SCHEMA = (
    ("Geometry", "INPUT", "NodeSocketGeometry"),
    ("Points Object", "INPUT", "NodeSocketObject"),
    ("Radius", "INPUT", "NodeSocketFloat"),
    ("Voxel Size", "INPUT", "NodeSocketFloat"),
    ("Material", "INPUT", "NodeSocketMaterial"),
    ("Geometry", "OUTPUT", "NodeSocketGeometry"),
)


def _load_openvdb():
    """Load either upstream or Blender-bundled OpenVDB Python bindings."""
    errors = []
    for module_name in ("openvdb", "pyopenvdb"):
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            errors.append(exc)
            continue
        missing = [
            name for name in ("FloatGrid", "createLinearTransform")
            if not callable(getattr(module, name, None))
        ]
        if not missing:
            return module
        errors.append(ImportError(
            f"{module_name} lacks required API: {', '.join(missing)}"))
    raise RuntimeError(
        "Paper MCF meshing requires Blender's OpenVDB Python module "
        "('openvdb' or the official-build name 'pyopenvdb'); use an official "
        "Blender build with OpenVDB enabled"
    ) from errors[-1]


def place_paper_surface_object(obj, world_origin):
    """Give a domain-local Paper mesh an exact translation-only world frame."""
    if obj is None:
        return None
    try:
        origin = np.asarray(world_origin, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("world origin must contain three finite values") from exc
    if origin.shape != (3,) or not bool(np.all(np.isfinite(origin))):
        raise ValueError("world origin must contain three finite values")
    values = tuple(float(value) for value in origin)

    # Neutralize delta transforms before assigning matrix_world.  Assignment
    # preserves an existing parent while solving the local transform required
    # for this exact world-space translation; it is also rotation-mode agnostic.
    for name, value in (
        ("delta_location", (0.0, 0.0, 0.0)),
        ("delta_rotation_euler", (0.0, 0.0, 0.0)),
        ("delta_rotation_quaternion", (1.0, 0.0, 0.0, 0.0)),
        ("delta_scale", (1.0, 1.0, 1.0)),
    ):
        if hasattr(obj, name):
            setattr(obj, name, value)
    try:
        from mathutils import Matrix

        transform = Matrix.Identity(4)
        transform.translation = values
        obj.matrix_world = transform
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        # Ordinary-Python tests and stripped Blender-like hosts have no
        # mathutils.  Generated output objects are unparented there, so the
        # explicit identity channels are equivalent.
        obj.location = values
        if hasattr(obj, "rotation_euler"):
            obj.rotation_euler = (0.0, 0.0, 0.0)
        if hasattr(obj, "rotation_quaternion"):
            obj.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
        if hasattr(obj, "rotation_axis_angle"):
            obj.rotation_axis_angle = (0.0, 0.0, 1.0, 0.0)
        if hasattr(obj, "scale"):
            obj.scale = (1.0, 1.0, 1.0)
    return obj


def _new_socket(ng, name, in_out, socket_type):
    return ng.interface.new_socket(name=name, in_out=in_out,
                                   socket_type=socket_type)


def _set_resolution_mode(node, legacy_value, menu_value):
    """Blender <=4.x exposes resolution_mode as a node property; 5.x turned
    it into a 'Resolution Mode' menu input socket."""
    if hasattr(node, "resolution_mode"):
        node.resolution_mode = legacy_value
    else:
        node.inputs["Resolution Mode"].default_value = menu_value


def _is_geometry_group(ng):
    return getattr(ng, "bl_idname", "") == "GeometryNodeTree"


def _schema_version(ng):
    try:
        return int(ng.get(GROUP_SCHEMA_KEY, 0))
    except (TypeError, ValueError):
        return 0


def _find_current_node_group():
    """Find the generated group even if Blender had to suffix its name."""
    named = bpy.data.node_groups.get(GROUP_NAME)
    if named is not None and _is_geometry_group(named) \
            and _schema_version(named) == GROUP_SCHEMA_VERSION:
        return named
    for candidate in bpy.data.node_groups:
        if candidate is named or not candidate.name.startswith(GROUP_NAME):
            continue
        if _is_geometry_group(candidate) \
                and _schema_version(candidate) == GROUP_SCHEMA_VERSION:
            return candidate
    return None


def _populate_node_group(ng):
    """Populate a newly-created group and stamp it only after it is valid."""
    ng.is_modifier = True

    sockets = {
        (name, in_out): _new_socket(ng, name, in_out, socket_type)
        for name, in_out, socket_type in _INTERFACE_SCHEMA
    }
    s_rad = sockets[("Radius", "INPUT")]
    s_vox = sockets[("Voxel Size", "INPUT")]
    s_rad.default_value = 0.05
    s_rad.min_value = 0.0
    s_vox.default_value = 0.04
    s_vox.min_value = 0.001

    n_in = ng.nodes.new("NodeGroupInput")
    n_in.location = (-800, 0)
    n_out = ng.nodes.new("NodeGroupOutput")
    n_out.location = (400, 0)

    info = ng.nodes.new("GeometryNodeObjectInfo")
    info.location = (-600, -100)
    info.transform_space = "RELATIVE"

    mtp = ng.nodes.new("GeometryNodeMeshToPoints")
    mtp.location = (-400, 0)
    mtp.mode = "VERTICES"

    ptv = ng.nodes.new("GeometryNodePointsToVolume")
    ptv.location = (-200, 0)
    _set_resolution_mode(ptv, "VOXEL_SIZE", "Size")
    # Do not inherit Blender-version or user-preference defaults. These values
    # define the preview field and therefore must be deterministic.
    ptv.inputs["Density"].default_value = 1.0

    vtm = ng.nodes.new("GeometryNodeVolumeToMesh")
    vtm.location = (0, 0)
    _set_resolution_mode(vtm, "GRID", "Grid")
    vtm.inputs["Threshold"].default_value = 0.5
    vtm.inputs["Adaptivity"].default_value = 0.0

    smooth = ng.nodes.new("GeometryNodeSetShadeSmooth")
    smooth.location = (200, 0)

    # Generated geometry ignores object material slots; assign explicitly.
    set_mat = ng.nodes.new("GeometryNodeSetMaterial")
    set_mat.location = (300, 0)

    ln = ng.links.new
    ln(n_in.outputs["Points Object"], info.inputs["Object"])
    ln(info.outputs["Geometry"], mtp.inputs["Mesh"])
    ln(n_in.outputs["Radius"], mtp.inputs["Radius"])
    ln(mtp.outputs["Points"], ptv.inputs["Points"])
    ln(n_in.outputs["Radius"], ptv.inputs["Radius"])
    ln(n_in.outputs["Voxel Size"], ptv.inputs["Voxel Size"])
    ln(ptv.outputs["Volume"], vtm.inputs["Volume"])
    ln(vtm.outputs["Mesh"], smooth.inputs["Geometry"])
    ln(smooth.outputs["Geometry"], set_mat.inputs["Geometry"])
    ln(n_in.outputs["Material"], set_mat.inputs["Material"])
    ln(set_mat.outputs["Geometry"], n_out.inputs["Geometry"])
    ng[GROUP_SCHEMA_KEY] = GROUP_SCHEMA_VERSION
    return ng


def build_node_group():
    """Return the current generated surface group.

    Older files can contain a same-named group built for a different Blender
    node API.  Such groups have no schema marker.  Leave that datablock intact
    (so opening/migrating a file is non-destructive) and build a marked local
    replacement; Blender will give it a numeric suffix when necessary.
    """
    ng = _find_current_node_group()
    if ng is not None:
        return ng

    ng = bpy.data.node_groups.new(GROUP_NAME, "GeometryNodeTree")
    try:
        return _populate_node_group(ng)
    except Exception:
        # Do not leave a half-built group that a later call could mistake for a
        # successful migration.
        bpy.data.node_groups.remove(ng)
        raise


# Data-driven fluid material library.  Each entry is a self-contained recipe the
# builder translates into a single Principled BSDF (plus a Blackbody node for the
# emissive lava).  Design decisions validated against Blender 5.1 EEVEE Next:
#   * Tint is carried by Base Color with a high Transmission Weight (colored
#     glass), NOT by a Volume Absorption node: raytraced refraction in EEVEE Next
#     traces screen buffers and does not integrate volume absorption along the
#     refracted ray, and the per-frame-rebuilt thin surface makes a volumetric
#     pass flicker.  Base-Color tint is reliable and cheap.
#   * Refractive fluids need the material flag ``use_raytrace_refraction`` (the
#     real EEVEE Next property; ``use_raytraced_transmission`` does not exist) AND
#     the scene flag ``scene.eevee.use_raytracing`` (set by the studio operator).
#   * Milk uses real Principled subsurface (supported in EEVEE Next); lava emits
#     via a Blackbody node into Emission Color with a non-zero Emission Strength
#     (the v2 default strength is 0, which would render black).
FLUID_MATERIAL_ITEMS = (
    ("WATER", "Water", "Clear blue-tinted refractive water"),
    ("CLEAR", "Clear", "Colorless refractive liquid / glass"),
    ("HONEY", "Honey", "Thick amber translucent honey"),
    ("JUICE", "Juice", "Orange translucent fruit juice"),
    ("MILK", "Milk", "Opaque white milk (subsurface)"),
    ("LAVA", "Lava", "Molten rock: dark body, glowing emission"),
    ("FOAM", "Foam", "Bright whitewater foam (for instanced spray)"),
)

_FLUID_MATERIAL_SPECS = {
    "WATER": {
        "name": "STFLIP Water", "base_color": (0.55, 0.78, 0.85, 1.0),
        "roughness": 0.02, "ior": 1.333, "transmission": 1.0,
        "refractive": True,
    },
    "CLEAR": {
        "name": "STFLIP Clear", "base_color": (1.0, 1.0, 1.0, 1.0),
        "roughness": 0.01, "ior": 1.45, "transmission": 1.0,
        "refractive": True,
    },
    "HONEY": {
        "name": "STFLIP Honey", "base_color": (0.60, 0.28, 0.05, 1.0),
        "roughness": 0.12, "ior": 1.49, "transmission": 0.9,
        "refractive": True,
    },
    "JUICE": {
        "name": "STFLIP Juice", "base_color": (0.85, 0.30, 0.05, 1.0),
        "roughness": 0.05, "ior": 1.34, "transmission": 0.85,
        "refractive": True,
    },
    "MILK": {
        "name": "STFLIP Milk", "base_color": (0.92, 0.92, 0.94, 1.0),
        "roughness": 0.3, "ior": 1.35, "transmission": 0.0,
        "refractive": False,
        "subsurface": {"weight": 0.5, "radius": (1.0, 0.55, 0.35),
                       "scale": 0.02},
    },
    "LAVA": {
        "name": "STFLIP Lava", "base_color": (0.015, 0.006, 0.004, 1.0),
        "roughness": 0.85, "ior": 1.45, "transmission": 0.0,
        "refractive": False,
        "emission": {"blackbody_k": 1200.0, "strength": 8.0},
    },
    "FOAM": {
        "name": "STFLIP Foam", "base_color": (0.95, 0.96, 0.98, 1.0),
        "roughness": 0.85, "ior": 1.33, "transmission": 0.0,
        "refractive": False,
        "emission": {"color": (0.9, 0.95, 1.0, 1.0), "strength": 0.2},
    },
}

DEFAULT_FLUID_MATERIAL = "WATER"


def _set_bsdf_input(bsdf, name, value):
    """Set a Principled BSDF input if that socket exists in this Blender build."""
    socket = bsdf.inputs.get(name) if hasattr(bsdf.inputs, "get") else None
    if socket is None and name in getattr(bsdf, "inputs", {}):
        socket = bsdf.inputs[name]
    if socket is not None:
        socket.default_value = value


def _set_refraction_flag(mat, enabled):
    """Enable EEVEE Next raytraced refraction defensively across builds.

    ``use_raytrace_refraction`` is the canonical property; ``use_screen_refraction``
    is a still-registered alias for the same bit.  Both are set when present so
    the material refracts once ``scene.eevee.use_raytracing`` is on.
    """
    for attr in ("use_raytrace_refraction", "use_screen_refraction"):
        if hasattr(mat, attr):
            try:
                setattr(mat, attr, bool(enabled))
            except (AttributeError, TypeError):
                pass


def _ensure_principled_bsdf(tree):
    """Return the material's Principled BSDF, rebuilding it if a same-named
    datablock was appended/edited without one so the recipe always applies."""
    bsdf = tree.nodes.get("Principled BSDF")
    if bsdf is None:
        for node in tree.nodes:
            if getattr(node, "type", None) == "BSDF_PRINCIPLED":
                bsdf = node
                break
    if bsdf is None:
        bsdf = tree.nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.name = "Principled BSDF"
        bsdf.location = (0.0, 0.0)
    output = None
    for node in tree.nodes:
        if getattr(node, "type", None) == "OUTPUT_MATERIAL":
            output = node
            break
    if output is None:
        output = tree.nodes.new("ShaderNodeOutputMaterial")
        output.location = (300.0, 0.0)
    surface = output.inputs.get("Surface") \
        if hasattr(output.inputs, "get") else None
    if surface is not None and not surface.is_linked:
        try:
            tree.links.new(bsdf.outputs["BSDF"], surface)
        except (RuntimeError, TypeError, KeyError):
            pass
    return bsdf


def build_fluid_material(kind):
    """Get-or-create the named fluid material and (re)apply its recipe.

    The material is a single persistent datablock reused across the per-frame
    surface rebuild; calling this again refreshes its parameters in place rather
    than allocating a new datablock.
    """
    spec = _FLUID_MATERIAL_SPECS.get(kind, _FLUID_MATERIAL_SPECS[DEFAULT_FLUID_MATERIAL])
    mat = bpy.data.materials.get(spec["name"])
    if mat is None:
        mat = bpy.data.materials.new(spec["name"])
    mat.use_nodes = True
    tree = mat.node_tree
    bsdf = _ensure_principled_bsdf(tree)
    if bsdf is not None:
        _set_bsdf_input(bsdf, "Base Color", spec["base_color"])
        _set_bsdf_input(bsdf, "Roughness", spec["roughness"])
        _set_bsdf_input(bsdf, "IOR", spec["ior"])
        _set_bsdf_input(bsdf, "Transmission Weight", spec["transmission"])
        subsurface = spec.get("subsurface")
        _set_bsdf_input(bsdf, "Subsurface Weight",
                        subsurface["weight"] if subsurface else 0.0)
        if subsurface:
            _set_bsdf_input(bsdf, "Subsurface Radius", subsurface["radius"])
            _set_bsdf_input(bsdf, "Subsurface Scale", subsurface["scale"])
        emission = spec.get("emission")
        if emission and "color" in emission:
            _set_bsdf_input(bsdf, "Emission Color", emission["color"])
        _set_bsdf_input(bsdf, "Emission Strength",
                        emission["strength"] if emission else 0.0)
        if emission and "blackbody_k" in emission:
            _link_blackbody_emission(tree, bsdf, emission["blackbody_k"])
    _set_refraction_flag(mat, spec.get("refractive", False))
    return mat


def _link_blackbody_emission(tree, bsdf, temperature_k):
    """Feed a Blackbody colour into the BSDF Emission Color (molten look)."""
    node = tree.nodes.get("STFLIP Blackbody")
    if node is None:
        node = tree.nodes.new("ShaderNodeBlackbody")
        node.name = "STFLIP Blackbody"
        node.location = (-320.0, -220.0)
    node.inputs["Temperature"].default_value = float(temperature_k)
    emission_socket = bsdf.inputs.get("Emission Color") \
        if hasattr(bsdf.inputs, "get") else None
    if emission_socket is not None and not emission_socket.is_linked:
        try:
            tree.links.new(node.outputs["Color"], emission_socket)
        except (RuntimeError, TypeError):
            pass


def _selected_fluid_kind(scene):
    """The scene's chosen fluid material key, or None when unavailable."""
    settings = getattr(scene, "stflip", None) if scene is not None else None
    kind = getattr(settings, "fluid_material", None) if settings else None
    return kind if kind in _FLUID_MATERIAL_SPECS else None


def resolve_fluid_material(scene=None):
    """Return the material for the scene's selected fluid, defaulting to water.

    Water (and the no-selection fallback) delegates to :func:`ensure_water_material`
    so existing callers and tests that stub that factory keep working.
    """
    if scene is None:
        scene = getattr(bpy.context, "scene", None)
    kind = _selected_fluid_kind(scene)
    if kind is None or kind == "WATER":
        return ensure_water_material()
    return build_fluid_material(kind)


def ensure_water_material():
    return build_fluid_material("WATER")


def apply_material_to_surface(obj, material) -> bool:
    """Assign ``material`` to a surface object via both delivery paths.

    Fast-preview surfaces receive it through the Geometry-Nodes group's
    "Material" input socket; paper-MCF surfaces are plain meshes, so it is also
    written to the mesh material slot.  Returns True if either path applied.
    """
    if obj is None or material is None:
        return False
    applied = False
    mod = None
    modifiers = getattr(obj, "modifiers", None)
    if modifiers is not None and hasattr(modifiers, "get"):
        mod = modifiers.get(SURFACE_MODIFIER)
    node_group = getattr(mod, "node_group", None) if mod is not None else None
    if node_group is not None:
        ident = _socket_identifier(node_group, "Material")
        if ident is not None:
            try:
                mod[ident] = material
                applied = True
            except (KeyError, TypeError):
                pass
    mesh = getattr(obj, "data", None)
    materials = getattr(mesh, "materials", None) if mesh is not None else None
    if materials is not None:
        try:
            if len(materials):
                materials[0] = material
            else:
                materials.append(material)
            applied = True
        except (AttributeError, IndexError, RuntimeError, TypeError):
            pass
    if hasattr(obj, "update_tag"):
        obj.update_tag()
    return applied


def _socket_identifier(ng, name):
    for item in ng.interface.items_tree:
        if getattr(item, "name", None) == name and item.item_type == "SOCKET" \
                and item.in_out == "INPUT":
            return item.identifier
    return None


def _bound_scene_object(property_name):
    scene = getattr(bpy.context, "scene", None)
    settings = getattr(scene, "stflip", None) if scene is not None else None
    return getattr(settings, property_name, None) if settings is not None else None


def _id_key(value):
    """Return a stable identity key for Blender RNA values and test doubles."""
    try:
        return ("RNA", int(value.as_pointer()))
    except (AttributeError, ReferenceError, TypeError, ValueError):
        return ("PY", id(value))


def _object_in_scene(scene, obj):
    objects = getattr(scene, "objects", None)
    if objects is None or obj is None:
        return False
    try:
        candidate = objects.get(obj.name)
        if candidate is not None:
            return _id_key(candidate) == _id_key(obj)
    except (AttributeError, ReferenceError, TypeError):
        pass
    try:
        return obj in objects
    except (ReferenceError, TypeError):
        return False


def _scenes():
    try:
        return list(bpy.data.scenes)
    except (AttributeError, ReferenceError, TypeError):
        return []


def _collection_values(collection):
    values = getattr(collection, "values", None)
    try:
        return list(values()) if values is not None else list(collection)
    except (AttributeError, ReferenceError, TypeError):
        return []


def _all_objects():
    objects = _collection_values(getattr(bpy.data, "objects", ()))
    if not objects:
        objects = []
        seen = set()
        for scene in _scenes():
            candidates = _collection_values(getattr(scene, "objects", ()))
            for candidate in candidates:
                key = _id_key(candidate)
                if key not in seen:
                    seen.add(key)
                    objects.append(candidate)
    return objects


def output_is_exclusive(scene, obj):
    """Whether an output object and its mesh are safe for this scene to edit.

    Blender copies scene pointer properties verbatim. A normal ``Scene.copy``
    therefore leaves both scenes pointing at the same output object and mesh.
    Cached-frame playback mutates mesh vertices in place, so accepting either
    a cross-scene object or a mesh used by another object would make one scene
    overwrite another scene's visible bake.
    """
    if obj is None or getattr(obj, "type", None) != "MESH":
        return False
    mesh = getattr(obj, "data", None)
    if mesh is None:
        return False
    if (getattr(obj, "library", None) is not None
            or getattr(mesh, "library", None) is not None):
        return False
    # Generated outputs normally have one object link and one mesh user. These
    # ID counters are constant-time and keep frame playback independent of the
    # number of scenes/objects in ordinary files. Only ambiguous multi-user
    # datablocks need the more expensive ownership scans below.
    try:
        if int(obj.users) <= 1 and int(mesh.users) <= 1:
            return True
    except (AttributeError, ReferenceError, TypeError, ValueError):
        pass
    scene_key = _id_key(scene)
    try:
        users_scene = list(obj.users_scene)
    except (AttributeError, ReferenceError, TypeError):
        users_scene = []
    if users_scene:
        if any(_id_key(user) != scene_key for user in users_scene):
            return False
    else:
        for other_scene in _scenes():
            if _id_key(other_scene) == scene_key:
                continue
            if _object_in_scene(other_scene, obj):
                return False
    # The normal playback path has one mesh user, so avoid scanning every
    # Blender object on every frame. More than one ID user needs the scan to
    # distinguish a second object from a harmless fake-user flag.
    try:
        if int(mesh.users) <= 1:
            return True
    except (AttributeError, ReferenceError, TypeError, ValueError):
        pass
    for candidate in _all_objects():
        if _id_key(candidate) == _id_key(obj):
            continue
        candidate_mesh = getattr(candidate, "data", None)
        if (candidate_mesh is not None
                and _id_key(candidate_mesh) == _id_key(mesh)):
            return False
    return True


def _scene_collections(scene):
    root = getattr(scene, "collection", None)
    if root is None:
        return []
    result = []
    stack = [root]
    seen = set()
    while stack:
        collection = stack.pop()
        key = _id_key(collection)
        if key in seen:
            continue
        seen.add(key)
        result.append(collection)
        try:
            stack.extend(list(collection.children))
        except (AttributeError, ReferenceError, TypeError):
            pass
    return result


def _detach_rejected_output(scene, obj):
    """Unlink a copied output from collections owned only by ``scene``.

    Generated outputs are linked directly to the scene's root collection. A
    copied scene gets a distinct root containing the same objects, so unlinking
    there is safe and prevents both the stale shared output and its fresh local
    replacement from being visible together. Shared child collections are
    deliberately left untouched because unlinking from one would affect every
    scene that uses it.
    """
    if scene is None or obj is None:
        return
    current = _scene_collections(scene)
    other_keys = {
        _id_key(collection)
        for other_scene in _scenes()
        if _id_key(other_scene) != _id_key(scene)
        for collection in _scene_collections(other_scene)
    }
    for collection in current:
        if _id_key(collection) in other_keys:
            continue
        objects = getattr(collection, "objects", None)
        try:
            linked = objects is not None and obj.name in objects
        except (AttributeError, ReferenceError, TypeError):
            linked = False
        if not linked:
            continue
        try:
            objects.unlink(obj)
        except (AttributeError, ReferenceError, RuntimeError, TypeError):
            pass


def scene_exclusive_output(scene, obj):
    """Return a safe local output, detaching a copied/shared one if needed."""
    if obj is None or getattr(obj, "type", None) != "MESH":
        return None
    if output_is_exclusive(scene, obj):
        return obj
    _detach_rejected_output(scene, obj)
    return None


def _mesh_output(existing_obj, property_name, fallback_name):
    """Prefer a scene binding, then a same-scene legacy-named output.

    Blender object datablocks are global. Reusing a canonical name from a
    different scene would make both scenes write into the same output mesh.
    """
    scene = getattr(bpy.context, "scene", None)
    obj = existing_obj
    if obj is None:
        obj = _bound_scene_object(property_name)
    if obj is not None and getattr(obj, "type", None) == "MESH":
        local = scene_exclusive_output(scene, obj)
        if local is not None:
            return local
    obj = bpy.data.objects.get(fallback_name)
    if (obj is not None and getattr(obj, "type", None) == "MESH"
            and scene is not None and obj.name in scene.objects):
        local = scene_exclusive_output(scene, obj)
        if local is not None:
            return local
    return None


def _bind_scene_object(property_name, obj):
    scene = getattr(bpy.context, "scene", None)
    settings = getattr(scene, "stflip", None) if scene is not None else None
    if settings is not None:
        setattr(settings, property_name, obj)


def _link_if_needed(obj):
    scene = getattr(bpy.context, "scene", None)
    if scene is not None and obj.name not in scene.objects:
        scene.collection.objects.link(obj)


def _ensure_surface_output(existing_obj=None):
    """Return a scene-exclusive surface object, creating one if necessary."""
    obj = _mesh_output(existing_obj, "surface_object", SURFACE_OBJ)
    if obj is None:
        mesh = bpy.data.meshes.new(SURFACE_OBJ)
        obj = bpy.data.objects.new(SURFACE_OBJ, mesh)
        bpy.context.scene.collection.objects.link(obj)
    else:
        _link_if_needed(obj)
    _bind_scene_object("surface_object", obj)
    return obj


def _set_modifier_enabled(obj, name: str, enabled: bool):
    """Enable/disable one generated modifier without destroying its setup."""
    if obj is None:
        return None
    modifier = obj.modifiers.get(name)
    if modifier is None:
        return None
    modifier.show_viewport = bool(enabled)
    modifier.show_render = bool(enabled)
    return modifier


def _stamp_surface_method(obj, method: str) -> None:
    try:
        obj["stflip_surface_method"] = method
    except (AttributeError, KeyError, RuntimeError, TypeError):
        pass


def ensure_surface_object(particle_obj, dx: float, radius_factor: float,
                          voxel_factor: float, existing_obj=None):
    """Create/update the surface object driven by the particle object."""
    ng = build_node_group()
    obj = _ensure_surface_output(existing_obj)

    mod = obj.modifiers.get(SURFACE_MODIFIER)
    if mod is None:
        mod = obj.modifiers.new(SURFACE_MODIFIER, "NODES")
    mod.node_group = ng
    mod.show_viewport = True
    mod.show_render = True

    for name, value in (("Points Object", particle_obj),
                        ("Radius", dx * radius_factor),
                        ("Voxel Size", dx * voxel_factor),
                        ("Material", resolve_fluid_material())):
        ident = _socket_identifier(ng, name)
        if ident is not None:
            mod[ident] = value
    _stamp_surface_method(obj, "FAST_PREVIEW")
    obj.update_tag()
    return obj


def restore_preview_surface(particle_obj, dx: float, radius_factor: float,
                            voxel_factor: float, existing_obj=None):
    """Restore the deterministic Geometry Nodes preview after paper mode."""
    return ensure_surface_object(
        particle_obj,
        dx,
        radius_factor,
        voxel_factor,
        existing_obj=existing_obj,
    )


def _polygon_array(value, width: int, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.size == 0:
        return np.empty((0, width), dtype=np.int32)
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"{name} must have shape (N, {width})")
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must contain integer vertex indices")
    return np.ascontiguousarray(array, dtype=np.int64)


def _validated_polygon_mesh(vertices, triangles, quads):
    try:
        points = np.asarray(vertices)
    except (TypeError, ValueError) as exc:
        raise ValueError("vertices must be a finite numeric (N, 3) array") from exc
    if points.size == 0:
        points = np.empty((0, 3), dtype=np.float32)
    if (points.ndim != 2 or points.shape[1:] != (3,)
            or not np.issubdtype(points.dtype, np.number)
            or not np.isrealobj(points)):
        raise ValueError("vertices must be a finite numeric (N, 3) array")
    try:
        finite = bool(np.all(np.isfinite(points)))
    except TypeError as exc:
        raise ValueError("vertices must be a finite numeric (N, 3) array") from exc
    if not finite:
        raise ValueError("vertices must be a finite numeric (N, 3) array")
    points = np.ascontiguousarray(points, dtype=np.float32)
    tris = _polygon_array(triangles, 3, "triangles")
    quad_array = _polygon_array(quads, 4, "quads")
    for name, faces in (("triangles", tris), ("quads", quad_array)):
        if faces.size and (
                int(faces.min()) < 0 or int(faces.max()) >= len(points)):
            raise ValueError(f"{name} contain an out-of-range vertex index")
    return points, tris, quad_array


def density_field_to_polygons(
    density,
    origin,
    voxel_size: float,
    isovalue: float = 0.5,
    adaptivity: float = 0.0,
):
    """Extract a world-space polygon mesh with Blender's OpenVDB module.

    OpenVDB is imported lazily so the add-on and its CPU/CUDA solver remain
    usable in Blender builds where the optional Python module is unavailable.
    Meshing itself is CPU work and is not accelerated by the CUDA solver.
    """
    try:
        values = np.asarray(density)
    except (TypeError, ValueError) as exc:
        raise ValueError("density must be a finite numeric 3D array") from exc
    if (values.ndim != 3 or any(axis <= 0 for axis in values.shape)
            or not np.issubdtype(values.dtype, np.number)
            or not np.isrealobj(values)):
        raise ValueError("density must be a finite numeric 3D array")
    try:
        finite_density = bool(np.all(np.isfinite(values)))
    except TypeError as exc:
        raise ValueError("density must be a finite numeric 3D array") from exc
    if not finite_density:
        raise ValueError("density must be a finite numeric 3D array")
    values = np.ascontiguousarray(values, dtype=np.float32)

    try:
        world_origin = np.asarray(origin, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("origin must contain three finite values") from exc
    if world_origin.shape != (3,) or not bool(np.all(np.isfinite(world_origin))):
        raise ValueError("origin must contain three finite values")

    try:
        voxel = float(voxel_size)
        level = float(isovalue)
        adaptive = float(adaptivity)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "voxel_size, isovalue, and adaptivity must be finite scalars"
        ) from exc
    if not math.isfinite(voxel) or voxel <= 0.0:
        raise ValueError("voxel_size must be finite and positive")
    if not math.isfinite(level):
        raise ValueError("isovalue must be finite")
    if not math.isfinite(adaptive) or not 0.0 <= adaptive <= 1.0:
        raise ValueError("adaptivity must be between 0 and 1")

    openvdb = _load_openvdb()

    try:
        grid = openvdb.FloatGrid()
        grid.copyFromArray(values)
        transform = openvdb.createLinearTransform(voxelSize=voxel)
        translate = getattr(transform, "postTranslate", None)
        if translate is None:
            translate = getattr(transform, "translate", None)
        if translate is None:
            raise AttributeError(
                "the bundled OpenVDB transform has no translation API"
            )
        translate(tuple(float(value) for value in world_origin))
        grid.transform = transform
        vertices, triangles, quads = grid.convertToPolygons(
            isovalue=level,
            adaptivity=adaptive,
        )
    except Exception as exc:
        raise RuntimeError(f"OpenVDB surface extraction failed: {exc}") from exc
    return _validated_polygon_mesh(vertices, triangles, quads)


def _assign_water_material(mesh) -> None:
    material = resolve_fluid_material()
    materials = getattr(mesh, "materials", None)
    if materials is None:
        return
    try:
        if len(materials):
            materials[0] = material
        else:
            materials.append(material)
    except (AttributeError, IndexError, RuntimeError, TypeError):
        try:
            materials.clear()
            materials.append(material)
        except (AttributeError, RuntimeError, TypeError):
            return
    for polygon in getattr(mesh, "polygons", ()):
        try:
            polygon.material_index = 0
        except (AttributeError, RuntimeError, TypeError):
            pass


def update_paper_surface_mesh(obj, vertices, triangles, quads):
    """Replace a scene-exclusive output with an ordinary polygon mesh."""
    if obj is None or getattr(obj, "type", None) != "MESH":
        raise ValueError("paper surface output must be a Blender mesh object")
    points, tris, quad_array = _validated_polygon_mesh(
        vertices, triangles, quads)
    mesh = getattr(obj, "data", None)
    if mesh is None or not hasattr(mesh, "from_pydata"):
        raise ValueError("paper surface output has no editable mesh datablock")
    faces = [tuple(int(index) for index in face) for face in tris]
    faces.extend(tuple(int(index) for index in face) for face in quad_array)
    try:
        mesh.clear_geometry()
        mesh.from_pydata(points.tolist(), [], faces)
        _assign_water_material(mesh)
        mesh.update()
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError(f"failed to update paper surface mesh: {exc}") from exc
    _set_modifier_enabled(obj, SURFACE_MODIFIER, False)
    _set_modifier_enabled(obj, SMOOTH_MODIFIER, False)
    _stamp_surface_method(obj, "PAPER_MCF")
    obj.update_tag()
    return obj


def ensure_paper_surface_object(
    vertices,
    triangles,
    quads,
    existing_obj=None,
):
    """Create or update the scene-owned plain mesh used by paper surfacing."""
    obj = _ensure_surface_output(existing_obj)
    return update_paper_surface_mesh(obj, vertices, triangles, quads)


def configure_surface_smoothing(obj, enabled: bool, iterations: int,
                                factor: float):
    """Configure Blender-only post-process smoothing on a surface object.

    This intentionally uses Blender's Laplacian Smooth modifier.  It improves
    the fast-preview mesh without changing cached particles and is distinct
    from the paper MCF mode's cached density-field reconstruction.
    """
    if obj is None or getattr(obj, "type", None) != "MESH":
        return None
    modifier = obj.modifiers.get(SMOOTH_MODIFIER)
    if modifier is None:
        modifier = obj.modifiers.new(SMOOTH_MODIFIER, "LAPLACIANSMOOTH")
    modifier.iterations = max(1, int(iterations))
    modifier.lambda_factor = float(factor)
    if hasattr(modifier, "lambda_border"):
        modifier.lambda_border = float(factor)
    if hasattr(modifier, "use_volume_preserve"):
        modifier.use_volume_preserve = True
    modifier.show_viewport = bool(enabled)
    modifier.show_render = bool(enabled)
    # Keep geometric smoothing after the generated Geometry Nodes surface.
    try:
        index = list(obj.modifiers).index(modifier)
        obj.modifiers.move(index, len(obj.modifiers) - 1)
    except (AttributeError, RuntimeError, ValueError):
        pass
    obj.update_tag()
    return modifier


def ensure_particle_object(existing_obj=None):
    """Return the scene-bound particle mesh, including after user renames."""
    obj = _mesh_output(existing_obj, "particle_object", PARTICLE_OBJ)
    if obj is None:
        mesh = bpy.data.meshes.new(PARTICLE_OBJ)
        obj = bpy.data.objects.new(PARTICLE_OBJ, mesh)
        bpy.context.scene.collection.objects.link(obj)
    else:
        _link_if_needed(obj)
    _bind_scene_object("particle_object", obj)
    return obj
