"""Geometry Nodes surfacing: points -> volume -> mesh.

Builds a node group that pulls the baked particle object's vertices, splats
them as spheres into a volume, and meshes the result -- the standard Blender
way to get a renderable liquid surface that works with materials, motion
blur (via the velocity attribute), and normal modifier stacks.
"""

import bpy

GROUP_NAME = "STFLIP_Surface"
SURFACE_OBJ = "STFLIP Liquid Surface"
PARTICLE_OBJ = "STFLIP Particles"
GROUP_SCHEMA_KEY = "stflip_schema_version"
GROUP_SCHEMA_VERSION = 2

_INTERFACE_SCHEMA = (
    ("Geometry", "INPUT", "NodeSocketGeometry"),
    ("Points Object", "INPUT", "NodeSocketObject"),
    ("Radius", "INPUT", "NodeSocketFloat"),
    ("Voxel Size", "INPUT", "NodeSocketFloat"),
    ("Material", "INPUT", "NodeSocketMaterial"),
    ("Geometry", "OUTPUT", "NodeSocketGeometry"),
)


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

    vtm = ng.nodes.new("GeometryNodeVolumeToMesh")
    vtm.location = (0, 0)
    _set_resolution_mode(vtm, "GRID", "Grid")

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


def ensure_water_material():
    mat = bpy.data.materials.get("STFLIP Water")
    if mat is not None:
        return mat
    mat = bpy.data.materials.new("STFLIP Water")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = (0.02, 0.3, 0.75, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.08
        if "Transmission Weight" in bsdf.inputs:
            bsdf.inputs["Transmission Weight"].default_value = 0.5
        if "IOR" in bsdf.inputs:
            bsdf.inputs["IOR"].default_value = 1.33
    return mat


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


def _mesh_output(existing_obj, property_name, fallback_name):
    """Prefer a scene binding, then a same-scene legacy-named output.

    Blender object datablocks are global. Reusing a canonical name from a
    different scene would make both scenes write into the same output mesh.
    """
    obj = existing_obj
    if obj is None:
        obj = _bound_scene_object(property_name)
    if obj is not None and getattr(obj, "type", None) == "MESH":
        return obj
    obj = bpy.data.objects.get(fallback_name)
    scene = getattr(bpy.context, "scene", None)
    if (obj is not None and getattr(obj, "type", None) == "MESH"
            and scene is not None and obj.name in scene.objects):
        return obj
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


def ensure_surface_object(particle_obj, dx: float, radius_factor: float,
                          voxel_factor: float, existing_obj=None):
    """Create/update the surface object driven by the particle object."""
    ng = build_node_group()

    obj = _mesh_output(existing_obj, "surface_object", SURFACE_OBJ)
    if obj is None:
        mesh = bpy.data.meshes.new(SURFACE_OBJ)
        obj = bpy.data.objects.new(SURFACE_OBJ, mesh)
        bpy.context.scene.collection.objects.link(obj)
    else:
        _link_if_needed(obj)

    mod = obj.modifiers.get("STFLIP Surface")
    if mod is None:
        mod = obj.modifiers.new("STFLIP Surface", "NODES")
    mod.node_group = ng

    for name, value in (("Points Object", particle_obj),
                        ("Radius", dx * radius_factor),
                        ("Voxel Size", dx * voxel_factor),
                        ("Material", ensure_water_material())):
        ident = _socket_identifier(ng, name)
        if ident is not None:
            mod[ident] = value
    obj.update_tag()
    _bind_scene_object("surface_object", obj)
    return obj


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
