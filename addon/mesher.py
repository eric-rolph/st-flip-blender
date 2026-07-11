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
SURFACE_MODIFIER = "STFLIP Surface"
SMOOTH_MODIFIER = "STFLIP Geometric Smoothing"
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

    mod = obj.modifiers.get(SURFACE_MODIFIER)
    if mod is None:
        mod = obj.modifiers.new(SURFACE_MODIFIER, "NODES")
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


def configure_surface_smoothing(obj, enabled: bool, iterations: int,
                                factor: float):
    """Configure Blender-only post-process smoothing on a surface object.

    This intentionally uses Blender's Laplacian Smooth modifier.  It improves
    the viewport/render mesh without changing cached particles and must not be
    confused with the paper's unavailable mean-curvature-flow reconstruction.
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
