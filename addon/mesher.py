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


def _new_socket(ng, name, in_out, socket_type):
    return ng.interface.new_socket(name=name, in_out=in_out,
                                   socket_type=socket_type)


def build_node_group():
    ng = bpy.data.node_groups.get(GROUP_NAME)
    if ng is not None:
        return ng
    ng = bpy.data.node_groups.new(GROUP_NAME, "GeometryNodeTree")
    ng.is_modifier = True

    _new_socket(ng, "Geometry", "INPUT", "NodeSocketGeometry")
    s_obj = _new_socket(ng, "Points Object", "INPUT", "NodeSocketObject")
    s_rad = _new_socket(ng, "Radius", "INPUT", "NodeSocketFloat")
    s_vox = _new_socket(ng, "Voxel Size", "INPUT", "NodeSocketFloat")
    _new_socket(ng, "Geometry", "OUTPUT", "NodeSocketGeometry")
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
    ptv.resolution_mode = "VOXEL_SIZE"

    vtm = ng.nodes.new("GeometryNodeVolumeToMesh")
    vtm.location = (0, 0)
    vtm.resolution_mode = "GRID"

    smooth = ng.nodes.new("GeometryNodeSetShadeSmooth")
    smooth.location = (200, 0)

    ln = ng.links.new
    ln(n_in.outputs["Points Object"], info.inputs["Object"])
    ln(info.outputs["Geometry"], mtp.inputs["Mesh"])
    ln(n_in.outputs["Radius"], mtp.inputs["Radius"])
    ln(mtp.outputs["Points"], ptv.inputs["Points"])
    ln(n_in.outputs["Radius"], ptv.inputs["Radius"])
    ln(n_in.outputs["Voxel Size"], ptv.inputs["Voxel Size"])
    ln(ptv.outputs["Volume"], vtm.inputs["Volume"])
    ln(vtm.outputs["Mesh"], smooth.inputs["Geometry"])
    ln(smooth.outputs["Geometry"], n_out.inputs["Geometry"])
    return ng


def _socket_identifier(ng, name):
    for item in ng.interface.items_tree:
        if getattr(item, "name", None) == name and item.item_type == "SOCKET" \
                and item.in_out == "INPUT":
            return item.identifier
    return None


def ensure_surface_object(particle_obj, dx: float, radius_factor: float,
                          voxel_factor: float):
    """Create/update the surface object driven by the particle object."""
    ng = build_node_group()

    obj = bpy.data.objects.get(SURFACE_OBJ)
    if obj is None:
        mesh = bpy.data.meshes.new(SURFACE_OBJ)
        obj = bpy.data.objects.new(SURFACE_OBJ, mesh)
        bpy.context.scene.collection.objects.link(obj)

    mod = obj.modifiers.get("STFLIP Surface")
    if mod is None:
        mod = obj.modifiers.new("STFLIP Surface", "NODES")
    mod.node_group = ng

    for name, value in (("Points Object", particle_obj),
                        ("Radius", dx * radius_factor),
                        ("Voxel Size", dx * voxel_factor)):
        ident = _socket_identifier(ng, name)
        if ident is not None:
            mod[ident] = value
    obj.update_tag()
    return obj


def ensure_particle_object():
    obj = bpy.data.objects.get(PARTICLE_OBJ)
    if obj is None:
        mesh = bpy.data.meshes.new(PARTICLE_OBJ)
        obj = bpy.data.objects.new(PARTICLE_OBJ, mesh)
        bpy.context.scene.collection.objects.link(obj)
    return obj
