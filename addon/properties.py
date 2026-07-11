"""Property groups for scenes (simulation settings) and objects (roles)."""

import bpy
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)

from ..stflip.experiments import PROFILE_ENUM_ITEMS


class STFLIPObjectSettings(bpy.types.PropertyGroup):
    role: EnumProperty(
        name="Role",
        items=[
            ("NONE", "None", "Not part of the simulation"),
            ("LIQUID", "Liquid", "Initial liquid volume (closed mesh)"),
            ("INFLOW", "Inflow", "Continuously emits liquid (closed mesh)"),
            ("OBSTACLE", "Obstacle", "Solid obstacle (closed mesh)"),
        ],
        default="NONE",
    )
    inflow_velocity: FloatVectorProperty(
        name="Inflow Velocity", subtype="VELOCITY", size=3,
        default=(0.0, 0.0, 0.0),
    )
    initial_velocity: FloatVectorProperty(
        name="Initial Velocity", subtype="VELOCITY", size=3,
        default=(0.0, 0.0, 0.0),
        description="Uniform starting velocity for particles seeded from "
                    "this liquid volume",
    )


class STFLIPSettings(bpy.types.PropertyGroup):
    experiment_profile: EnumProperty(
        name="Experiment Profile",
        items=PROFILE_ENUM_ITEMS,
        default="CUSTOM",
        description="Paper-inspired parameter snapshot; profiles do not "
                    "replace the scene geometry or unsupported baselines",
    )
    collect_metrics: BoolProperty(
        name="Record Frame Metrics", default=False,
        description="Write strict per-frame diagnostics to the bake cache",
    )
    collect_enstrophy: BoolProperty(
        name="Compute Enstrophy", default=False,
        description="Compute the paper's vorticity diagnostic from the MAC "
                    "grid; adds grid-wide work and GPU synchronization",
    )
    domain: PointerProperty(
        name="Domain", type=bpy.types.Object,
        description="Axis-aligned box defining the simulation region",
    )
    resolution: IntProperty(
        name="Resolution", default=64, min=8, soft_max=128, max=512,
        description="Grid cells along the longest domain axis. Above ~128 "
                    "scene voxelization becomes slow; bake setup estimates "
                    "RAM/VRAM and blocks settings that cannot fit safely",
    )
    cfl_target: FloatProperty(
        name="Target CFL", default=8.0, min=0.5, max=30.0,
        description="Time-step size in grid cells travelled per step. "
                    "Standard FLIP uses 1-2; ST-FLIP stays coherent at 8-15+",
    )
    particles_per_cell: IntProperty(
        name="Particles / Cell", default=8, min=1, max=64,
        description="Initial samples per occupied cell. The paper sweeps "
                    "1-16 against a 50-particle reference; higher values "
                    "increase RAM/VRAM use",
    )
    seed: IntProperty(
        name="Random Seed", default=0, min=0, max=2_147_483_647,
        description="Seed for deterministic particle placement and temporal "
                    "jitter; use the same value for comparable reruns",
    )
    flip_blend: FloatProperty(
        name="FLIP Fraction", default=0.98, min=0.0, max=1.0,
        description="FLIP/PIC blend factor (1 = pure FLIP)",
    )
    st_enabled: BoolProperty(
        name="Spatiotemporal Sampling", default=True,
        description="Enable ST-FLIP temporal weighting and jitter. Disable "
                    "for an instantaneous-P2G ablation, not a full "
                    "standard-FLIP/GFM baseline",
    )
    jitter_strength: FloatProperty(
        name="Jitter Strength", default=1.0, min=0.0, max=1.0,
        description="Base temporal jitter strength (gamma)",
    )
    adaptive_gamma: BoolProperty(
        name="Adaptive Attenuation", default=True,
        description="Reduce jitter noise on calm surfaces (paper Sec. 3.10)",
    )
    eta_phi: FloatProperty(
        name="Interface Steepness", default=0.5, min=0.1, max=2.0,
        description="Phase-field eta (paper Eq. 13): smaller steepens the "
                    "transition and levels sampling wells more aggressively; "
                    "larger preserves more detail but also more noise",
    )
    backend: EnumProperty(
        name="Compute Backend",
        items=[
            ("auto", "Auto", "Use CUDA GPU when available, else CPU"),
            ("cpu", "CPU (NumPy)", "Portable CPU backend"),
            ("cuda", "GPU (CUDA)", "NVIDIA GPU via compute-tested CuPy; "
                                     "falls back to CPU with a warning"),
        ],
        default="auto",
    )
    cache_dir: StringProperty(
        name="Cache Directory", subtype="DIR_PATH", default="//stflip_cache",
    )
    create_surface: BoolProperty(
        name="Create Surface", default=True,
        description="Attach a Geometry Nodes points-to-mesh surface",
    )
    particle_radius: FloatProperty(
        name="Particle Radius", default=0.5, min=0.1, max=2.0,
        description="Surfacing sphere radius in cell widths",
    )
    surface_voxel: FloatProperty(
        name="Surface Voxel", default=0.5, min=0.1, max=2.0,
        description="Surfacing voxel size in cell widths",
    )
    bake_status: StringProperty(name="Bake Status", default="")
    # Robust bindings to the bake outputs (survive renames; null on delete).
    particle_object: PointerProperty(type=bpy.types.Object)
    surface_object: PointerProperty(type=bpy.types.Object)


def register():
    bpy.utils.register_class(STFLIPObjectSettings)
    bpy.utils.register_class(STFLIPSettings)
    bpy.types.Scene.stflip = PointerProperty(type=STFLIPSettings)
    bpy.types.Object.stflip = PointerProperty(type=STFLIPObjectSettings)


def unregister():
    del bpy.types.Object.stflip
    del bpy.types.Scene.stflip
    bpy.utils.unregister_class(STFLIPSettings)
    bpy.utils.unregister_class(STFLIPObjectSettings)
