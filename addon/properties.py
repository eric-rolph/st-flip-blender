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


_VELOCITY_MODE_ITEMS = [
    ("UNIFORM", "Uniform", "Use one world-space velocity vector"),
    (
        "SOLID_BODY",
        "Solid Body Rotation",
        "Use linear velocity plus a world-space rigid rotation",
    ),
]


class STFLIPObjectSettings(bpy.types.PropertyGroup):
    role: EnumProperty(
        name="Role",
        items=[
            ("NONE", "None", "Not part of the simulation"),
            ("LIQUID", "Liquid", "Initial liquid volume (closed mesh)"),
            (
                "INFLOW",
                "Inflow",
                "Emits liquid from a closed mesh, optionally over a frame "
                "range",
            ),
            (
                "OUTFLOW",
                "Outflow",
                "Removes liquid through a closed-mesh outlet volume",
            ),
            ("OBSTACLE", "Obstacle", "Solid obstacle (closed mesh)"),
        ],
        default="NONE",
    )
    inflow_velocity: FloatVectorProperty(
        name="Inflow Velocity", subtype="VELOCITY", size=3,
        default=(0.0, 0.0, 0.0),
        description="World-space linear velocity for particles emitted by "
                    "this inflow; in Solid Body mode it is superposed on "
                    "the rotational velocity",
    )
    inflow_velocity_mode: EnumProperty(
        name="Inflow Velocity Mode",
        items=_VELOCITY_MODE_ITEMS,
        default="UNIFORM",
    )
    inflow_is_gas: BoolProperty(
        name="Emit Gas", default=False,
        description="Emit gas particles instead of liquid (two-phase only)",
    )
    inflow_use_frame_range: BoolProperty(
        name="Limit Active Frames",
        default=False,
        description="Emit into evolved output frames from inclusive Start "
                    "through End; the initial cache frame is a pre-step "
                    "snapshot. Disable for the whole bake and extensions",
    )
    inflow_start_frame: IntProperty(
        name="Start Frame",
        default=1,
        min=-1_048_574,
        max=1_048_574,
        soft_min=1,
        soft_max=250,
        description="First evolved output frame receiving this inflow "
                    "(inclusive)",
    )
    inflow_end_frame: IntProperty(
        name="End Frame",
        default=250,
        min=-1_048_574,
        max=1_048_574,
        soft_min=1,
        soft_max=250,
        description="Last evolved output frame receiving this inflow "
                    "(inclusive)",
    )
    initial_velocity: FloatVectorProperty(
        name="Initial Velocity", subtype="VELOCITY", size=3,
        default=(0.0, 0.0, 0.0),
        description="World-space linear starting velocity for particles "
                    "seeded from this liquid volume; in Solid Body mode it "
                    "is superposed on the rotational velocity",
    )
    initial_velocity_mode: EnumProperty(
        name="Initial Velocity Mode",
        items=_VELOCITY_MODE_ITEMS,
        default="UNIFORM",
    )
    rotation_center_world: FloatVectorProperty(
        name="Rotation Center (World)",
        subtype="TRANSLATION",
        size=3,
        default=(0.0, 0.0, 0.0),
        description="Rotation center in Blender world coordinates; it is "
                    "not transformed with the source object",
    )
    rotation_axis_world: FloatVectorProperty(
        name="Rotation Axis (World)",
        subtype="DIRECTION",
        size=3,
        default=(0.0, 0.0, 1.0),
        description="World-space rotation axis; normalized when baking and "
                    "not transformed with the source object",
    )
    angular_speed: FloatProperty(
        name="Angular Speed (rad/s)",
        default=0.1,
        soft_min=-10.0,
        soft_max=10.0,
        precision=4,
        description="Signed angular speed in radians per scene second; "
                    "positive values follow the right-hand rule",
    )
    outflow_mode: EnumProperty(
        name="Outflow Mode",
        items=[
            (
                "VOLUME",
                "Volume Sink",
                "Delete particles inside this volume; this is a geometric "
                "sink, not a pressure boundary",
            ),
            (
                "PRESSURE",
                "Pressure Outlet",
                "Open covered exterior domain faces at atmospheric pressure "
                "and remove particles after they cross those faces; the mesh "
                "must intersect a domain boundary",
            ),
        ],
        default="VOLUME",
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
    transfer: EnumProperty(
        name="Transfer",
        items=[
            ("flip", "FLIP", "FLIP/PIC blend (detail-preserving, noisier)"),
            ("apic", "APIC", "Affine PIC (low dissipation, smooth, stable)"),
            ("pic", "PIC", "Pure PIC (very smooth, dissipative)"),
        ],
        default="flip",
        description="Particle-grid velocity transfer scheme (paper Sec. 3.9)",
    )
    two_phase: BoolProperty(
        name="Two-Phase (Gas)", default=False,
        description="Couple a light gas phase to the liquid so air can drive "
                    "splashes and rising bubbles (glugging). Fills the domain "
                    "with gas particles; not combined with the sparse grid",
    )
    rho_gas: FloatProperty(
        name="Gas Density", default=1.2, min=1e-3, soft_max=100.0,
        description="Density of the gas phase (air ~ 1.2 mass/unit^3)",
    )
    gas_particles_per_cell: IntProperty(
        name="Gas Particles / Cell", default=8, min=1, max=64,
    )
    surface_tension: FloatProperty(
        name="Surface Tension", default=0.0, min=0.0, soft_max=1.0,
        description="CSF surface-tension coefficient sigma. 0 disables. "
                    "Small-scale effect; needs high resolution (paper Sec 3.9)",
    )
    sparse: BoolProperty(
        name="Sparse Grid", default=False,
        description="Crop the solver to the active fluid region each step. "
                    "Large speed/memory win for localized free-surface flows; "
                    "disengages when outflows or cut-cell solids are present",
    )
    density: FloatProperty(
        name="Liquid Density", default=1000.0, min=1e-6,
        soft_max=5000.0,
        description="Liquid density used by the variable-density pressure "
                    "projection (mass per cubic Blender unit)",
    )
    local_cfl: FloatProperty(
        name="Local Advection CFL", default=1.0, min=0.05, max=4.0,
        description="Maximum particle travel in grid cells per local "
                    "advection substep; independent of Target CFL",
    )
    pcg_tolerance: FloatProperty(
        name="PCG Tolerance", default=1e-4, min=1e-8, max=1e-1,
        precision=6,
        description="Relative residual tolerance for the pressure solve",
    )
    pcg_max_iterations: IntProperty(
        name="PCG Max Iterations", default=400, min=1, max=10000,
        description="Maximum pressure-solver iterations per simulation step",
    )
    density_floor_relative: FloatProperty(
        name="Relative Density Floor", default=1e-3, min=1e-8, max=1.0,
        precision=6,
        description="Minimum face density as a fraction of Liquid Density; "
                    "prevents singular projection coefficients",
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
        description="Create either the fast Geometry Nodes preview or the "
                    "paper-style reconstructed surface",
    )
    surface_method: EnumProperty(
        name="Surface Method",
        items=[
            (
                "FAST_PREVIEW",
                "Fast Preview",
                "Deterministic Geometry Nodes points-to-volume preview; "
                "interactive, but not the paper's MCF reconstruction",
            ),
            (
                "PAPER_MCF",
                "Paper MCF",
                "Paper-style dense reconstruction and mean-curvature-flow "
                "iterations followed by CPU OpenVDB meshing",
            ),
        ],
        default="FAST_PREVIEW",
    )
    particle_radius: FloatProperty(
        name="Particle Radius", default=0.5, min=0.1, max=2.0,
        description="Surfacing sphere radius in cell widths",
    )
    surface_voxel: FloatProperty(
        name="Surface Voxel", default=0.5, min=0.1, max=2.0,
        description="Surfacing voxel size in cell widths",
    )
    surface_smoothing: BoolProperty(
        name="Geometric Smoothing", default=False,
        description="Apply Blender's Laplacian Smooth modifier to the "
                    "display mesh; this is post-process geometry smoothing, "
                    "not the paper's MCF reconstruction",
    )
    surface_smoothing_iterations: IntProperty(
        name="Smoothing Iterations", default=2, min=1, max=50,
        description="Iterations for Blender's Laplacian Smooth modifier",
    )
    surface_smoothing_factor: FloatProperty(
        name="Smoothing Factor", default=0.35, min=-2.0, max=2.0,
        description="Lambda factor for Blender's Laplacian Smooth modifier",
    )
    paper_mcf_iterations: IntProperty(
        name="MCF Iterations",
        default=30,
        min=1,
        max=100,
        description="Mean-curvature-flow reconstruction iterations; the "
                    "paper uses 30",
    )
    paper_mesh_adaptivity: FloatProperty(
        name="OpenVDB Mesh Adaptivity",
        default=0.0,
        min=0.0,
        max=1.0,
        subtype="FACTOR",
        description="CPU OpenVDB polygon reduction; 0 preserves the full "
                    "isosurface and matches the paper-facing default",
    )
    paper_max_reconstruction_voxels: IntProperty(
        name="Max Reconstruction Voxels",
        default=16_777_216,
        min=262_144,
        max=268_435_456,
        soft_max=134_217_728,
        description="Hard field-size cap for dense paper reconstruction; a "
                    "separate conservative preflight estimates temporary, "
                    "particle, OpenVDB, and live-solver memory",
    )
    bake_status: StringProperty(name="Bake Status", default="")
    bake_state: EnumProperty(
        name="Bake State",
        items=[
            ("IDLE", "Idle", "No bake is active"),
            ("RUNNING", "Running", "A bake is currently running"),
            ("COMPLETE", "Complete", "The requested frame range was baked"),
            ("CANCELLED", "Cancelled", "The bake was stopped by the user"),
            ("FAILED", "Failed", "The bake stopped because of an error"),
        ],
        default="IDLE",
    )
    bake_error: StringProperty(name="Bake Error", default="")
    bake_progress: FloatProperty(
        name="Bake Progress", default=0.0, min=0.0, max=1.0,
        subtype="FACTOR",
    )
    cache_id: StringProperty(
        name="Cache Owner ID", default="", options={"HIDDEN"},
        description="Persistent scene identifier used to prevent cache "
                    "ownership collisions",
    )
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
