"""ST-FLIP sidebar UI (3D Viewport > N-panel > ST-FLIP)."""

import bpy

from .operators import current_cuda_diagnostics, surface_rebuild_running

# Checking CUDA means importing CuPy; doing that inside draw() would stall
# the UI (first import takes seconds) and re-run per redraw. Cache it.
_GPU_STATE = None


def gpu_state():
    global _GPU_STATE
    if _GPU_STATE is None:
        _GPU_STATE = current_cuda_diagnostics()
    return _GPU_STATE


def invalidate_gpu_state():
    global _GPU_STATE
    _GPU_STATE = None


class STFLIP_PT_main(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "ST-FLIP"
    bl_label = "ST-FLIP Fluid"

    def draw(self, context):
        layout = self.layout
        st = context.scene.stflip

        running = st.bake_state == "RUNNING"
        row = layout.row(align=True)
        row.enabled = not running
        row.operator("stflip.quick_setup", icon="MOD_FLUIDSIM")
        row.operator(
            "stflip.whirlpool_preview", icon="FORCE_VORTEX",
            text="Whirlpool Preview (Approx.)",
        )
        row = layout.row()
        row.enabled = not running
        row.operator(
            "stflip.high_cfl_jet_leak",
            icon="MOD_FLUIDSIM",
            text="High-CFL Jet Preview (Approx.)",
        )
        if context.scene.get("stflip_setup") in {
                "WHIRLPOOL_PREVIEW_APPROXIMATE",
                "HIGH_CFL_JET_LEAK_APPROXIMATE"}:
            layout.label(
                text="Domain/resolution/FPS edits break preset ratios.",
                icon="INFO",
            )
        domain_row = layout.row()
        domain_row.enabled = not running
        domain_row.prop(st, "domain")

        col = layout.column(align=True)
        col.enabled = not running
        col.prop(st, "resolution")
        col.prop(st, "cfl_target")
        col.prop(st, "particles_per_cell")

        row = layout.row(align=True)
        if st.bake_state == "RUNNING":
            row.operator("stflip.cancel_bake", icon="CANCEL", text="Cancel")
        else:
            row.operator("stflip.bake", icon="PLAY")
            row.operator(
                "stflip.resume_bake", icon="RECOVER_LAST", text="Resume")
        row.operator("stflip.free_bake", icon="TRASH", text="")
        if st.bake_state == "RUNNING":
            layout.prop(st, "bake_progress", text="", slider=True)
        if st.bake_status:
            layout.label(text=st.bake_status)
        if st.bake_state == "FAILED" and st.bake_error:
            layout.label(text=st.bake_error, icon="ERROR")


class STFLIP_PT_object(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "ST-FLIP"
    bl_label = "Active Object"
    bl_parent_id = "STFLIP_PT_main"

    def draw(self, context):
        layout = self.layout
        obj = context.active_object
        if obj is None or obj.type != "MESH":
            layout.label(text="Select a mesh object")
            return
        layout.enabled = context.scene.stflip.bake_state != "RUNNING"
        layout.prop(obj.stflip, "role", text="Role")
        if obj.stflip.role == "LIQUID":
            settings = obj.stflip
            layout.prop(settings, "initial_velocity_mode", text="Velocity")
            layout.prop(
                settings,
                "initial_velocity",
                text=("Uniform Velocity"
                      if settings.initial_velocity_mode == "UNIFORM"
                      else "Linear Velocity"),
            )
            if settings.initial_velocity_mode == "SOLID_BODY":
                col = layout.column(align=True)
                col.prop(settings, "rotation_center_world")
                col.prop(settings, "rotation_axis_world")
                col.prop(settings, "angular_speed")
        elif obj.stflip.role == "INFLOW":
            settings = obj.stflip
            layout.prop(settings, "inflow_velocity_mode", text="Velocity")
            layout.prop(
                settings,
                "inflow_velocity",
                text=("Uniform Velocity"
                      if settings.inflow_velocity_mode == "UNIFORM"
                      else "Linear Velocity"),
            )
            if settings.inflow_velocity_mode == "SOLID_BODY":
                col = layout.column(align=True)
                col.prop(settings, "rotation_center_world")
                col.prop(settings, "rotation_axis_world")
                col.prop(settings, "angular_speed")
            if context.scene.stflip.two_phase:
                layout.prop(settings, "inflow_is_gas")
            layout.separator()
            layout.prop(settings, "inflow_use_frame_range")
            frames = layout.row(align=True)
            frames.enabled = settings.inflow_use_frame_range
            frames.prop(settings, "inflow_start_frame")
            frames.prop(settings, "inflow_end_frame")
            layout.label(text="Inclusive evolved output frames.",
                         icon="INFO")
        elif obj.stflip.role == "OBSTACLE":
            layout.prop(obj.stflip, "obstacle_animated")
        elif obj.stflip.role == "OUTFLOW":
            layout.prop(obj.stflip, "outflow_mode")
            if obj.stflip.outflow_mode == "VOLUME":
                layout.label(text="Deletes particles inside the volume.",
                             icon="INFO")
                layout.label(text="Not a pressure boundary.")
            else:
                layout.label(text="Atmospheric-pressure exterior opening.",
                             icon="INFO")
                layout.label(text="Must intersect a domain boundary.")


class STFLIP_PT_solver(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "ST-FLIP"
    bl_label = "Solver"
    bl_parent_id = "STFLIP_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        st = context.scene.stflip
        layout.enabled = st.bake_state != "RUNNING"

        layout.prop(st, "st_enabled")
        sub = layout.column(align=True)
        sub.enabled = st.st_enabled
        sub.prop(st, "jitter_strength")
        sub.prop(st, "adaptive_gamma")
        sub.prop(st, "eta_phi")

        layout.separator()
        layout.prop(st, "transfer")
        row = layout.row()
        row.enabled = st.transfer == "flip"
        row.prop(st, "flip_blend")
        layout.prop(st, "seed")

        layout.separator()
        layout.prop(st, "two_phase")
        gas = layout.column(align=True)
        gas.enabled = st.two_phase
        gas.prop(st, "rho_gas")
        gas.prop(st, "gas_particles_per_cell")
        layout.prop(st, "surface_tension")
        layout.prop(st, "viscosity")
        layout.prop(st, "sparse")

        layout.prop(st, "whitewater")
        ww = layout.column(align=True)
        ww.enabled = st.whitewater
        ww.prop(st, "whitewater_rate")
        ww.prop(st, "whitewater_max")

        layout.separator()
        layout.prop(st, "backend")
        state = gpu_state()
        if state["available"]:
            layout.label(text=f"GPU: {state['device']}", icon="CHECKMARK")
            if state["free_bytes"] and state["total_bytes"]:
                gib = 1024 ** 3
                layout.label(
                    text=(f"VRAM: {state['free_bytes'] / gib:.1f} / "
                          f"{state['total_bytes'] / gib:.1f} GiB free"),
                )
        else:
            layout.label(text="CUDA compute unavailable", icon="INFO")
            if state["error"]:
                detail = " ".join(state["error"].split())
                if len(detail) > 90:
                    detail = detail[:87] + "..."
                layout.label(text=detail, icon="ERROR")
            layout.operator("stflip.install_gpu", icon="IMPORT")
        layout.prop(st, "cache_dir")


class STFLIP_PT_advanced(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "ST-FLIP"
    bl_label = "Advanced Solver"
    bl_parent_id = "STFLIP_PT_solver"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        st = context.scene.stflip
        layout.enabled = st.bake_state != "RUNNING"
        col = layout.column(align=True)
        col.prop(st, "density")
        col.prop(st, "density_floor_relative")
        col.prop(st, "local_cfl")
        col.prop(st, "pcg_tolerance")
        col.prop(st, "pcg_max_iterations")


class STFLIP_PT_experiment(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "ST-FLIP"
    bl_label = "Experiment Diagnostics"
    bl_parent_id = "STFLIP_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        st = context.scene.stflip
        layout.enabled = st.bake_state != "RUNNING"

        layout.prop(st, "experiment_profile")
        row = layout.row()
        row.enabled = st.experiment_profile != "CUSTOM"
        row.operator("stflip.apply_experiment_profile", icon="PRESET")
        layout.label(text="Profiles set parameters, not scene geometry.",
                     icon="INFO")

        layout.separator()
        layout.prop(st, "collect_metrics")
        sub = layout.column()
        sub.enabled = st.collect_metrics
        sub.prop(st, "collect_enstrophy")
        layout.operator("stflip.export_metrics", icon="EXPORT")


class STFLIP_PT_export(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "ST-FLIP"
    bl_label = "Downstream Export"
    bl_parent_id = "STFLIP_PT_main"

    def draw(self, context):
        layout = self.layout
        st = context.scene.stflip
        layout.label(text="ZIP of committed playback frames.", icon="EXPORT")
        layout.operator(
            "stflip.export_handoff",
            icon="EXPORT",
            text="Export Playback Handoff",
        )
        layout.label(text="Positions + velocity; no AI model.", icon="INFO")

        layout.separator()
        layout.label(text="Animated mesh cache for render farms/DCCs.",
                     icon="MESH_DATA")
        layout.operator("stflip.export_cache", icon="EXPORT",
                        text="Export Alembic/USD")
        layout.operator("stflip.setup_motion_blur", icon="ONIONSKIN_ON",
                        text="Set Up Motion Blur")
        if st.bake_state == "IDLE":
            layout.label(text="Bake first to enable export.", icon="INFO")


class STFLIP_PT_display(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "ST-FLIP"
    bl_label = "Surface"
    bl_parent_id = "STFLIP_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        st = context.scene.stflip
        simulation_running = st.bake_state == "RUNNING"
        rebuilding = surface_rebuild_running()
        layout.enabled = not simulation_running
        controls = layout.column(align=True)
        controls.enabled = not rebuilding
        controls.prop(st, "create_surface")
        col = controls.column(align=True)
        col.enabled = st.create_surface
        col.prop(st, "surface_method")
        if st.surface_method == "FAST_PREVIEW":
            col.prop(st, "particle_radius")
            col.prop(st, "surface_voxel")
            col.prop(st, "surface_smoothing")
            smooth = col.column(align=True)
            smooth.enabled = st.surface_smoothing
            smooth.prop(st, "surface_smoothing_iterations")
            smooth.prop(st, "surface_smoothing_factor")
            col.label(text="Deterministic Geometry Nodes preview.",
                      icon="INFO")
            col.label(text="Laplacian smoothing is not paper MCF.")
        else:
            paper = col.column(align=True)
            paper.prop(st, "paper_mcf_iterations")
            paper.prop(st, "paper_mesh_adaptivity")
            paper.prop(st, "paper_max_reconstruction_voxels")
            paper.separator()
            paper.label(text="Paper constants: radius 0.5Δx, voxel 0.5Δx",
                        icon="INFO")
            paper.label(text="Gaussian σ = 2Δx")
            paper.label(text="Feature mask: θ = 2, ζ = 5")
            paper.separator()
            paper.label(text="Dense reconstruction uses NumPy or CuPy.",
                        icon="INFO")
            paper.label(text="Dense fields consume host RAM or CUDA VRAM.")
            paper.label(text="OpenVDB polygonization uses CPU/RAM only.")
            paper.separator()
            paper.operator(
                "stflip.rebuild_paper_surfaces",
                icon="MOD_FLUIDSIM",
                text="Rebuild Paper Surface Cache",
            )
        col.operator("stflip.refresh_surface", icon="FILE_REFRESH")
        if rebuilding:
            layout.operator(
                "stflip.cancel_surface_rebuild",
                icon="CANCEL",
                text="Cancel Surface Rebuild",
            )


CLASSES = (
    STFLIP_PT_main,
    STFLIP_PT_object,
    STFLIP_PT_solver,
    STFLIP_PT_advanced,
    STFLIP_PT_experiment,
    STFLIP_PT_export,
    STFLIP_PT_display,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
