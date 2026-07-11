"""ST-FLIP sidebar UI (3D Viewport > N-panel > ST-FLIP)."""

import bpy

from .operators import current_cuda_diagnostics

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
            "stflip.whirlpool_preview", icon="FORCE_VORTEX", text="Whirlpool",
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
            layout.prop(obj.stflip, "inflow_velocity")
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
        layout.prop(st, "flip_blend")
        layout.prop(st, "seed")

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
        layout.enabled = st.bake_state != "RUNNING"
        layout.prop(st, "create_surface")
        col = layout.column(align=True)
        col.enabled = st.create_surface
        col.prop(st, "particle_radius")
        col.prop(st, "surface_voxel")
        col.prop(st, "surface_smoothing")
        smooth = col.column(align=True)
        smooth.enabled = st.surface_smoothing
        smooth.prop(st, "surface_smoothing_iterations")
        smooth.prop(st, "surface_smoothing_factor")
        col.label(text="Blender geometric smoothing; not paper MCF.",
                  icon="INFO")
        col.operator("stflip.refresh_surface", icon="FILE_REFRESH")


CLASSES = (
    STFLIP_PT_main,
    STFLIP_PT_object,
    STFLIP_PT_solver,
    STFLIP_PT_advanced,
    STFLIP_PT_experiment,
    STFLIP_PT_display,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
