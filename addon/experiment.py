"""Paper-inspired parameter profiles and diagnostics export operators."""

from pathlib import Path

import bpy
from bpy.props import EnumProperty, StringProperty
from bpy_extras.io_utils import ExportHelper

from ..stflip import cache, handoff
from ..stflip.experiments import get_profile
from . import handlers
from .handlers import resolve_cache_dir


def _handoff_cache_ready(scene) -> tuple[bool, str]:
    """Cheap UI/execute gate for an owned cache with committed outputs."""
    cache_dir = resolve_cache_dir(scene)
    metadata = cache.read_meta(cache_dir)
    ownership = handlers.scene_cache_ownership(scene, metadata)
    if ownership not in {cache.OWNERSHIP_OWNED, cache.OWNERSHIP_LEGACY}:
        return False, f"cache ownership is {ownership}"
    if not isinstance(metadata, dict):
        return False, "cache metadata is missing"
    start = metadata.get("frame_start")
    end = metadata.get("frame_end_baked")
    if (isinstance(start, bool) or isinstance(end, bool)
            or not isinstance(start, int) or not isinstance(end, int)
            or end < start):
        return False, "cache has no committed frame range"
    expected = list(range(start, end + 1))
    available = [
        frame for frame in cache.baked_frames(cache_dir)
        if start <= frame <= end
    ]
    if available != expected:
        return False, "cache is missing one or more committed playback frames"
    return True, ""


class STFLIP_OT_apply_experiment_profile(bpy.types.Operator):
    """Apply a paper-inspired parameter snapshot to the current scene."""

    bl_idname = "stflip.apply_experiment_profile"
    bl_label = "Apply Parameter Profile"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        from .operators import _BAKE

        return not _BAKE.get("running")

    def execute(self, context):
        settings = context.scene.stflip
        if settings.experiment_profile == "CUSTOM":
            self.report({"WARNING"}, "Choose a parameter profile first")
            return {"CANCELLED"}
        try:
            profile = get_profile(settings.experiment_profile)
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        for name, value in profile.settings().items():
            setattr(settings, name, value)
        settings.bake_status = (
            f"Applied {profile.label} ({profile.paper_reference})")
        self.report({"INFO"}, settings.bake_status)
        return {"FINISHED"}


class STFLIP_OT_export_metrics(bpy.types.Operator, ExportHelper):
    """Export recorded frame metrics as a self-contained CSV or JSON file."""

    bl_idname = "stflip.export_metrics"
    bl_label = "Export Recorded Metrics"
    filename_ext = ".csv"
    check_extension = False
    filter_glob: StringProperty(
        default="*.csv;*.json", options={"HIDDEN"}, maxlen=255)
    export_format: EnumProperty(
        name="Format",
        items=(
            ("CSV", "CSV", "Flat rows with embedded run metadata"),
            ("JSON", "JSON", "Self-contained run and frame records"),
        ),
        default="CSV",
    )

    @classmethod
    def poll(cls, context):
        from .operators import _BAKE

        return not _BAKE.get("running")

    def execute(self, context):
        if getattr(bpy.app, "background", False) and not self.filepath:
            self.report({"ERROR"}, "Set an explicit export path")
            return {"CANCELLED"}
        cache_dir = resolve_cache_dir(context.scene)
        destination = Path(bpy.path.abspath(self.filepath))
        suffix = ".json" if self.export_format == "JSON" else ".csv"
        if destination.suffix.lower() != suffix:
            destination = destination.with_suffix(suffix)
        try:
            if self.export_format == "JSON":
                exported = cache.export_metrics(
                    cache_dir, destination, "json")
            else:
                exported = cache.export_metrics(
                    cache_dir, destination, "csv")
        except (OSError, ValueError) as exc:
            self.report({"ERROR"}, f"Metrics export failed: {exc}")
            return {"CANCELLED"}
        self.report({"INFO"}, f"Exported metrics to {exported}")
        return {"FINISHED"}

    def draw(self, context):
        self.layout.prop(self, "export_format")


class STFLIP_OT_export_handoff(bpy.types.Operator, ExportHelper):
    """Export committed particle playback for downstream enhancement tools."""

    bl_idname = "stflip.export_handoff"
    bl_label = "Export Playback Handoff"
    filename_ext = ".zip"
    check_extension = True
    filter_glob: StringProperty(
        default="*.zip", options={"HIDDEN"}, maxlen=255)

    @classmethod
    def poll(cls, context):
        from .operators import _BAKE

        if _BAKE.get("running") or context is None:
            return False
        scene = getattr(context, "scene", None)
        return scene is not None and _handoff_cache_ready(scene)[0]

    def execute(self, context):
        if getattr(bpy.app, "background", False) and not self.filepath:
            self.report({"ERROR"}, "Set an explicit export path")
            return {"CANCELLED"}
        ready, reason = _handoff_cache_ready(context.scene)
        if not ready:
            self.report({"ERROR"}, f"Playback handoff unavailable: {reason}")
            return {"CANCELLED"}
        cache_dir = resolve_cache_dir(context.scene)
        destination = Path(bpy.path.abspath(self.filepath))
        if destination.suffix.lower() != ".zip":
            destination = destination.with_suffix(".zip")
        try:
            exported = handoff.export_handoff(cache_dir, destination)
        except (OSError, ValueError) as exc:
            self.report({"ERROR"}, f"Playback handoff export failed: {exc}")
            return {"CANCELLED"}
        self.report({"INFO"}, f"Exported playback handoff to {exported}")
        return {"FINISHED"}


CLASSES = (
    STFLIP_OT_apply_experiment_profile,
    STFLIP_OT_export_metrics,
    STFLIP_OT_export_handoff,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
