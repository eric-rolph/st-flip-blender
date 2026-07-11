"""Paper-inspired parameter profiles and diagnostics export operators."""

from pathlib import Path

import bpy
from bpy.props import EnumProperty, StringProperty
from bpy_extras.io_utils import ExportHelper

from ..stflip import cache
from ..stflip.experiments import get_profile
from .handlers import resolve_cache_dir


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


CLASSES = (
    STFLIP_OT_apply_experiment_profile,
    STFLIP_OT_export_metrics,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
