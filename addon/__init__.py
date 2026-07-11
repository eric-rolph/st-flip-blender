"""Blender-facing layer of the ST-FLIP addon."""

from . import experiment, handlers, operators, panels, properties

_MODULES = (properties, operators, experiment, panels, handlers)


def register():
    for mod in _MODULES:
        mod.register()


def unregister():
    for mod in reversed(_MODULES):
        mod.unregister()
