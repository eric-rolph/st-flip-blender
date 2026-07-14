"""ST-FLIP Fluid for Blender.

Large-time-step FLIP liquid simulation based on:
  Braun, Winchenbach, Bender, Thuerey.
  "Spatiotemporal FLIP for Fast Free-Surface and Two-Phase Simulation With
  Very Large Time Steps". ACM Transactions on Graphics 45(4), 2026.
  https://doi.org/10.1145/3811289
"""

# Legacy addon metadata; Blender 4.2+ extensions read blender_manifest.toml.
bl_info = {
    "name": "ST-FLIP Fluid",
    "author": "Eric Rolph",
    "version": (0, 30, 0),
    "blender": (4, 2, 0),
    "location": "3D Viewport > Sidebar > ST-FLIP",
    "description": "Large-time-step FLIP liquid simulation (ST-FLIP, "
                   "SIGGRAPH 2026) with optional CUDA GPU acceleration",
    "category": "Physics",
}


def register():
    from .addon import register as _register
    _register()


def unregister():
    from .addon import unregister as _unregister
    _unregister()
