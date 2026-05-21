"""Utility namespace for Chrono-Code."""

from importlib import import_module

__all__ = [
    "setup_logger",
    "get_logger",
    "AgentLogger",
    "setup_preview_camera",
    "CameraRecorder",
    "get_registered_recorders",
    "run_recording_loop",
    "DEFAULT_PLACEMENT_MARGIN",
    "FootprintRegistry",
    "SurfaceStack",
    "find_non_overlapping_pos",
    "footprints_overlap",
    "has_any_overlap",
    "AssetDescriptor",
    "add_visual_assets",
    "convex_decompose_asset",
    "ensure_convex_json",
    "add_collision_from_decomposition",
    "add_collision_via_subbodies",
    "create_asset_body",
    "load_convex_hulls",
    "transform_vertices",
    "make_contact_material",
    "box_inertia",
    "write_placement_csv",
    "write_contacts_csv",
    "write_links_csv",
    "build_fsi_tank",
    "build_fsi_floating_box",
    "write_fsi_alignment_csv",
    "setup_vsg_recording",
    "lock_side_camera",
    "hide_vsg_gui",
]

_LAZY_IMPORTS = {
    "setup_logger": ("chrono_code.utils.logger", "setup_logger"),
    "get_logger": ("chrono_code.utils.logger", "get_logger"),
    "AgentLogger": ("chrono_code.utils.logger", "AgentLogger"),
    "setup_preview_camera": ("chrono_code.utils.sensor_camera", "setup_preview_camera"),
    "CameraRecorder": ("chrono_code.utils.sensor_camera", "CameraRecorder"),
    "get_registered_recorders": (
        "chrono_code.utils.sensor_camera",
        "get_registered_recorders",
    ),
    "run_recording_loop": ("chrono_code.utils.simulation_loop", "run_recording_loop"),
    "DEFAULT_PLACEMENT_MARGIN": (
        "chrono_code.utils.scene_placement",
        "DEFAULT_PLACEMENT_MARGIN",
    ),
    "FootprintRegistry": ("chrono_code.utils.scene_placement", "FootprintRegistry"),
    "SurfaceStack": ("chrono_code.utils.scene_placement", "SurfaceStack"),
    "find_non_overlapping_pos": (
        "chrono_code.utils.scene_placement",
        "find_non_overlapping_pos",
    ),
    "footprints_overlap": ("chrono_code.utils.scene_placement", "footprints_overlap"),
    "has_any_overlap": ("chrono_code.utils.scene_placement", "has_any_overlap"),
    "AssetDescriptor": ("chrono_code.utils.scene_assets", "AssetDescriptor"),
    "add_visual_assets": ("chrono_code.utils.scene_assets", "add_visual_assets"),
    "convex_decompose_asset": (
        "chrono_code.utils.scene_assets",
        "convex_decompose_asset",
    ),
    "ensure_convex_json": ("chrono_code.utils.scene_assets", "ensure_convex_json"),
    "add_collision_from_decomposition": (
        "chrono_code.utils.scene_assets",
        "add_collision_from_decomposition",
    ),
    "add_collision_via_subbodies": (
        "chrono_code.utils.scene_assets",
        "add_collision_via_subbodies",
    ),
    "create_asset_body": ("chrono_code.utils.scene_assets", "create_asset_body"),
    "load_convex_hulls": ("chrono_code.utils.scene_assets", "load_convex_hulls"),
    "transform_vertices": ("chrono_code.utils.scene_assets", "transform_vertices"),
    "make_contact_material": ("chrono_code.utils.scene_assets", "make_contact_material"),
    "box_inertia": ("chrono_code.utils.scene_assets", "box_inertia"),
    "write_placement_csv": ("chrono_code.utils.scene_assets", "write_placement_csv"),
    "write_contacts_csv": ("chrono_code.utils.scene_assets", "write_contacts_csv"),
    "write_links_csv": ("chrono_code.utils.scene_assets", "write_links_csv"),
    "build_fsi_tank": ("chrono_code.utils.fsi_assets", "build_fsi_tank"),
    "build_fsi_floating_box": ("chrono_code.utils.fsi_assets", "build_fsi_floating_box"),
    "write_fsi_alignment_csv": ("chrono_code.utils.fsi_assets", "write_fsi_alignment_csv"),
    "setup_vsg_recording": ("chrono_code.utils.vsg_recording", "setup_vsg_recording"),
    "lock_side_camera": ("chrono_code.utils.vsg_recording", "lock_side_camera"),
    "hide_vsg_gui": ("chrono_code.utils.vsg_recording", "hide_vsg_gui"),
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_IMPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
