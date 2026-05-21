"""Utility namespace for Chrono-Agent."""

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
    "setup_logger": ("chrono_agent.utils.logger", "setup_logger"),
    "get_logger": ("chrono_agent.utils.logger", "get_logger"),
    "AgentLogger": ("chrono_agent.utils.logger", "AgentLogger"),
    "setup_preview_camera": ("chrono_agent.utils.sensor_camera", "setup_preview_camera"),
    "CameraRecorder": ("chrono_agent.utils.sensor_camera", "CameraRecorder"),
    "get_registered_recorders": (
        "chrono_agent.utils.sensor_camera",
        "get_registered_recorders",
    ),
    "run_recording_loop": ("chrono_agent.utils.simulation_loop", "run_recording_loop"),
    "DEFAULT_PLACEMENT_MARGIN": (
        "chrono_agent.utils.scene_placement",
        "DEFAULT_PLACEMENT_MARGIN",
    ),
    "FootprintRegistry": ("chrono_agent.utils.scene_placement", "FootprintRegistry"),
    "SurfaceStack": ("chrono_agent.utils.scene_placement", "SurfaceStack"),
    "find_non_overlapping_pos": (
        "chrono_agent.utils.scene_placement",
        "find_non_overlapping_pos",
    ),
    "footprints_overlap": ("chrono_agent.utils.scene_placement", "footprints_overlap"),
    "has_any_overlap": ("chrono_agent.utils.scene_placement", "has_any_overlap"),
    "AssetDescriptor": ("chrono_agent.utils.scene_assets", "AssetDescriptor"),
    "add_visual_assets": ("chrono_agent.utils.scene_assets", "add_visual_assets"),
    "convex_decompose_asset": (
        "chrono_agent.utils.scene_assets",
        "convex_decompose_asset",
    ),
    "ensure_convex_json": ("chrono_agent.utils.scene_assets", "ensure_convex_json"),
    "add_collision_from_decomposition": (
        "chrono_agent.utils.scene_assets",
        "add_collision_from_decomposition",
    ),
    "add_collision_via_subbodies": (
        "chrono_agent.utils.scene_assets",
        "add_collision_via_subbodies",
    ),
    "create_asset_body": ("chrono_agent.utils.scene_assets", "create_asset_body"),
    "load_convex_hulls": ("chrono_agent.utils.scene_assets", "load_convex_hulls"),
    "transform_vertices": ("chrono_agent.utils.scene_assets", "transform_vertices"),
    "make_contact_material": ("chrono_agent.utils.scene_assets", "make_contact_material"),
    "box_inertia": ("chrono_agent.utils.scene_assets", "box_inertia"),
    "write_placement_csv": ("chrono_agent.utils.scene_assets", "write_placement_csv"),
    "write_contacts_csv": ("chrono_agent.utils.scene_assets", "write_contacts_csv"),
    "write_links_csv": ("chrono_agent.utils.scene_assets", "write_links_csv"),
    "build_fsi_tank": ("chrono_agent.utils.fsi_assets", "build_fsi_tank"),
    "build_fsi_floating_box": ("chrono_agent.utils.fsi_assets", "build_fsi_floating_box"),
    "write_fsi_alignment_csv": ("chrono_agent.utils.fsi_assets", "write_fsi_alignment_csv"),
    "setup_vsg_recording": ("chrono_agent.utils.vsg_recording", "setup_vsg_recording"),
    "lock_side_camera": ("chrono_agent.utils.vsg_recording", "lock_side_camera"),
    "hide_vsg_gui": ("chrono_agent.utils.vsg_recording", "hide_vsg_gui"),
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_IMPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
