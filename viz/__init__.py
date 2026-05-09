from viz.recorder import PipelineRecorder, STEP_ORDER
from pipeline.features import FrameFeatures, compute_frame_features_cache
from viz.step_runner import (
    ensure_all_step_pngs_exist,
    ensure_sequence_outputs_exist,
    export_all_stages,
    export_sequence_consecutive_pairs,
    export_single_pair_stages,
    iter_sequence_pairs,
)

# Re-export fusion helpers used with sequence export
from pipeline.fusion import FusedLandmarkMap, FusedLandmark, fused_world_points_homogeneous
from viz.overlays import (
    blend_photo_depth_colormap,
    draw_epilines,
    draw_matches,
    draw_keypoints,
    draw_inlier_outlier_matches,
    depth_colormap_range_m,
    depth_to_bgr_red_near_blue_far,
    estimated_depth_visualization,
    project_points_topdown,
    project_world_points_to_camera_uv_z,
    render_dense_depth_colormap,
    render_depth_histogram_panel,
    render_sparse_depth_pixels,
    render_trajectory_topdown,
    sparse_depth_error_heatmap,
)

__all__ = [
    "FrameFeatures",
    "compute_frame_features_cache",
    "PipelineRecorder",
    "STEP_ORDER",
    "blend_photo_depth_colormap",
    "depth_colormap_range_m",
    "depth_to_bgr_red_near_blue_far",
    "draw_epilines",
    "draw_matches",
    "draw_keypoints",
    "draw_inlier_outlier_matches",
    "estimated_depth_visualization",
    "project_points_topdown",
    "project_world_points_to_camera_uv_z",
    "render_dense_depth_colormap",
    "render_depth_histogram_panel",
    "render_sparse_depth_pixels",
    "render_trajectory_topdown",
    "sparse_depth_error_heatmap",
    "export_all_stages",
    "export_sequence_consecutive_pairs",
    "export_single_pair_stages",
    "ensure_all_step_pngs_exist",
    "ensure_sequence_outputs_exist",
    "iter_sequence_pairs",
    "FusedLandmarkMap",
    "FusedLandmark",
    "fused_world_points_homogeneous",
]
