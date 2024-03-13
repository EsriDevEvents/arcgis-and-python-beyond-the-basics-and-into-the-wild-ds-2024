"""
All the spatial analysis tools from the analyze_patterns, enrich_data, find_locations, manage_data, summarize_data and
use_proximity submodules, in one place for convenience.
"""

from .analyze_patterns import (
    calculate_density,
    find_hot_spots,
    find_outliers,
    find_point_clusters,
    interpolate_points,
    summarize_center_and_dispersion,
)
from .use_proximity import (
    connect_origins_to_destinations,
    create_buffers,
    create_drive_time_areas,
    find_nearest,
    plan_routes,
)
from .enrich_data import enrich_layer
from .find_locations import (
    choose_best_facilities,
    create_viewshed,
    create_watersheds,
    derive_new_locations,
    find_centroids,
    find_existing_locations,
    find_similar_locations,
    trace_downstream,
)
from .manage_data import (
    create_route_layers,
    dissolve_boundaries,
    extract_data,
    generate_tessellation,
    merge_layers,
    overlay_layers,
)
from .summarize_data import (
    aggregate_points,
    join_features,
    summarize_center_and_dispersion,
    summarize_nearby,
    summarize_within,
)
