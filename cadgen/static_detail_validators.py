"""Detailed static geometry validators extracted for readability.

Behavior-preserving split from static_geometry_validator. Public API stays there.
"""

from __future__ import annotations

from collections import Counter
import math
from typing import Any

from cadgen.typed_data_models import (
    CriticReport,
    CriticViewRequest,
    Direction,
    IntentResult,
    IssueSeverity,
    PatchSuggestion,
    PipeState,
    Port,
    ResolvedAction,
    StateTransition,
    StaticIssue,
    StepVerification,
)
from cadgen.vector3_math import (
    add,
    canonical_circular_arc_frame,
    circular_rim_mismatch,
    cross,
    direction_to_vector,
    dot,
    length,
    mul,
    normalize,
    rotate,
    sub,
    vec,
)
from cadgen.static_geometry_metrics import (
    _analytic_route_arc_tangents,
    _arc_endpoint_tangents,
    _circumradius,
    _collision_envelope_reliable,
    _connection_contract_invalid,
    _connection_interface_metrics,
    _direction_score,
    _find_connectable_port,
    _find_port,
    _goal_path_points,
    _include_primary_outlet,
    _is_start_anchor_bootstrap_transition,
    _match_vectors_to_ports,
    _module_centerline_length,
    _module_centerline_points,
    _module_collision_segments,
    _module_envelope_radius,
    _module_primary_displacement,
    _module_spatial_samples,
    _module_turn_angle,
    _near,
    _normalize_vector_list,
    _point_segment_distance,
    _point_to_circular_arc_projection,
    _point_to_goal_path_projection,
    _point_to_polyline_distance,
    _point_to_polyline_projection,
    _port_role,
    _required_outlet_vectors,
    _same_direction,
    _segment_distance,
    _segment_has_endpoint,
    _three_point_arc_length,
    _vectors_json,
)

# Shared thresholds (kept identical to static_geometry_validator).
VECTOR_TOLERANCE = 1e-4
PARALLEL_DOT_THRESHOLD = 0.9999
BRANCH_DIRECTION_DOT_THRESHOLD = 0.35
EXPLICIT_VECTOR_DOT_THRESHOLD = 0.9999

from cadgen.static_issue_builder import append_issue as _append_issue
from cadgen.static_final_validators import (
    _validate_conservative_collision,
    _validate_final_goal_lengths,
    _validate_final_graph,
    _validate_final_spline_curvature,
    _validate_geometric_constraints,
    _validate_intra_module_clearance,
)
from cadgen.static_transition_validators import (
    _validate_graph_transition,
    _validate_module_connection,
    _validate_route_continuity,
)
from cadgen.static_goal_validators import (
    _validate_branch_goal_direction,
    _validate_goal_completion,
    _validate_move_goal_direction,
    _validate_turn_goal_direction,
)
