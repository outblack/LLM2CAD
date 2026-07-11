# Primitive redesign probe

## Decision

- A right-angle turn must not be represented by two perpendicular straight tubes sharing only an endpoint. Their port-axis compatibility is zero (`anti_parallel_axis_dot = 0`), so the connection must be rejected before publication.
- Replace the failing sampled-arc/end-extension construction with one exact circular edge and an annular sweep, or with an exact torus segment. Both alternatives passed strict OCCT checks.
- Keep verified planar S-bends under the spline route family. Do not add native `Part.makeHelix` as a production primitive: its swept tube returned `BOPAlgo GeomAbs_C0` errors, and the probed non-planar splines did not clear the same strict gate. Compose currently safe 3D routes from analytic elbows in different planes.

## Verified results

| Case | Strict result | Evidence |
|---|---:|---|
| Exact one-edge circular-arc annular sweep | PASS | `analytic_prototypes.validation.json` |
| Exact quarter-torus annular elbow | PASS | `analytic_prototypes.validation.json` |
| Native `Part.makeHelix` annular sweep | FAIL (`GeomAbs_C0`) | `analytic_prototypes.validation.json` |
| v6 circular-arc route | PASS | `route_arc_v6.validation.json`, `route_arc_v6.png` |
| v6 watertight L: line + exact arc + line | PASS | `watertight_L_v6.validation.json`, `watertight_L_v6.png` |
| Initial aggressive S spline | FAIL (self-intersection) | `s_bend_spline_v6.validation.json`, `s_bend_spline_v6.png` |
| Revised gentle S spline | PASS | `s_bend_spline_v6b.validation.json`, `s_bend_spline_v6b.png` |
| Spatial route: two analytic elbows in different planes | PASS | `spatial_double_elbow_v6.validation.json`, `spatial_double_elbow_v6_white.png` |

The revised S-bend has a densely sampled minimum curvature radius of
`17.818466 mm`, exceeding its authored lower bound of `10.1 mm`.

The watertight L, revised S-bend, and spatial double-elbow route report one assembly solid, one outer-network solid, one bore-network solid, zero bore intrusion, no connection failures, no wall-section failures, and no non-adjacent overlaps.

## Artifact warning

`production_demos.validation.json`, `SmoothSBend.png`, `TorsionSpline.png`, and
`production_demos_overview.png` preserve an exploratory failure run. Only its
`watertight_l` record passed. These files must not be cited as successful S-bend
or torsion evidence. The `spatial_torsion_spline_v6` and `spatial_spline_v6b/v6c`
runs are also preserved failures; use `spatial_double_elbow_v6` as the current
production-safe 3D evidence.

## Recommended implementation policy

1. Build circular turns from one analytic arc edge; avoid endpoint bore cutters at the curved boundary when the co-terminal annular sweep is valid.
2. Require tangent continuity at internal path joins (`dot >= 0.999`).
3. Gate every spline with dense knot-span curvature sampling, minimum bend radius, strict BOP validation, wall samples, bore connectivity, and non-adjacent overlap checks.
4. Expose helix/coil only after a C2 representation passes those same gates; until then use multi-plane analytic elbows and do not bypass validation with a native helix edge.
