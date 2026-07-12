# PAPR verify report

- timestamp: `20260712T051406376067Z`
- workspace: `/Users/jhkim/Documents/cadgen04`
- output: `/Users/jhkim/Documents/cadgen04/outputs/papr_verify_20260712T051406376067Z`
- overall: **PARTIAL_PASS**
- host realization: **PASS**
- freecad_script build: **PASS**
- FreeCAD MCP: **FAIL** (reachable; B-Rep validation rejected)

## Spec (host compiler, no step planner LLM)

1. 70mm line along +X, OD 20, wall 2
2. binary Y with primary +X continuation and branch along normalize(+1,-1,+1) length 85
3. 40mm line on primary along primary port axis

## Host compiler realization — PASS

| Step | Goal | Compiled module | Target port | Open ports after |
|------|------|-----------------|-------------|------------------|
| 1 | G1 route line 70 | route (M1) | START | M1.out |
| 2 | G2 branch Y | junction (M2) | M1.out | M2.out (primary), M2.out_1 (branch) |
| 3 | G3 route line 40 | route (M3) | M2.out (primary) | M2.out_1, M3.out |

Checks:
- all goals realized (`remaining_goals == []`)
- module sequence: `route → junction → route`
- open_ports == 2 after full realization
- exactly one primary port: `M3.out` axis `(1,0,0)`
- branch open port `M2.out_1` axis ≈ normalize(+1,-1,+1) = `(0.57735, -0.57735, 0.57735)`
- junction outlet lengths both 85mm; primary continues +X to trunk_end `(155,0,0)`; M3 extends to `(195,0,0)`

Artifacts:
- `actions_summary.json` — intent, per-step drafts/resolved actions, final state summary
- `state.json` — full host `PipeState`
- `freecad_script.py` — generated candidate builder (108135 chars)
- geometry_payload_digest: `82d34f28afa97a111f957e604060f220315cf89a0c02f0839ce58e945361448d`

## FreeCAD MCP — FAIL (not unavailable)

Settings from project `.env`:
- `freecad_mcp_enabled=True`
- command: `uvx --from freecad-mcp==0.1.19 freecad-mcp --only-text-feedback`
- probe: **PASS** (`{"passed": true, "protocol": 1}`)
- execute freecad_script: **ran** (returned CADGEN_VALIDATION sentinel)
- assess_freecad_validation: **rejected**

### What broke

Junction module **M2** solid construction failed inside FreeCAD:

```json
[
  {
    "details": {
      "module_type": "junction"
    },
    "error": "Bnd_Box is void",
    "failure_code": "MODULE_SHAPE_CONSTRUCTION_FAILED",
    "module_id": "M2",
    "stage": "make_module_shape"
  }
]
```

Per-module outcomes:
- **M1 route**: passed (volume≈7916.8, bounds x 0..70)
- **M2 junction**: failed — `Bnd_Box is void` / `MODULE_SHAPE_CONSTRUCTION_FAILED` at `make_module_shape`
- **M3 route**: passed (volume≈4523.9, bounds x 155..195)
- **assembly**: not built (`is_null`, bop_errors: "assembly not built")

This is **not** a host-compile bug: PAPR `compile_next_action` + `StateEngine` fully realized the multi-port graph with correct axes/ports. The failure is FreeCAD-side junction B-Rep construction (`make_junction` / fillet smooth_hub path producing a void bounding box).

No code fix applied: host realization was not blocked; FreeCAD junction solid failure is outside the "clear host-compile bug" fix scope for this verification.

### Screenshots / views

Not captured — candidate assembly never passed validation (no valid PipeAssembly digest binding for `capture_freecad_views`).

## Artifacts list

- `outputs/papr_verify_20260712T051406376067Z/actions_summary.json`
- `outputs/papr_verify_20260712T051406376067Z/state.json`
- `outputs/papr_verify_20260712T051406376067Z/freecad_script.py`
- `outputs/papr_verify_20260712T051406376067Z/mcp_raw_result.json`
- `outputs/papr_verify_20260712T051406376067Z/freecad_validation.json`
- `outputs/papr_verify_20260712T051406376067Z/mcp_status.json` (if present)
- `outputs/papr_verify_20260712T051406376067Z/report.md`

## Verdict

| Gate | Result |
|------|--------|
| Host multi-step realization without step planner LLM | **PASS** |
| FreeCAD script generation | **PASS** |
| FreeCAD MCP probe | **PASS** |
| FreeCAD B-Rep validation of realized Y-pipe | **FAIL** (M2 junction void solid) |
| View capture | **SKIPPED** (validation fail) |
| **Overall** | **PARTIAL_PASS** |
