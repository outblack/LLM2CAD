import json
import math
import FreeCAD as App
import FreeCADGui as Gui
import Part

DOC_NAME = "PrimitiveRedesignProductionDemos"
OUT_DIR = "/Users/jhkim/Documents/cadgen02/artifacts/primitive_redesign_probe"
TOL = 1e-6


def normalized(value):
    result = App.Vector(value.x, value.y, value.z)
    if result.Length <= 1e-12:
        raise ValueError("zero vector")
    result.normalize()
    return result


def perpendicular(axis):
    axis = normalized(axis)
    for candidate in (
        App.Vector(0, 0, 1),
        App.Vector(0, 1, 0),
        App.Vector(1, 0, 0),
    ):
        radial = axis.cross(candidate)
        if radial.Length > 1e-9:
            return normalized(radial)
    raise ValueError("no perpendicular")


def circle_wire(radius, center, normal):
    return Part.Wire([Part.makeCircle(radius, center, normalized(normal))])


def shape_metrics(shape):
    errors = []
    try:
        errors = [str(item) for item in (shape.check(True) or [])]
    except Exception as exc:
        errors = ["check_exception: " + str(exc)]
    box = shape.BoundBox
    result = {
        "is_null": bool(shape.isNull()),
        "is_valid": bool(shape.isValid()),
        "is_closed": bool(shape.isClosed()),
        "solid_count": len(shape.Solids),
        "volume": float(shape.Volume),
        "bounds": {
            "minimum": [float(box.XMin), float(box.YMin), float(box.ZMin)],
            "maximum": [float(box.XMax), float(box.YMax), float(box.ZMax)],
        },
        "bop_errors": errors,
    }
    result["passed"] = bool(
        not result["is_null"]
        and result["is_valid"]
        and result["is_closed"]
        and result["solid_count"] == 1
        and result["volume"] > TOL ** 3
        and not errors
    )
    return result


def swept_tube(wire, outer_radius, bore_radius, frenet=False):
    first_edge = wire.Edges[0]
    start = first_edge.valueAt(first_edge.FirstParameter)
    tangent = normalized(first_edge.tangentAt(first_edge.FirstParameter))
    outer = wire.makePipeShell(
        [circle_wire(outer_radius, start, tangent)], True, bool(frenet)
    )
    bore = wire.makePipeShell(
        [circle_wire(bore_radius, start, tangent)], True, bool(frenet)
    )
    assembly = outer.cut(bore).removeSplitter()
    return assembly, outer, bore


def wall_samples(assembly, wire, outer_radius, bore_radius):
    failures = []
    count = 0
    wall_radius = (outer_radius + bore_radius) / 2.0
    outside_radius = outer_radius + max(0.25, (outer_radius - bore_radius) * 0.25)
    for edge_index, edge in enumerate(wire.Edges):
        first = float(edge.FirstParameter)
        last = float(edge.LastParameter)
        for fraction in (0.12, 0.35, 0.62, 0.88):
            parameter = first + (last - first) * fraction
            point = edge.valueAt(parameter)
            axis = normalized(edge.tangentAt(parameter))
            radial_a = perpendicular(axis)
            radial_b = normalized(axis.cross(radial_a))
            for radial_index in range(8):
                angle = 2.0 * math.pi * radial_index / 8.0
                radial = radial_a * math.cos(angle) + radial_b * math.sin(angle)
                count += 1
                if not assembly.isInside(point + radial * wall_radius, TOL, True):
                    failures.append(
                        {
                            "edge": edge_index,
                            "fraction": fraction,
                            "radial_sample": radial_index,
                            "kind": "missing_wall_material",
                        }
                    )
                if assembly.isInside(point + radial * outside_radius, TOL, True):
                    failures.append(
                        {
                            "edge": edge_index,
                            "fraction": fraction,
                            "radial_sample": radial_index,
                            "kind": "material_outside_outer_radius",
                        }
                    )
            if assembly.isInside(point, TOL, True):
                failures.append(
                    {
                        "edge": edge_index,
                        "fraction": fraction,
                        "kind": "blocked_bore",
                    }
                )
    return {"passed": not failures, "sample_count": count, "failures": failures}


def tangent_dot(left_edge, left_at_end, right_edge, right_at_end):
    left_parameter = left_edge.LastParameter if left_at_end else left_edge.FirstParameter
    right_parameter = right_edge.LastParameter if right_at_end else right_edge.FirstParameter
    left = normalized(left_edge.tangentAt(left_parameter))
    right = normalized(right_edge.tangentAt(right_parameter))
    return float(left.dot(right))


def demo_result(name, wire, assembly, outer, bore, outer_radius, bore_radius, connection):
    assembly_check = shape_metrics(assembly)
    outer_check = shape_metrics(outer)
    bore_check = shape_metrics(bore)
    intrusion = float(assembly.common(bore).Volume)
    intrusion_allowance = max(TOL ** 3, float(bore.Volume) * 1e-8)
    bore_check["assembly_intrusion_volume"] = intrusion
    bore_check["assembly_intrusion_allowance"] = intrusion_allowance
    bore_check["passed"] = bool(bore_check["passed"] and intrusion <= intrusion_allowance)
    expected_outer_volume = float(wire.Length) * math.pi * outer_radius ** 2
    outer_volume_error_ratio = abs(float(outer.Volume) - expected_outer_volume) / max(
        expected_outer_volume, TOL
    )
    overlap = {
        "passed": outer_volume_error_ratio <= 1e-6,
        "non_adjacent_overlaps": [],
        "sweep_volume_relative_error": outer_volume_error_ratio,
    }
    wall = wall_samples(assembly, wire, outer_radius, bore_radius)
    checks = {
        "assembly": assembly_check,
        "outer_network": outer_check,
        "bore_network": bore_check,
        "connection": connection,
        "wall": wall,
        "overlap": overlap,
    }
    return {
        "name": name,
        "path_edge_count": len(wire.Edges),
        "centerline_length": float(wire.Length),
        "outer_radius": outer_radius,
        "bore_radius": bore_radius,
        "wall_thickness": outer_radius - bore_radius,
        "checks": checks,
        "passed": all(item.get("passed", False) for item in checks.values()),
    }


if DOC_NAME in App.listDocuments():
    App.closeDocument(DOC_NAME)
doc = App.newDocument(DOC_NAME)
results = {}
objects = []

# 1. Watertight L: line + exact circular arc + line as one C1 path sweep.
p0 = App.Vector(0, 0, 0)
p1 = App.Vector(50, 0, 0)
pm = App.Vector(50 + 30 / math.sqrt(2), 30 * (1 - 1 / math.sqrt(2)), 0)
p2 = App.Vector(80, 30, 0)
p3 = App.Vector(80, 80, 0)
l_edges = [Part.makeLine(p0, p1), Part.Arc(p1, pm, p2).toShape(), Part.makeLine(p2, p3)]
l_wire = Part.Wire(l_edges)
l_assembly, l_outer, l_bore = swept_tube(l_wire, 10.0, 8.0)
l_connection = {
    "passed": True,
    "join_tangent_dots": [
        tangent_dot(l_edges[0], True, l_edges[1], False),
        tangent_dot(l_edges[1], True, l_edges[2], False),
    ],
    "connection_failures": [],
}
l_connection["passed"] = all(value >= 0.999 for value in l_connection["join_tangent_dots"])
obj = doc.addObject("Part::Feature", "WatertightL")
obj.Label = "Watertight L: line + exact arc + line"
obj.Shape = l_assembly
obj.ViewObject.ShapeColor = (0.20, 0.70, 0.95)
objects.append(obj)
results["watertight_l"] = demo_result(
    "watertight_l", l_wire, l_assembly, l_outer, l_bore, 10.0, 8.0, l_connection
)

# 2. S-bend: one endpoint-tangent C2 B-spline, no hard junction seams.
s_points = [
    App.Vector(0, 0, 0),
    App.Vector(35, 18, 0),
    App.Vector(105, 32, 0),
    App.Vector(140, 50, 0),
]
s_curve = Part.BSplineCurve()
s_curve.interpolate(
    Points=s_points,
    InitialTangent=App.Vector(1, 0, 0),
    FinalTangent=App.Vector(1, 0, 0),
)
s_wire = Part.Wire([s_curve.toShape()])
s_assembly, s_outer, s_bore = swept_tube(s_wire, 10.0, 8.0, False)
s_edge = s_wire.Edges[0]
s_connection = {
    "passed": True,
    "initial_tangent_dot": float(
        normalized(s_edge.tangentAt(s_edge.FirstParameter)).dot(App.Vector(1, 0, 0))
    ),
    "final_tangent_dot": float(
        normalized(s_edge.tangentAt(s_edge.LastParameter)).dot(App.Vector(1, 0, 0))
    ),
    "connection_failures": [],
}
s_connection["passed"] = bool(
    s_connection["initial_tangent_dot"] >= 0.999
    and s_connection["final_tangent_dot"] >= 0.999
)
results["smooth_s_bend"] = demo_result(
    "smooth_s_bend", s_wire, s_assembly, s_outer, s_bore, 10.0, 8.0, s_connection
)
s_assembly.translate(App.Vector(130, 0, 0))
s_outer.translate(App.Vector(130, 0, 0))
s_bore.translate(App.Vector(130, 0, 0))
obj = doc.addObject("Part::Feature", "SmoothSBend")
obj.Label = "Smooth S-bend: endpoint-tangent B-spline"
obj.Shape = s_assembly
obj.ViewObject.ShapeColor = (0.25, 0.85, 0.45)
objects.append(obj)

# 3. 3D torsion/helical route: sampled analytic locus re-interpolated as one C2 B-spline.
helix_radius = 35.0
pitch = 80.0
turns = 1.0
sample_count = 9
h_points = []
for index in range(sample_count):
    t = 2.0 * math.pi * turns * index / (sample_count - 1)
    h_points.append(
        App.Vector(
            helix_radius * math.sin(t),
            helix_radius * (1.0 - math.cos(t)),
            pitch * t / (2.0 * math.pi),
        )
    )
h_tangent = normalized(App.Vector(helix_radius, 0, pitch / (2.0 * math.pi)))
h_curve = Part.BSplineCurve()
h_curve.interpolate(
    Points=h_points,
    InitialTangent=h_tangent,
    FinalTangent=h_tangent,
)
h_wire = Part.Wire([h_curve.toShape()])
h_assembly, h_outer, h_bore = swept_tube(h_wire, 4.0, 3.0, False)
h_edge = h_wire.Edges[0]
h_connection = {
    "passed": True,
    "initial_tangent_dot": float(
        normalized(h_edge.tangentAt(h_edge.FirstParameter)).dot(h_tangent)
    ),
    "final_tangent_dot": float(
        normalized(h_edge.tangentAt(h_edge.LastParameter)).dot(h_tangent)
    ),
    "connection_failures": [],
}
h_connection["passed"] = bool(
    h_connection["initial_tangent_dot"] >= 0.999
    and h_connection["final_tangent_dot"] >= 0.999
)
results["torsion_bspline"] = demo_result(
    "torsion_bspline", h_wire, h_assembly, h_outer, h_bore, 4.0, 3.0, h_connection
)
h_assembly.translate(App.Vector(360, 10, 0))
h_outer.translate(App.Vector(360, 10, 0))
h_bore.translate(App.Vector(360, 10, 0))
obj = doc.addObject("Part::Feature", "TorsionSpline")
obj.Label = "3D torsion route: C2 B-spline"
obj.Shape = h_assembly
obj.ViewObject.ShapeColor = (0.95, 0.55, 0.18)
objects.append(obj)

# Persist strict results before any GUI-only capture work.  This also leaves
# auditable evidence if the shared GUI session is interrupted during rendering.
doc.recompute()
with open(OUT_DIR + "/production_demos.validation.json", "w", encoding="utf-8") as handle:
    json.dump(results, handle, ensure_ascii=False, indent=2, sort_keys=True)

doc.recompute()
for target in objects:
    for obj in objects:
        obj.ViewObject.Visibility = obj is target
    Gui.activeDocument().activeView().viewAxonometric()
    Gui.activeDocument().activeView().fitAll()
    Gui.activeDocument().activeView().saveImage(
        OUT_DIR + "/" + target.Name + ".png", 1600, 1000, "White"
    )
for obj in objects:
    obj.ViewObject.Visibility = True
Gui.activeDocument().activeView().viewAxonometric()
Gui.activeDocument().activeView().fitAll()
Gui.activeDocument().activeView().saveImage(
    OUT_DIR + "/production_demos_overview.png", 2000, 1000, "White"
)
doc.recompute()
doc.saveAs(OUT_DIR + "/production_demos.FCStd")
with open(OUT_DIR + "/production_demos.validation.json", "w", encoding="utf-8") as handle:
    json.dump(results, handle, ensure_ascii=False, indent=2, sort_keys=True)
print("PRODUCTION_DEMOS=" + json.dumps(results, sort_keys=True, separators=(",", ":")))
