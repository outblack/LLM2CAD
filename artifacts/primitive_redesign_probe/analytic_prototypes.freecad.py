import json
import math
import FreeCAD as App
import FreeCADGui as Gui
import Part

DOC_NAME = "PrimitiveRedesignAnalyticPrototypes"
OUT_DIR = "/Users/jhkim/Documents/cadgen02/artifacts/primitive_redesign_probe"


def normalized(value):
    result = App.Vector(value.x, value.y, value.z)
    result.normalize()
    return result


def circle_wire(radius, center, normal):
    return Part.Wire([Part.makeCircle(radius, center, normalized(normal))])


def metrics(shape):
    errors = []
    try:
        errors = [str(item) for item in (shape.check(True) or [])]
    except Exception as exc:
        errors = ["check_exception: " + str(exc)]
    result = {
        "is_null": bool(shape.isNull()),
        "is_valid": bool(shape.isValid()),
        "is_closed": bool(shape.isClosed()),
        "solid_count": len(shape.Solids),
        "volume": float(shape.Volume),
        "bop_errors": errors,
    }
    result["passed"] = bool(
        not result["is_null"]
        and result["is_valid"]
        and result["is_closed"]
        and result["solid_count"] == 1
        and result["volume"] > 1e-9
        and not errors
    )
    return result


if DOC_NAME in App.listDocuments():
    App.closeDocument(DOC_NAME)
doc = App.newDocument(DOC_NAME)
results = {}

# Prototype A: exact one-edge circular centerline and co-terminal annular sweeps.
start = App.Vector(0, 0, 0)
middle = App.Vector(30 / math.sqrt(2), 30 * (1 - 1 / math.sqrt(2)), 0)
end = App.Vector(30, 30, 0)
arc_edge = Part.Arc(start, middle, end).toShape()
arc_wire = Part.Wire([arc_edge])
tangent = normalized(arc_edge.tangentAt(arc_edge.FirstParameter))
arc_outer = arc_wire.makePipeShell([circle_wire(10, start, tangent)], True, True)
arc_bore = arc_wire.makePipeShell([circle_wire(8, start, tangent)], True, True)
analytic_arc = arc_outer.cut(arc_bore).removeSplitter()
obj = doc.addObject("Part::Feature", "AnalyticArcSweep")
obj.Label = "A. analytic circular-arc annular sweep"
obj.Shape = analytic_arc
obj.ViewObject.ShapeColor = (0.20, 0.70, 0.95)
results["analytic_arc_sweep"] = metrics(analytic_arc)
results["analytic_arc_sweep"]["centerline_edge_count"] = len(arc_wire.Edges)

# Prototype B: exact torus segment, which avoids profile/end-face sweep ambiguity.
torus_center = App.Vector(85, 0, 0)
torus_outer = Part.makeTorus(30, 10, torus_center, App.Vector(0, 0, 1), -180, 180, 90)
torus_bore = Part.makeTorus(30, 8, torus_center, App.Vector(0, 0, 1), -180, 180, 90)
torus_elbow = torus_outer.cut(torus_bore).removeSplitter()
obj = doc.addObject("Part::Feature", "TorusElbow")
obj.Label = "B. exact quarter-torus elbow"
obj.Shape = torus_elbow
obj.ViewObject.ShapeColor = (0.25, 0.85, 0.45)
results["torus_elbow"] = metrics(torus_elbow)

# Prototype C: native helical centerline swept as one continuous hollow solid.
helix_wire = Part.makeHelix(25, 75, 25)
helix_edge = helix_wire.Edges[0]
helix_start = helix_edge.valueAt(helix_edge.FirstParameter)
helix_tangent = normalized(helix_edge.tangentAt(helix_edge.FirstParameter))
helix_outer = helix_wire.makePipeShell(
    [circle_wire(6, helix_start, helix_tangent)], True, True
)
helix_bore = helix_wire.makePipeShell(
    [circle_wire(4, helix_start, helix_tangent)], True, True
)
helix_tube = helix_outer.cut(helix_bore).removeSplitter()
helix_tube.translate(App.Vector(170, 0, 0))
obj = doc.addObject("Part::Feature", "NativeHelixTube")
obj.Label = "C. Part.makeHelix continuous tube"
obj.Shape = helix_tube
obj.ViewObject.ShapeColor = (0.95, 0.55, 0.18)
results["native_helix_pipe_shell"] = metrics(helix_tube)
results["native_helix_pipe_shell"]["centerline_edge_count"] = len(helix_wire.Edges)

doc.recompute()
Gui.activeDocument().activeView().viewAxonometric()
Gui.activeDocument().activeView().fitAll()
Gui.activeDocument().activeView().saveImage(
    OUT_DIR + "/analytic_prototypes.png", 1800, 900, "White"
)
with open(OUT_DIR + "/analytic_prototypes.validation.json", "w", encoding="utf-8") as handle:
    json.dump(results, handle, ensure_ascii=False, indent=2, sort_keys=True)
print("PROBE_RESULT=" + json.dumps(results, sort_keys=True, separators=(",", ":")))
