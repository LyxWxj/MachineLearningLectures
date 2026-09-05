"""Generate dependency-free SVG diagrams for the regularization introduction."""

from math import exp, pi, sin
from pathlib import Path
from xml.etree.ElementTree import Element, ElementTree, SubElement, register_namespace


ROOT = Path(__file__).parent
NS = "http://www.w3.org/2000/svg"
register_namespace("", NS)


def node(parent, tag, **attrs):
    return SubElement(parent, f"{{{NS}}}{tag}", {key: str(value) for key, value in attrs.items()})


def text(parent, x, y, content, size=13, color="#0F172A", anchor="middle", weight="normal"):
    item = node(parent, "text", x=x, y=y, fill=color, **{
        "font-family": "Arial, Helvetica, sans-serif", "font-size": size,
        "font-weight": weight, "text-anchor": anchor,
    })
    item.text = content


def line_path(points):
    return "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in points)


def objectives():
    svg = Element(f"{{{NS}}}svg", width="1000", height="455", viewBox="0 0 1000 455", version="1.1")
    node(svg, "rect", width=1000, height=455, fill="#FFFFFF")
    text(svg, 500, 34, "Regularization and Optimization: Three Training Goals", 22, weight="bold")
    text(svg, 500, 58, "Generalize, reach that solution efficiently, and keep updates stable", 13, "#475569")

    items = (
        ("1", "Generalization", "Find a solution that works on unseen data", "#2563EB"),
        ("2", "Efficiency", "Reach similar test performance with fewer updates", "#D97706"),
        ("3", "Stability", "Make loss decrease reliably during training", "#16A34A"),
    )
    for index, (number, title, subtitle, color) in enumerate(items):
        x = 35 + index * 322
        node(svg, "rect", x=x, y=100, width=286, height=175, rx=8, fill="#F8FAFC", stroke="#CBD5E1", **{"stroke-width": 1.3})
        node(svg, "circle", cx=x + 44, cy=143, r=23, fill=color)
        text(svg, x + 44, 150, number, 19, "#FFFFFF", weight="bold")
        text(svg, x + 82, 143, title, 18, anchor="start", weight="bold")
        text(svg, x + 22, 191, subtitle, 13, "#334155", anchor="start")
        if index == 0:
            node(svg, "path", d=f"M {x+25} 242 C {x+75} 205, {x+128} 238, {x+180} 217 S {x+240} 221, {x+262} 205", fill="none", stroke=color, **{"stroke-width": 3})
        elif index == 1:
            node(svg, "path", d=f"M {x+25} 240 C {x+80} 238, {x+113} 218, {x+150} 205 S {x+222} 194, {x+262} 190", fill="none", stroke=color, **{"stroke-width": 3})
            node(svg, "path", d=f"M {x+25} 240 C {x+105} 239, {x+160} 230, {x+210} 218 S {x+240} 213, {x+262} 211", fill="none", stroke="#94A3B8", **{"stroke-width": 2, "stroke-dasharray": "5 4"})
        else:
            node(svg, "path", d=f"M {x+25} 240 L {x+68} 215 L {x+104} 229 L {x+140} 199 L {x+180} 211 L {x+221} 186 L {x+262} 181", fill="none", stroke=color, **{"stroke-width": 3})
    node(svg, "line", x1=95, y1=337, x2=905, y2=337, stroke="#CBD5E1", **{"stroke-width": 1.5})
    text(svg, 500, 377, "Regularization selects among fitting solutions; optimization controls the path used to reach them.", 16, "#334155")
    text(svg, 500, 408, "Neither training accuracy nor a single loss value is sufficient evidence of generalization.", 14, "#64748B")
    ElementTree(svg).write(ROOT / "regularization_objectives.svg", encoding="utf-8", xml_declaration=True)


def landscape():
    svg = Element(f"{{{NS}}}svg", width="1180", height="585", viewBox="0 0 1180 585", version="1.1")
    node(svg, "rect", width=1180, height=585, fill="#FFFFFF")
    defs = node(svg, "defs")
    marker = node(defs, "marker", id="arrow", viewBox="0 0 10 10", refX="8", refY="5", markerWidth="6", markerHeight="6", orient="auto")
    node(marker, "path", d="M 0 0 L 10 5 L 0 10 z", fill="#334155")
    text(svg, 590, 34, "Loss Landscape Intuition", 22, weight="bold")
    text(svg, 590, 58, "Two-dimensional contour sketches: closer contours mean a steeper change in loss", 13, "#475569")
    panels = (
        ("Sharp Minimum", "Narrow valley; sensitive to perturbations", "sharp", "#DC2626"),
        ("Flat Minimum", "Wide valley; locally more robust", "flat", "#16A34A"),
        ("Saddle Region", "Downhill in one direction, uphill in another", "saddle", "#7C3AED"),
        ("High Plateau", "Very small slope; model is underfit", "plateau", "#D97706"),
    )
    for index, (title, subtitle, kind, color) in enumerate(panels):
        left, top, width, height = 25 + index * 290, 100, 245, 335
        node(svg, "rect", x=left, y=top, width=width, height=height, rx=8, fill="#F8FAFC", stroke="#CBD5E1", **{"stroke-width": 1.3})
        text(svg, left + width / 2, top + 31, title, 16, weight="bold")
        text(svg, left + width / 2, top + 52, subtitle, 10, "#475569")
        plot_left, plot_top, plot_width, plot_height = left + 25, top + 75, 195, 195
        node(svg, "rect", x=plot_left, y=plot_top, width=plot_width, height=plot_height, fill="#FFFFFF", stroke="#94A3B8")
        cx, cy = plot_left + plot_width / 2, plot_top + plot_height / 2
        if kind in ("sharp", "flat"):
            radii = (18, 34, 50, 66, 82) if kind == "sharp" else (30, 52, 74, 94)
            for radius in radii:
                node(svg, "ellipse", cx=cx, cy=cy, rx=radius * (0.48 if kind == "sharp" else 1.0), ry=radius,
                     fill="none", stroke=color, opacity="0.78", **{"stroke-width": 2})
            node(svg, "circle", cx=cx, cy=cy, r=5, fill=color)
            node(svg, "line", x1=cx - 70, y1=cy - 72, x2=cx - 18, y2=cy - 20, stroke="#334155", **{"stroke-width": 1.5, "marker-end": "url(#arrow)"})
        elif kind == "saddle":
            # Bounded hyperbolic branches: the crossing makes the saddle explicit.
            for offset in (16, 32, 48, 64):
                points_a, points_b = [], []
                for step in range(41):
                    u = -1.0 + 2.0 * step / 40
                    v = offset + (82 - offset) * abs(u)
                    points_a.append((cx + u * 82, cy + v))
                    points_b.append((cx + u * 82, cy - v))
                node(svg, "path", d=line_path(points_a), fill="none", stroke=color, opacity="0.78", **{"stroke-width": 2})
                node(svg, "path", d=line_path(points_b), fill="none", stroke=color, opacity="0.78", **{"stroke-width": 2})
            node(svg, "circle", cx=cx, cy=cy, r=5, fill=color)
            node(svg, "line", x1=cx - 74, y1=cy, x2=cx - 18, y2=cy, stroke="#334155", **{"stroke-width": 1.5, "marker-end": "url(#arrow)"})
        else:
            node(svg, "rect", x=plot_left + 18, y=plot_top + 18, width=plot_width - 36, height=plot_height - 36, fill="#FEF3C7", stroke="#F59E0B", **{"stroke-width": 2})
            for radius in (22, 42, 62):
                node(svg, "ellipse", cx=cx, cy=cy + 10, rx=radius * 1.25, ry=radius * 0.75, fill="none", stroke=color, opacity="0.78", **{"stroke-width": 2})
            node(svg, "circle", cx=cx, cy=cy + 10, r=5, fill=color)
            node(svg, "line", x1=cx - 76, y1=cy - 68, x2=cx - 20, y2=cy - 18, stroke="#334155", **{"stroke-width": 1.5, "marker-end": "url(#arrow)"})
        text(svg, plot_left + plot_width / 2, plot_top + plot_height + 20, "parameter 1", 10, "#64748B")
        text(svg, plot_left - 10, plot_top + 12, "parameter 2", 10, "#64748B", anchor="end")
    text(svg, 590, 485, "Contour spacing visualizes local sensitivity: dense contours = steep loss change; sparse contours = shallow change.", 14, "#334155")
    text(svg, 590, 515, "Flatness is a useful heuristic, but it changes under reparameterization; validate generalization on held-out data.", 14, "#334155")
    text(svg, 590, 545, "Optimizers, schedules, normalization, and regularization alter both the route and the solution reached.", 14, "#334155")
    ElementTree(svg).write(ROOT / "loss_landscape_states.svg", encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    objectives()
    landscape()
