"""Generate English-labelled SVG diagrams for diffusion, flow matching, and rectified flow."""

from math import pi, sin
from pathlib import Path
from xml.etree.ElementTree import Element, ElementTree, SubElement, register_namespace


ROOT = Path(__file__).parent
NS = "http://www.w3.org/2000/svg"
register_namespace("", NS)


def node(parent, tag, **attrs):
    return SubElement(parent, f"{{{NS}}}{tag}", {key: str(value) for key, value in attrs.items()})


def text(parent, x, y, content, size=14, color="#0F172A", anchor="middle", weight="normal"):
    item = node(parent, "text", x=x, y=y, fill=color, **{
        "font-family": "Arial, Helvetica, sans-serif", "font-size": size,
        "font-weight": weight, "text-anchor": anchor,
    })
    item.text = content


def path(points):
    return "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in points)


def add_defs(svg):
    group = node(svg, "defs")
    marker = node(group, "marker", id="arrow", viewBox="0 0 10 10", refX="8", refY="5", markerWidth="7", markerHeight="7", orient="auto")
    node(marker, "path", d="M 0 0 L 10 5 L 0 10 z", fill="#334155")


def ellipse_cloud(group, cx, cy, rx, ry, angle, color, label_text):
    node(group, "ellipse", cx=cx, cy=cy, rx=rx, ry=ry, fill=color, opacity="0.16", stroke=color, **{"stroke-width": 2.5, "transform": f"rotate({angle} {cx} {cy})"})
    node(group, "ellipse", cx=cx, cy=cy, rx=rx * 0.66, ry=ry * 0.66, fill="none", stroke=color, opacity="0.7", **{"stroke-width": 1.6, "transform": f"rotate({angle} {cx} {cy})"})
    text(group, cx, cy + ry + 28, label_text, 14, color, weight="bold")


def distribution_transport():
    svg = Element(f"{{{NS}}}svg", width="1180", height="420", viewBox="0 0 1180 420", version="1.1")
    node(svg, "rect", width=1180, height=420, fill="#FFFFFF")
    add_defs(svg)
    text(svg, 590, 34, "Diffusion as Distribution Transport", 22, weight="bold")
    text(svg, 590, 59, "Forward noising moves data toward simple noise; generation learns the reverse transport", 13, "#475569")
    centers = (150, 390, 630, 870, 1050)
    stages = (
        ("Data distribution", 52, 112, -22, "#2563EB", "t = 0"),
        ("Noisy distribution", 48, 90, -12, "#4F81BD", "t = 0.25"),
        ("Intermediate", 46, 73, -5, "#64748B", "t = 0.5"),
        ("Almost noise", 44, 61, 0, "#D97706", "t = 0.75"),
        ("Gaussian noise", 56, 56, 0, "#DC2626", "t = 1"),
    )
    for index, (title, rx, ry, angle, color, time) in enumerate(stages):
        ellipse_cloud(svg, centers[index], 205, rx, ry, angle, color, title)
        text(svg, centers[index], 118, time, 12, "#475569")
        if index < len(stages) - 1:
            node(svg, "line", x1=centers[index] + 66, y1=205, x2=centers[index + 1] - 66, y2=205, stroke="#334155", **{"stroke-width": 2, "marker-end": "url(#arrow)"})
    text(svg, 590, 365, "Forward noising: q_t gradually destroys structure", 15, "#2563EB", weight="bold")
    text(svg, 590, 390, "Reverse generation learns a vector field or score that transports p_1 back to p_0", 15, "#DC2626", weight="bold")
    ElementTree(svg).write(ROOT / "diffusion_distribution_transport.svg", encoding="utf-8", xml_declaration=True)


def flow_paths():
    svg = Element(f"{{{NS}}}svg", width="1180", height="520", viewBox="0 0 1180 520", version="1.1")
    node(svg, "rect", width=1180, height=520, fill="#FFFFFF")
    add_defs(svg)
    text(svg, 590, 34, "From Stochastic Paths to Straight Flows", 22, weight="bold")
    text(svg, 590, 59, "A vector field tells each point how to move as time t changes", 13, "#475569")
    text(svg, 285, 98, "Diffusion / score-based view", 17, "#2563EB", weight="bold")
    text(svg, 895, 98, "Flow Matching / Rectified Flow", 17, "#16A34A", weight="bold")
    for base_y in (168, 220, 272, 324):
        points = []
        for step in range(31):
            u = step / 30
            x = 82 + u * 405
            y = base_y + 32 * sin(pi * u + base_y / 45) * (1 - 0.15 * u)
            points.append((x, y))
        node(svg, "path", d=path(points), fill="none", stroke="#2563EB", opacity="0.68", **{"stroke-width": 2})
    for base_y in (168, 220, 272, 324):
        node(svg, "line", x1=690, y1=base_y, x2=1090, y2=base_y - 28, stroke="#16A34A", opacity="0.75", **{"stroke-width": 2.6, "marker-end": "url(#arrow)"})
    ellipse_cloud(svg, 86, 245, 40, 76, -20, "#2563EB", "source p_0")
    ellipse_cloud(svg, 490, 245, 54, 54, 0, "#D97706", "target p_1")
    ellipse_cloud(svg, 690, 245, 40, 76, -20, "#2563EB", "source p_0")
    ellipse_cloud(svg, 1090, 217, 54, 54, 0, "#D97706", "target p_1")
    text(svg, 285, 405, "Many small reverse steps; paths can bend", 14, "#334155")
    text(svg, 895, 405, "Fit v_theta(x,t) to velocities along interpolating paths", 14, "#334155")
    text(svg, 590, 466, "Rectification repeatedly straightens the learned coupling, enabling coarse ODE integration.", 15, "#334155", weight="bold")
    ElementTree(svg).write(ROOT / "flow_matching_paths.svg", encoding="utf-8", xml_declaration=True)


def sampler_steps():
    svg = Element(f"{{{NS}}}svg", width="1180", height="400", viewBox="0 0 1180 400", version="1.1")
    node(svg, "rect", width=1180, height=400, fill="#FFFFFF")
    add_defs(svg)
    text(svg, 590, 34, "Why Straight Flows Can Use Fewer Inference Steps", 22, weight="bold")
    text(svg, 590, 59, "The same endpoint distributions can be connected by different numerical trajectories", 13, "#475569")
    rows = (("Diffusion sampler", "Many stochastic denoising steps", 12, "#2563EB", 100), ("Rectified-flow sampler", "Few deterministic ODE steps", 4, "#16A34A", 260))
    for title, subtitle, count, color, y in rows:
        text(svg, 35, y + 6, title, 15, color, anchor="start", weight="bold")
        text(svg, 35, y + 29, subtitle, 12, "#475569", anchor="start")
        start, end = 270, 1080
        for i in range(count):
            u = i / (count - 1) if count > 1 else 0
            x = start + u * (end - start)
            node(svg, "circle", cx=x, cy=y, r=15 if count < 6 else 9, fill=color, opacity="0.2", stroke=color, **{"stroke-width": 2})
            if i < count - 1:
                nx = start + (i + 1) / (count - 1) * (end - start)
                node(svg, "line", x1=x + 18, y1=y, x2=nx - 18, y2=y, stroke=color, **{"stroke-width": 2, "marker-end": "url(#arrow)"})
        text(svg, start - 10, y + 5, "noise", 11, "#334155", anchor="end")
        text(svg, end + 10, y + 5, "data", 11, "#334155", anchor="start")
    text(svg, 590, 360, "Fewer steps reduce inference cost, but quality still depends on the learned vector field and solver.", 15, "#334155", weight="bold")
    ElementTree(svg).write(ROOT / "diffusion_vs_flow_steps.svg", encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    distribution_transport()
    flow_paths()
    sampler_steps()
