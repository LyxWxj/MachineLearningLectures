"""Render the TD-learning value-propagation example as a dependency-free SVG."""

from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, ElementTree, register_namespace


OUTPUT = Path(__file__).with_suffix(".svg")
NS = "http://www.w3.org/2000/svg"
register_namespace("", NS)


def node(parent, tag, **attrs):
    return SubElement(parent, f"{{{NS}}}{tag}", {key: str(value) for key, value in attrs.items()})


def text(parent, x, y, content, size, color="#0F172A", weight="normal", anchor="middle"):
    item = node(parent, "text", x=x, y=y, fill=color, **{
        "font-family": "Arial, Helvetica, sans-serif",
        "font-size": size, "font-weight": weight, "text-anchor": anchor,
    })
    item.text = content
    return item


def draw_path(group, active_edge):
    positions = {"A": (72, 174), "B": (200, 174), "C": (328, 174)}
    for start, end, reward in (("A", "B", "r = 0"), ("B", "C", "r = +4")):
        active = (start, end) == active_edge
        color = "#2563EB" if active else "#94A3B8"
        x1, y1 = positions[start]
        x2, y2 = positions[end]
        node(group, "line", x1=x1 + 31, y1=y1, x2=x2 - 34, y2=y2,
             stroke=color, **{"stroke-width": 4 if active else 2, "marker-end": "url(#arrow)"})
        text(group, (x1 + x2) / 2, 138, reward, 14, color, "bold" if active else "normal")
    for name, (x, y) in positions.items():
        active = name in active_edge
        node(group, "circle", cx=x, cy=y, r=31, fill="#DBEAFE" if active else "#F1F5F9",
             stroke="#2563EB" if active else "#64748B", **{"stroke-width": 2.5})
        text(group, x, y + 7, name, 25, "#1E3A8A" if active else "#334155", "bold")


def draw_table(group, values, changed):
    labels = ("V(A)", "V(B)", "V(C)")
    for index, (label, value) in enumerate(zip(labels, values)):
        x = 35 + index * 122
        node(group, "rect", x=x, y=294, width=112, height=65, rx=8,
             fill="#FDE68A" if label == changed else "#F8FAFC", stroke="#CBD5E1", **{"stroke-width": 1.2})
        text(group, x + 56, 319, label, 14, "#334155")
        text(group, x + 56, 346, str(value), 20, "#0F172A", "bold")


def main():
    svg = Element(f"{{{NS}}}svg", width="1200", height="455", viewBox="0 0 1200 455", version="1.1")
    defs = node(svg, "defs")
    marker = node(defs, "marker", id="arrow", viewBox="0 0 10 10", refX="8", refY="5",
                  markerWidth="6", markerHeight="6", orient="auto-start-reverse")
    node(marker, "path", d="M 0 0 L 10 5 L 0 10 z", fill="#2563EB")
    node(svg, "rect", width="1200", height="455", fill="white")

    steps = (
        ("Transition 1", ("A", "B"), "A → B; no known value yet", "δ = 0 + 0.9 × 0 − 0 = 0", "V(A) ← 0", (0, 0, 0), "V(A)"),
        ("Transition 2", ("B", "C"), "B → C; terminal reward observed", "δ = 4 + 0.9 × 0 − 0 = 4", "V(B) ← 0 + 0.5 × 4 = 2", (0, 2, 0), "V(B)"),
        ("Next episode", ("A", "B"), "A → B; use updated V(B)", "δ = 0 + 0.9 × 2 − 0 = 1.8", "V(A) ← 0 + 0.5 × 1.8 = 0.9", (0.9, 2, 0), "V(A)"),
    )
    for index, (title, edge, subtitle, delta, update, values, changed) in enumerate(steps):
        group = node(svg, "g", transform=f"translate({20 + 390 * index}, 16)")
        node(group, "rect", x=0, y=0, width=370, height=380, rx=12, fill="white", stroke="#CBD5E1", **{"stroke-width": 1.5})
        text(group, 185, 39, title, 21, "#0F172A", "bold")
        text(group, 185, 68, subtitle, 14, "#475569")
        draw_path(group, edge)
        text(group, 185, 235, delta, 15, "#1E3A8A")
        text(group, 185, 267, update, 15, "#B45309", "bold")
        text(group, 35, 285, "Values after update", 13, "#64748B", anchor="start")
        draw_table(group, values, changed)
    text(svg, 600, 430, "TD(0) bootstraps from the next-state estimate: reward updates B first, then propagates to A on the next A → B transition.  γ = 0.9, α = 0.5", 16, "#334155")
    ElementTree(svg).write(OUTPUT, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    main()
