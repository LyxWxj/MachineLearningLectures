"""Generate dependency-free SVG diagrams for the Week 3 RL control examples."""

from pathlib import Path
from xml.etree.ElementTree import Element, ElementTree, SubElement, register_namespace


OUT = Path(__file__).parent
NS = "http://www.w3.org/2000/svg"
register_namespace("", NS)


def elt(parent, tag, **attrs):
    return SubElement(parent, f"{{{NS}}}{tag}", {key: str(value) for key, value in attrs.items()})


def label(parent, x, y, content, size=16, color="#0F172A", weight="normal", anchor="middle"):
    text = elt(parent, "text", x=x, y=y, fill=color, **{
        "font-family": "Arial, Helvetica, sans-serif",
        "font-size": size, "font-weight": weight, "text-anchor": anchor,
    })
    text.text = content
    return text


def box(parent, x, y, width, height, content, fill="#FFFFFF", stroke="#CBD5E1", size=16, weight="normal"):
    elt(parent, "rect", x=x, y=y, width=width, height=height, rx=10, fill=fill, stroke=stroke, **{"stroke-width": 1.6})
    lines = content.split("\n")
    first_baseline = y + height / 2 - (len(lines) - 1) * size * 0.62 + size * 0.35
    for index, line in enumerate(lines):
        label(parent, x + width / 2, first_baseline + index * size * 1.25, line, size, "#0F172A", weight)


def arrow(parent, x1, y1, x2, y2, color="#2563EB", width=3, dashed=False):
    attrs = {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "stroke": color,
             "stroke-width": width, "marker-end": "url(#arrow)"}
    if dashed:
        attrs["stroke-dasharray"] = "7 6"
    elt(parent, "line", **attrs)


def base(width, height):
    root = Element(f"{{{NS}}}svg", width=str(width), height=str(height), viewBox=f"0 0 {width} {height}", version="1.1")
    defs = elt(root, "defs")
    marker = elt(defs, "marker", id="arrow", viewBox="0 0 10 10", refX="8", refY="5", markerWidth="6", markerHeight="6", orient="auto")
    elt(marker, "path", d="M 0 0 L 10 5 L 0 10 z", fill="#2563EB")
    elt(root, "rect", width=width, height=height, fill="#FFFFFF")
    return root


def save(root, filename):
    ElementTree(root).write(OUT / filename, encoding="utf-8", xml_declaration=True)


def q_learning():
    root = base(1200, 480)
    label(root, 600, 38, "Q-learning: target uses the largest Q value in the next state", 25, weight="bold")
    label(root, 600, 68, "Behavior may explore; the update target is always greedy (off-policy).", 16, "#475569")
    box(root, 55, 145, 220, 90, "State A\nAction: Left", "#DBEAFE", "#2563EB", 20, "bold")
    box(root, 430, 145, 220, 90, "State B", "#EFF6FF", "#2563EB", 22, "bold")
    box(root, 925, 145, 220, 90, "Update Q(A, Left)", "#FEF3C7", "#D97706", 19, "bold")
    arrow(root, 275, 190, 425, 190)
    label(root, 350, 166, "r = 0", 15, "#2563EB", "bold")
    box(root, 385, 286, 145, 70, "Forward\nQ = 5", "#FDE68A", "#D97706", 17, "bold")
    box(root, 552, 286, 145, 70, "Explore\nQ = 1", "#F8FAFC", "#94A3B8", 17)
    label(root, 518, 397, "maxₐ Q(B, a) = 5", 18, "#B45309", "bold")
    arrow(root, 650, 190, 920, 190, "#D97706", 4, True)
    label(root, 785, 165, "Take the maximum, not the actual next action", 15, "#B45309", "bold")
    label(root, 1035, 182, "Target = 0 + 0.9 × 5 = 4.5", 16, "#92400E", "bold")
    label(root, 1035, 211, "Q ← Q + α(Target − Q)", 15, "#92400E")
    label(root, 600, 450, "Only Q(sₜ, aₜ) is updated, but the target assumes optimal actions from sₜ₊₁ onward.", 16, "#334155")
    save(root, "q_learning_update.svg")


def sarsa():
    root = base(1200, 510)
    label(root, 600, 38, "SARSA: the update uses the action actually taken next", 25, weight="bold")
    label(root, 600, 67, "Same transition: A --Left/r=0--> B; exploration then selects Explore at B.", 16, "#475569")
    for x, title, target, result, color in (
        (35, "SARSA (on-policy)", "Actual action: Q(B, Explore) = 1", "Target = 0.9\nQ(A, Left): 2 → 1.45", "#16A34A"),
        (620, "Q-learning (off-policy)", "Greedy action: max Q(B, ·) = 4", "Target = 3.6\nQ(A, Left): 2 → 2.8", "#D97706"),
    ):
        elt(root, "rect", x=x, y=105, width=545, height=335, rx=14, fill="#FFFFFF", stroke=color, **{"stroke-width": 2})
        label(root, x + 272, 143, title, 21, color, "bold")
        box(root, x + 38, 185, 135, 76, "A\nLeft", "#DBEAFE", "#2563EB", 20, "bold")
        box(root, x + 255, 185, 135, 76, "B", "#EFF6FF", "#2563EB", 22, "bold")
        arrow(root, x + 173, 223, x + 250, 223)
        label(root, x + 212, 201, "r = 0", 14, "#2563EB", "bold")
        if x == 35:
            box(root, x + 397, 165, 115, 60, "Forward\nQ = 4", "#F8FAFC", "#94A3B8", 15)
            box(root, x + 397, 248, 115, 60, "Explore\nQ = 1", "#DCFCE7", "#16A34A", 15, "bold")
            arrow(root, x + 390, 253, x + 395, 278, "#16A34A", 3)
        else:
            box(root, x + 397, 165, 115, 60, "Forward\nQ = 4", "#FDE68A", "#D97706", 15, "bold")
            box(root, x + 397, 248, 115, 60, "Explore\nQ = 1", "#F8FAFC", "#94A3B8", 15)
            arrow(root, x + 390, 215, x + 395, 195, "#D97706", 3, True)
        label(root, x + 272, 343, target, 17, color, "bold")
        label(root, x + 272, 380, result.split("\n")[0], 16, "#334155")
        label(root, x + 272, 407, result.split("\n")[1], 17, color, "bold")
    label(root, 600, 480, "SARSA evaluates the current policy as executed; Q-learning evaluates greedy behavior from the next step onward.", 16, "#334155")
    save(root, "sarsa_update.svg")


def dyna_q():
    root = base(1200, 490)
    label(root, 600, 38, "Dyna-Q: one real experience drives learning, modeling, and planning", 25, weight="bold")
    label(root, 600, 67, "Real interaction gathers data; the model reuses the same experience internally.", 16, "#475569")
    box(root, 50, 150, 215, 86, "Real environment\nA --Left/r=0--> B", "#DBEAFE", "#2563EB", 19, "bold")
    box(root, 380, 125, 205, 76, "Direct RL update\nQ(A, Left)", "#FEF3C7", "#D97706", 18, "bold")
    box(root, 380, 260, 205, 76, "Store world model\nModel(A, Left) = (0, B)", "#F3E8FF", "#9333EA", 16, "bold")
    arrow(root, 265, 175, 375, 164, "#2563EB")
    arrow(root, 265, 208, 375, 296, "#9333EA")
    box(root, 710, 125, 185, 76, "Planning replay 1\nB --Forward/r=10--> Terminal", "#DCFCE7", "#16A34A", 16, "bold")
    box(root, 965, 125, 185, 76, "Q(B, Forward)\n0 → 5 → 7.5", "#DCFCE7", "#16A34A", 17, "bold")
    arrow(root, 585, 298, 705, 163, "#9333EA", 3, True)
    arrow(root, 895, 163, 960, 163, "#16A34A")
    box(root, 710, 285, 185, 76, "Planning replay 2\nModel imagines A --Left--> B", "#F3E8FF", "#9333EA", 16, "bold")
    box(root, 965, 285, 185, 76, "Q(A, Left)\n0 → 3.375", "#FDE68A", "#D97706", 18, "bold")
    arrow(root, 585, 298, 705, 323, "#9333EA", 3, True)
    arrow(root, 895, 323, 960, 323, "#D97706")
    label(root, 600, 430, "Planning needs no new environmental interaction: sample past experience from the model and repeat Q-learning updates.", 16, "#334155")
    save(root, "dyna_q_planning.svg")


if __name__ == "__main__":
    q_learning()
    sarsa()
    dyna_q()
