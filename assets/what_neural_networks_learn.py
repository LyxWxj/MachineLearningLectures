"""Generate one dependency-free SVG for each neural-network learning experiment."""

from math import cos, pi, sin
from pathlib import Path
from random import Random
from xml.etree.ElementTree import Element, ElementTree, SubElement, register_namespace


ROOT = Path(__file__).parent
NS = "http://www.w3.org/2000/svg"
register_namespace("", NS)


def node(parent, tag, **attrs):
    return SubElement(parent, f"{{{NS}}}{tag}", {key: str(value) for key, value in attrs.items()})


def label(parent, x, y, value, size=13, color="#0F172A", anchor="middle", weight="normal"):
    item = node(parent, "text", x=x, y=y, fill=color, **{
        "font-family": "Arial, Helvetica, sans-serif", "font-size": size,
        "font-weight": weight, "text-anchor": anchor,
    })
    item.text = value


def path(points):
    return "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in points)


def frame(svg, title, subtitle):
    node(svg, "rect", width=960, height=470, fill="#FFFFFF")
    label(svg, 480, 34, title, 22, weight="bold")
    label(svg, 480, 58, subtitle, 13, "#475569")


def axes(parent, left, top, width, height, x_name, y_name):
    node(parent, "rect", x=left, y=top, width=width, height=height, fill="#FFFFFF", stroke="#94A3B8", **{"stroke-width": 1.2})
    label(parent, left + width / 2, top + height + 25, x_name, 12, "#334155")
    label(parent, left - 14, top + 15, y_name, 12, "#334155", anchor="end")


def random_labels():
    svg = Element(f"{{{NS}}}svg", width="960", height="470", viewBox="0 0 960 470", version="1.1")
    frame(svg, "Experiment 0: Random Labels", "The labels have no relation to position, yet a large network can fit the training set")
    left, top, width, height = 55, 95, 375, 270
    axes(svg, left, top, width, height, "x1", "x2")
    rng = Random(7)
    for _ in range(110):
        x, y = left + 15 + rng.random() * (width - 30), top + 15 + rng.random() * (height - 30)
        color = "#DC2626" if rng.randrange(2) else "#2563EB"
        node(svg, "circle", cx=x, cy=y, r=3.8, fill=color, opacity="0.9")
    label(svg, left + 12, 392, "independent random labels", 13, "#475569", anchor="start")
    label(svg, 700, 94, "Accuracy during training", 16, weight="bold")
    plot_left, plot_top, plot_width, plot_height = 510, 120, 380, 245
    axes(svg, plot_left, plot_top, plot_width, plot_height, "epochs", "accuracy")
    train, test = [], []
    for step in range(101):
        u = step / 100
        x = plot_left + u * plot_width
        train_acc = 0.50 + 0.50 * (1 - pow(2.71828, -u * 6.2))
        test_acc = 0.50 + 0.022 * sin(18 * u) * (1 - 0.3 * u)
        train.append((x, plot_top + plot_height * (1 - train_acc)))
        test.append((x, plot_top + plot_height * (1 - test_acc)))
    node(svg, "line", x1=plot_left, y1=plot_top + plot_height * 0.5, x2=plot_left + plot_width, y2=plot_top + plot_height * 0.5, stroke="#94A3B8", **{"stroke-dasharray": "5 4"})
    node(svg, "path", d=path(train), fill="none", stroke="#16A34A", **{"stroke-width": 3})
    node(svg, "path", d=path(test), fill="none", stroke="#D97706", **{"stroke-width": 3})
    label(svg, 862, 143, "train -> 100%", 12, "#16A34A", anchor="end", weight="bold")
    label(svg, 862, plot_top + plot_height * 0.5 - 8, "test ~ 50%", 12, "#D97706", anchor="end", weight="bold")
    label(svg, 480, 433, "Principle: fitting a finite training set does not imply a rule that transfers to new points.", 15, "#334155", weight="bold")
    ElementTree(svg).write(ROOT / "experiment_random_labels.svg", encoding="utf-8", xml_declaration=True)


def spectral_bias_figure(filename, title, subtitle, fit, fit_label, fit_color):
    svg = Element(f"{{{NS}}}svg", width="960", height="470", viewBox="0 0 960 470", version="1.1")
    frame(svg, title, subtitle)
    left, top, width, height = 85, 95, 790, 250
    axes(svg, left, top, width, height, "x", "f(x)")

    def point(x, value):
        return left + (x + pi) / (2 * pi) * width, top + height / 2 - value * 88

    xs = [-pi + 2 * pi * i / 200 for i in range(201)]
    target = [point(x, sin(x) + 0.15 * sin(10 * x)) for x in xs]
    fitted = [point(x, fit(x)) for x in xs]
    node(svg, "path", d=path(target), fill="none", stroke="#64748B", **{"stroke-width": 2, "stroke-dasharray": "6 4"})
    node(svg, "path", d=path(fitted), fill="none", stroke=fit_color, **{"stroke-width": 3})
    label(svg, 120, 385, "target: sin(x) + 0.15 sin(10x)", 13, "#475569", anchor="start")
    label(svg, 560, 385, fit_label, 13, fit_color, anchor="start")
    label(svg, 480, 433, "Principle: the same target can be represented, but optimization reaches different frequency components at different times.", 15, "#334155", weight="bold")
    ElementTree(svg).write(ROOT / filename, encoding="utf-8", xml_declaration=True)


def spectral_bias():
    target = lambda x: sin(x) + 0.15 * sin(10 * x)
    spectral_bias_figure(
        "experiment_spectral_bias_early.svg",
        "Experiment 1A: Early Spectral Fit",
        "The first visible fit follows the large, low-frequency component",
        lambda x: 0.92 * sin(x),
        "early fit: approximately sin(x)",
        "#2563EB",
    )
    spectral_bias_figure(
        "experiment_spectral_bias_late.svg",
        "Experiment 1B: Late Spectral Fit",
        "With more optimization, the small high-frequency component is recovered",
        target,
        "late fit: target detail recovered",
        "#DC2626",
    )


def grokking():
    svg = Element(f"{{{NS}}}svg", width="960", height="470", viewBox="0 0 960 470", version="1.1")
    frame(svg, "Experiment 2: Grokking", "Training accuracy can saturate long before held-out modular-arithmetic accuracy rises")
    left, top, width, height = 85, 95, 790, 255
    axes(svg, left, top, width, height, "training steps", "accuracy")
    train, test = [], []
    for step in range(201):
        u = step / 200
        x = left + u * width
        train_acc = 0.15 + 0.85 * (1 - pow(2.71828, -u * 16))
        test_acc = 0.02 + 0.98 / (1 + pow(2.71828, -(u - 0.72) * 27))
        train.append((x, top + height * (1 - train_acc)))
        test.append((x, top + height * (1 - test_acc)))
    node(svg, "path", d=path(train), fill="none", stroke="#2563EB", **{"stroke-width": 3})
    node(svg, "path", d=path(test), fill="none", stroke="#D97706", **{"stroke-width": 3})
    boundary = left + 0.70 * width
    node(svg, "line", x1=boundary, y1=top, x2=boundary, y2=top + height, stroke="#64748B", **{"stroke-width": 1.5, "stroke-dasharray": "5 4"})
    label(svg, left + 100, top + 28, "train accuracy -> 100%", 13, "#2563EB", anchor="start", weight="bold")
    label(svg, boundary + 10, top + 30, "generalization onset", 12, "#475569", anchor="start")
    label(svg, left + 320, top + height - 15, "test near random", 13, "#D97706", anchor="start", weight="bold")
    label(svg, 480, 405, "Early phase: memorize the training pairs", 14, "#475569")
    label(svg, 480, 433, "Principle: training loss alone cannot reveal delayed generalization; compare train and test curves.", 15, "#334155", weight="bold")
    ElementTree(svg).write(ROOT / "experiment_grokking.svg", encoding="utf-8", xml_declaration=True)


def superposition():
    svg = Element(f"{{{NS}}}svg", width="960", height="470", viewBox="0 0 960 470", version="1.1")
    frame(svg, "Experiment 3: Superposition", "When representation dimensions are scarce, several sparse features can share non-orthogonal directions")
    cx, cy = 480, 225
    node(svg, "circle", cx=cx, cy=cy, r=130, fill="#F8FAFC", stroke="#CBD5E1", **{"stroke-width": 1.5})
    node(svg, "line", x1=cx - 150, y1=cy, x2=cx + 150, y2=cy, stroke="#94A3B8", **{"stroke-width": 1.2})
    node(svg, "line", x1=cx, y1=cy - 150, x2=cx, y2=cy + 150, stroke="#94A3B8", **{"stroke-width": 1.2})
    label(svg, cx + 156, cy + 5, "hidden dim 1", 12, "#64748B", anchor="start")
    label(svg, cx + 10, cy - 156, "hidden dim 2", 12, "#64748B")
    angles = (-76, -18, 40, 102, 164, 222)
    colors = ("#2563EB", "#DC2626", "#D97706", "#16A34A", "#7C3AED", "#0891B2")
    for index, (angle, color) in enumerate(zip(angles, colors), 1):
        radians = angle * pi / 180
        dx, dy = 112 * cos(radians), 112 * sin(radians)
        node(svg, "line", x1=cx - dx, y1=cy - dy, x2=cx + dx, y2=cy + dy, stroke=color, **{"stroke-width": 3.5})
        label(svg, cx + 150 * cos(radians), cy + 150 * sin(radians) + 4, f"Feature {index}", 12, color, weight="bold")
    node(svg, "circle", cx=cx, cy=cy, r=6, fill="#0F172A")
    label(svg, 480, 403, "6 sparse features encoded in a 2-dimensional hidden representation", 15, "#334155", weight="bold")
    label(svg, 480, 433, "Principle: a coordinate is not necessarily one concept; meaning can be distributed and directions shared.", 15, "#334155", weight="bold")
    ElementTree(svg).write(ROOT / "experiment_superposition.svg", encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    random_labels()
    spectral_bias()
    grokking()
    superposition()
