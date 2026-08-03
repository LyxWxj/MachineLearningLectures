# -*- coding: utf-8 -*-
"""
Speed-accuracy tradeoff in the drift-diffusion model (DDM).
Companion figure for ch07-neuromatch-week3.md, section "DDM 的速度-精度权衡".

Continuous-time DDM (classical parameterization): the decision variable
x(t) is a drifted Brownian motion
    dx = mu*s*dt + sigma*dW,   x(0) = 0,   absorbing boundaries +B / -B.
For the true state s=+1 (drift toward +B), hitting -B first is an error.
First-passage theory gives (exact in the continuous-time limit):

    P(error) = 1 / (1 + exp(2*mu*B / sigma^2))
    E[tau]   = (B/mu) * tanh(mu*B / sigma^2)

Parameters follow the notes' example: mu=0.5, sigma=1 (SNR = 0.5).
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

MU, SIGMA = 0.5, 1.0          # drift and noise per unit time
SNR = MU / SIGMA

def p_err(B, mu=MU, sigma=SIGMA):
    """P(hit wrong bound first) = 1/(1+e^{2 mu B / sigma^2})"""
    return 1.0 / (1.0 + np.exp(2.0 * mu * B / sigma**2))

def mean_time(B, mu=MU, sigma=SIGMA):
    """E[tau] = (B/mu) * tanh(mu*B/sigma^2)"""
    return (B / mu) * np.tanh(mu * B / sigma**2)

def Phi(z):
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

# ----------------------------------------------------------------------
# Panel (a): DDM sample paths -- a low bound decides faster but can err
# ----------------------------------------------------------------------
rng = np.random.default_rng(7)
B2, B1 = 2.0, 1.0
n_paths, n_steps = 7, 9

# Find a path that crosses -1 early (error if B=1) but eventually hits +2
while True:
    eps = rng.standard_normal((2000, n_steps))
    walk = np.cumsum(MU + SIGMA * eps, axis=1)
    hit1_first = np.where(np.min(walk[:, :2], axis=1) <= -B1)[0]
    ok = [i for i in hit1_first if walk[i, -1] >= B2]
    if ok:
        highlight = ok[0]
        break
paths = np.cumsum(MU + SIGMA * rng.standard_normal((n_paths, n_steps)), axis=1)

fig = plt.figure(figsize=(14.5, 11))
gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.2,
                      left=0.065, right=0.945, top=0.91, bottom=0.06)

axa = fig.add_subplot(gs[0, 0])
t = np.arange(n_steps + 1)
for i in range(n_paths):
    col = f"C{i}"
    axa.plot(t, np.r_[0.0, paths[i]], color=col, lw=1.6, alpha=0.9, zorder=3)
    for j in range(1, n_steps + 1):
        if abs(paths[i, j - 1]) < B2 <= abs(paths[i, j]):
            axa.plot(j, B2 if paths[i, j] > 0 else -B2, "o",
                     ms=7, color=col, mec="k", mew=0.6, zorder=4)
            break
w = np.r_[0.0, walk[highlight]]
axa.plot(t, w, color="crimson", lw=2.6, zorder=5)
j1 = int(np.where(w <= -B1)[0][0])
axa.plot(j1, -B1, "*", ms=17, color="crimson", mec="k", mew=0.8, zorder=6)
j2 = int(np.where(w >= B2)[0][0])
axa.plot(j2, B2, "*", ms=17, color="crimson", mec="k", mew=0.8, zorder=6)
axa.annotate("If B=1: hits wrong bound\nhere -> error",
             xy=(j1, -B1), xytext=(0.4, -3.15),
             fontsize=10, color="crimson",
             arrowprops=dict(arrowstyle="->", color="crimson", lw=1.2))
axa.annotate("If B=2: keeps accumulating\n-> eventually correct",
             xy=(j2, B2), xytext=(4.2, 3.35),
             fontsize=10, color="crimson",
             arrowprops=dict(arrowstyle="->", color="crimson", lw=1.2))
axa.axhline(B2, color="k", lw=2, ls="-")
axa.axhline(-B2, color="k", lw=2, ls="-")
axa.axhline(B1, color="0.45", lw=1.4, ls="--")
axa.axhline(-B1, color="0.45", lw=1.4, ls="--")
axa.text(7.3, B2 + 0.18, "bound +B = +2", fontsize=10)
axa.text(7.3, -B2 - 0.45, "bound -B = -2", fontsize=10)
axa.text(7.3, B1 + 0.18, "+-1 (low bound)", fontsize=9, color="0.35")
axa.set_xlim(0, 8.6)
axa.set_ylim(-3.9, 3.9)
axa.set_xlabel("time t (observation steps)", fontsize=11)
axa.set_ylabel(r"accumulated evidence $x(t)$", fontsize=11)
axa.set_title("(a) DDM sample paths: first hit decides ($\\bullet$ = decision)",
              fontsize=12, loc="left")
axa.grid(alpha=0.3, ls=":")

# ----------------------------------------------------------------------
# Panel (b): bound height B sets BOTH error rate (left) and time (right)
# ----------------------------------------------------------------------
axb = fig.add_subplot(gs[0, 1])
B = np.linspace(0.05, 8.0, 400)
axb.semilogy(B, p_err(B), color="C3", lw=2.5, label=r"$P(\mathrm{error})=1/(1+e^{2\mu B/\sigma^2})$")
axb.set_xlabel("bound height B", fontsize=11)
axb.set_ylabel("P(error) (log scale)", color="C3", fontsize=11)
axb.tick_params(axis="y", labelcolor="C3")
axb.axhline(0.5, color="0.6", lw=1, ls=":")
axb.text(0.3, 0.62, "B -> 0: guessing, 50% error", fontsize=9, color="0.4")
axb.set_ylim(1e-4, 1.0)
for bb in (1, 2, 4):
    axb.scatter([bb], [p_err(bb)], color="C3", zorder=5, s=28)
    axb.annotate(f"B={bb}: {p_err(bb)*100:.0f}%", xy=(bb, p_err(bb)),
                 xytext=(bb + 0.12, p_err(bb) * 1.6), fontsize=9.5, color="C3")

axb2 = axb.twinx()
axb2.plot(B, mean_time(B), color="C0", lw=2.5, ls="--",
          label=r"$E[\tau]=(B/\mu)\tanh(\mu B/\sigma^2)$")
axb2.set_ylabel("mean decision time $E[\\tau]$", color="C0", fontsize=11)
axb2.tick_params(axis="y", labelcolor="C0")
for bb in (1, 2, 4):
    axb2.scatter([bb], [mean_time(bb)], color="C0", zorder=5, s=28)
axb.annotate("higher bound:\nerror drops exponentially",
             xy=(5.2, p_err(5.2) * 2.0), fontsize=10.5, color="C3")
axb2.annotate("but mean time grows ~linearly\n(≈ B/μ for strong signals)",
              xy=(5.1, mean_time(5.1) * 1.05), fontsize=10.5, color="C0")
axb.set_title("(b) One parameter B controls both accuracy and speed",
              fontsize=12, loc="left")
axb.grid(alpha=0.3, ls=":")

# ----------------------------------------------------------------------
# Panel (c): the speed-accuracy tradeoff frontier
# ----------------------------------------------------------------------
axc = fig.add_subplot(gs[1, 0])
Bc = np.linspace(0.1, 12.0, 500)
tt, pe = mean_time(Bc), p_err(Bc)
axc.semilogy(tt, pe, color="0.25", lw=2.6, zorder=3)
axc.plot(mean_time(0.0), 0.5, "o", color="0.25", ms=6, zorder=4)
for bb, dx, dy in ((1, 0.0, -0.6), (2, 0.35, 0.3), (4, 0.35, 0.55)):
    axc.plot(mean_time(bb), p_err(bb), "o", color="C3", ms=9, zorder=5)
    axc.annotate(f"B={bb}", xy=(mean_time(bb), p_err(bb)),
                 xytext=(mean_time(bb) + dx, p_err(bb) * 10**dy),
                 fontsize=10.5, color="C3")
axc.annotate("fast & sloppy", xy=(0.75, 0.38), fontsize=12, color="C3",
             ha="center", style="italic")
axc.annotate("slow & accurate", xy=(12.6, 0.004), fontsize=12, color="C0",
             ha="center", style="italic")
axc.add_patch(FancyArrowPatch((2.2, 0.25), (6.5, 0.03), arrowstyle="->",
                              mutation_scale=18, color="0.25", lw=1.8))
axc.text(2.0, 0.09, "B up", fontsize=12, color="0.25", rotation=-38)
axc.text(9.2, 0.13, r"To be faster AND more accurate,\nmove the whole curve down:\nimprove $\mu/\sigma$ (signal quality)",
         fontsize=10, color="0.3", ha="center",
         bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", alpha=0.9))
axc.set_xlabel("mean decision time $E[\\tau]$ (left = faster)", fontsize=11)
axc.set_ylabel("error rate P(error) (log scale)", fontsize=11)
axc.set_title("(c) Tradeoff frontier: tuning B only moves along the curve",
              fontsize=12, loc="left")
axc.grid(alpha=0.3, ls=":")
axc.set_xlim(0, 14.8)
axc.set_ylim(1e-4, 0.8)

# ----------------------------------------------------------------------
# Panel (d): hypothesis testing -- the two error types (single sample)
# ----------------------------------------------------------------------
axd = fig.add_subplot(gs[1, 1])
x = np.linspace(-4, 4, 600)
g_minus = np.exp(-(x + MU) ** 2 / (2 * SIGMA**2)) / (SIGMA * math.sqrt(2 * math.pi))
g_plus = np.exp(-(x - MU) ** 2 / (2 * SIGMA**2)) / (SIGMA * math.sqrt(2 * math.pi))
axd.plot(x, g_minus, color="C0", lw=2.2, label=r"$H_-$: s=-1 (left)")
axd.plot(x, g_plus, color="C2", lw=2.2, label=r"$H_+$: s=+1 (right)")
axd.fill_between(x, g_minus, where=x > 0, color="C3", alpha=0.55)
axd.fill_between(x, g_plus, where=x < 0, color="C3", alpha=0.35)
axd.axvline(0, color="k", lw=1.6)
axd.text(0.02, 0.012, "x=0 (decision boundary)", fontsize=8, color="0.3",
         bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.15))
alpha1 = Phi(-MU / SIGMA)
p2 = p_err(2.0)
axd.annotate(f"Type I error $\\alpha$ (false alarm):\n≈ {alpha1*100:.0f}% per observation\n→ {p2*100:.0f}% at B=2",
             xy=(0.75, 0.10), xytext=(1.5, 0.35), fontsize=8.5, color="C3",
             arrowprops=dict(arrowstyle="->", color="C3", lw=1.0))
axd.annotate(f"Type II error $\\beta$ (miss):\n≈ {alpha1*100:.0f}% per observation\n→ {p2*100:.0f}% at B=2",
             xy=(-0.75, 0.10), xytext=(-3.95, 0.35), fontsize=8.5, color="C3",
             arrowprops=dict(arrowstyle="->", color="C3", lw=1.0))
axd.text(-3.95, 0.012,
         "Accumulate to ±B (DDM):\n"
         r"$P(\mathrm{corr}) = 1/(1+e^{-2\mu B/\sigma^2})$" "\n"
         r"$P(\mathrm{err})\, = 1/(1+e^{+2\mu B/\sigma^2})$" "\n"
         r"$P(\mathrm{corr})/P(\mathrm{err}) = e^{2\mu B/\sigma^2}$" "\n"
         "(B=2: 88/12; B→∞: 100/0)",
         fontsize=8.5, color="0.25",
         bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", alpha=0.9))
axd.set_xlabel("single observation m (noisy evidence)", fontsize=11)
axd.set_ylabel("probability density", fontsize=11)
axd.set_title("(d) Hypothesis testing: two error types ($H_-$ vs $H_+$)",
              fontsize=12, loc="left")
axd.legend(loc="center left", fontsize=9.5, framealpha=0.95)
axd.grid(alpha=0.3, ls=":")

fig.suptitle("Speed-accuracy tradeoff in the DDM "
             r"($\mu = 0.5,\ \sigma = 1$, SNR = 0.5)",
             fontsize=15, y=0.975)
fig.savefig("MachineLearningLectures/assets/ddm_speed_accuracy.png",
            bbox_inches="tight")
print("Saved: MachineLearningLectures/assets/ddm_speed_accuracy.png")

# ----------------------------------------------------------------------
# Monte-Carlo sanity check: fine-grained discrete walk vs continuous limit
# ----------------------------------------------------------------------
rng = np.random.default_rng(42)
B, dt, n = 2.0, 0.001, 30_000
x = np.zeros(n)
done = np.zeros(n, bool)
err = np.zeros(n, bool)
times = np.zeros(n)
t = 0.0
while not done.all() and t < 100:
    m = ~done
    x[m] += MU * dt + SIGMA * math.sqrt(dt) * rng.standard_normal(m.sum())
    t += dt
    up = (x >= B) & ~done
    dn = (x <= -B) & ~done
    times[up | dn] = t
    err[dn] = True
    done |= up | dn
print(f"Monte Carlo (dt={dt}, n={n}): P(error) = {err.mean():.4f}, "
      f"E[tau] = {times.mean():.4f}")
print(f"Analytic formulas : P(error) = {p_err(B):.4f}, "
      f"E[tau] = {mean_time(B):.4f}   (B={B}, mu={MU}, sigma={SIGMA})")
print("Note: the small gap is overshoot bias of the discrete walk; it shrinks as dt -> 0.")
