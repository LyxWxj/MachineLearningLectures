"""
可视化：复数 a 与振荡动力学 - 欧拉公式的几何解释
Visualization: Complex a and oscillation dynamics - Euler's formula geometry
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 11

# 时间轴
t = np.linspace(0, 10, 500)
omega = 2.0  # 角频率

# 纯虚数情况: a = i * omega => x(t) = exp(i * omega * t)
x = np.exp(1j * omega * t)

# 复数情况: a = sigma + i*omega => x(t) = exp((sigma + i*omega)*t)
sigma_list = [-0.2, 0, 0.2]  # 负:衰减, 零:等幅, 正:增长

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Euler's Formula: $e^{i\\omega t} = \\cos(\\omega t) + i\\sin(\\omega t)$\n"
             "and Complex $a$: $x(t) = e^{(\\sigma + i\\omega)t}$",
             fontsize=14, y=1.02)

# ─── 1. 纯虚数：复平面轨迹 ───
ax = axes[0, 0]
ax.plot(x.real, x.imag, color='#2196F3', linewidth=2, alpha=0.8)
ax.scatter(x[0].real, x[0].imag, color='red', s=80, zorder=5, label='t=0', marker='o')
ax.scatter(x[-1].real, x[-1].imag, color='darkred', s=80, zorder=5, label='t=10', marker='s')
# 单位圆
theta = np.linspace(0, 2*np.pi, 100)
ax.plot(np.cos(theta), np.sin(theta), 'gray', linewidth=1, linestyle='--', alpha=0.5)
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)
ax.set_aspect('equal')
ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
ax.set_xlabel('Real part (cos)')
ax.set_ylabel('Imaginary part (sin)')
ax.set_title('Complex Plane: $a=i\\omega$')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ─── 2. 纯虚数：实部/虚部时域 ───
ax = axes[0, 1]
ax.plot(t, x.real, color='#2196F3', linewidth=2, label='Real = cos(ωt)')
ax.plot(t, x.imag, color='#FF5722', linewidth=2, label='Imag = sin(ωt)')
ax.set_xlabel('Time t')
ax.set_ylabel('x(t)')
ax.set_title('Time Domain: Real and Imaginary Parts')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_ylim(-1.5, 1.5)

# ─── 3. 纯虚数：幅值 ───
ax = axes[0, 2]
ax.plot(t, np.abs(x), color='#4CAF50', linewidth=2)
ax.set_xlabel('Time t')
ax.set_ylabel('|x(t)|')
ax.set_title('Amplitude: $|e^{i\\omega t}| = 1$')
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1.5)

# ─── 4. 复数 a 的复平面轨迹 ───
ax = axes[1, 0]
colors = ['#F44336', '#2196F3', '#4CAF50']
labels = [f'σ={s}' for s in sigma_list]
for s, c, lab in zip(sigma_list, colors, labels):
    x_c = np.exp((s + 1j*omega) * t)
    ax.plot(x_c.real, x_c.imag, color=c, linewidth=1.5, alpha=0.8, label=lab)
    ax.scatter(x_c[0].real, x_c[0].imag, color=c, s=60, zorder=5, marker='o')
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)
ax.set_aspect('equal')
ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
ax.set_xlabel('Real'); ax.set_ylabel('Imag')
ax.set_title('Complex $a$: Spiral Trajectories')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ─── 5. 复数 a 的时域幅值 ───
ax = axes[1, 1]
for s, c, lab in zip(sigma_list, colors, labels):
    x_c = np.exp((s + 1j*omega) * t)
    ax.plot(t, x_c.real, color=c, linewidth=1.5, alpha=0.8,
            label=f'{lab} (real)')
ax.set_xlabel('Time t')
ax.set_ylabel('Re[x(t)]')
ax.set_title('Real Part: Different $\\sigma$')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ─── 6. 复数 a 的幅值 ───
ax = axes[1, 2]
for s, c, lab in zip(sigma_list, colors, labels):
    x_c = np.exp((s + 1j*omega) * t)
    ax.plot(t, np.abs(x_c), color=c, linewidth=1.5, alpha=0.8, label=lab)
ax.set_xlabel('Time t')
ax.set_ylabel('|x(t)|')
ax.set_title('Amplitude: $e^{\\sigma t}$ Envelope')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig('/media/lyxwxj/Data/common/Workspace/Slides/MachineLearningLectures/assets/complex_oscillation.png',
            dpi=150, bbox_inches='tight')
plt.close()

print("Done: complex_oscillation.png saved.")
