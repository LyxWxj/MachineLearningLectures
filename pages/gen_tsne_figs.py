"""Generate t-SNE and RSA visualizations for the slides."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.facecolor': '#1a1a2e',
    'axes.facecolor': '#16213e',
    'text.color': '#e0e0e0',
    'axes.labelcolor': '#e0e0e0',
    'xtick.color': '#b0b0b0',
    'ytick.color': '#b0b0b0',
})

np.random.seed(42)

# ============================================================
# Figure 1: t-SNE algorithm steps
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))

# Step 1: High-dimensional pairwise distances
ax = axes[0]
X_small = np.array([[2, 2], [2.5, 2.3], [1.8, 1.7],
                     [-2, -2], [-2.3, -1.8], [-1.7, -2.2],
                     [2, -2], [2.2, -1.8], [1.7, -2.3]])
labels_small = [0, 0, 0, 1, 1, 1, 2, 2, 2]
colors_small = ['#5dade2', '#e74c3c', '#f39c12']

D = squareform(pdist(X_small))
im = ax.imshow(D, cmap='viridis')
ax.set_title('Step 1: Pairwise distances\n(high-D space)', fontsize=11, color='#e0e0e0')
ax.set_xlabel('Point index', fontsize=9)
ax.set_ylabel('Point index', fontsize=9)
for i in range(9):
    for j in range(9):
        ax.text(j, i, f'{D[i,j]:.1f}', ha='center', va='center', fontsize=6, color='white')
fig.colorbar(im, ax=ax, shrink=0.7, label='Distance')

# Step 2: Convert to probabilities (Gaussian kernel)
ax = axes[1]
sigma = 1.0
P = np.exp(-D**2 / (2 * sigma**2))
np.fill_diagonal(P, 0)
P = P / P.sum(axis=1, keepdims=True)
im = ax.imshow(P, cmap='hot')
ax.set_title('Step 2: Gaussian probabilities $p_{ij}$\n(similarity in high-D)', fontsize=11, color='#e0e0e0')
ax.set_xlabel('Point index', fontsize=9)
ax.set_ylabel('Point index', fontsize=9)
fig.colorbar(im, ax=ax, shrink=0.7, label='Probability')

# Step 3: Gaussian vs t-distribution
ax = axes[2]
x_range = np.linspace(-6, 6, 300)
gauss = np.exp(-x_range**2 / 2) / np.sqrt(2 * np.pi)
t_dist = 1 / (np.pi * (1 + x_range**2))  # Cauchy (df=1, t-dist with 1 dof)
ax.plot(x_range, gauss, color='#5dade2', lw=2.5, label='Gaussian (high-D)')
ax.plot(x_range, t_dist, color='#e74c3c', lw=2.5, label='t-dist (low-D)')
ax.fill_between(x_range, gauss, alpha=0.15, color='#5dade2')
ax.fill_between(x_range, t_dist, alpha=0.15, color='#e74c3c')
ax.set_title('Step 3: Heavy-tailed kernel\n(solves crowding problem)', fontsize=11, color='#e0e0e0')
ax.set_xlabel('Distance in low-D space', fontsize=9)
ax.set_ylabel('Probability density', fontsize=9)
ax.legend(fontsize=9, facecolor='#16213e', edgecolor='#555', labelcolor='#e0e0e0')
ax.grid(True, alpha=0.12, color='white')
# Annotate the key difference
ax.annotate('t-dist allows\nmedium-distance\npoints to spread',
            xy=(2.5, 0.06), xytext=(3.8, 0.2),
            fontsize=8, color='#f39c12',
            arrowprops=dict(arrowstyle='->', color='#f39c12', lw=1.5),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e', edgecolor='#f39c12', alpha=0.8))

fig.suptitle('t-SNE: Algorithm Intuition', fontsize=14, color='#e0e0e0', y=1.02)
fig.tight_layout()
fig.savefig('/media/lyxwxj/Data/common/Workspace/Slides/MachineLearningLectures/public/tsne_steps.png',
            dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print('Saved tsne_steps.png')

# ============================================================
# Figure 2: Z-scored response matrix visualization
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Raw response matrix R (neurons x stimuli)
np.random.seed(7)
n_neurons, n_stimuli = 8, 6
R = np.random.poisson(lam=5, size=(n_neurons, n_stimuli)).astype(float)
# Add some structure: neurons 0-2 respond more to stimuli 0-2, etc.
R[0:3, 0:3] += 8
R[3:6, 3:6] += 6

ax = axes[0]
im = ax.imshow(R, cmap='YlOrRd', aspect='auto')
ax.set_title('Raw response matrix $R$\n(neurons × stimuli)', fontsize=11, color='#e0e0e0')
ax.set_xlabel('Stimulus', fontsize=9)
ax.set_ylabel('Neuron', fontsize=9)
ax.set_xticks(range(n_stimuli))
ax.set_yticks(range(n_neurons))
for i in range(n_neurons):
    for j in range(n_stimuli):
        val = R[i, j]
        ax.text(j, i, f'{val:.0f}', ha='center', va='center', fontsize=8,
                color='black' if val < 12 else 'white')
fig.colorbar(im, ax=ax, shrink=0.7, label='Spike count')

# Z-scored matrix Z
R_mean = R.mean(axis=1, keepdims=True)
R_std = R.std(axis=1, keepdims=True)
Z = (R - R_mean) / R_std

ax = axes[1]
im = ax.imshow(Z, cmap='RdBu_r', vmin=-2, vmax=2, aspect='auto')
ax.set_title('Z-scored matrix $Z$\n(what does z-score do?)', fontsize=11, color='#e0e0e0')
ax.set_xlabel('Stimulus', fontsize=9)
ax.set_ylabel('Neuron', fontsize=9)
ax.set_xticks(range(n_stimuli))
ax.set_yticks(range(n_neurons))
for i in range(n_neurons):
    for j in range(n_stimuli):
        val = Z[i, j]
        ax.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=7,
                color='black' if abs(val) < 1 else 'white')
fig.colorbar(im, ax=ax, shrink=0.7, label='Z-score')

# RDM = 1 - (1/N) Z Z^T
N = n_neurons
RDM = 1 - (1/N) * Z @ Z.T

ax = axes[2]
im = ax.imshow(RDM, cmap='viridis', aspect='auto')
ax.set_title('RDM $M = 1 - \\frac{1}{N}ZZ^T$\n(dissimilarity between stimuli)', fontsize=11, color='#e0e0e0')
ax.set_xlabel('Stimulus', fontsize=9)
ax.set_ylabel('Stimulus', fontsize=9)
ax.set_xticks(range(n_stimuli))
ax.set_yticks(range(n_stimuli))
for i in range(n_stimuli):
    for j in range(n_stimuli):
        ax.text(j, i, f'{RDM[i,j]:.2f}', ha='center', va='center', fontsize=8, color='white')
fig.colorbar(im, ax=ax, shrink=0.7, label='Dissimilarity')

fig.suptitle('RSA Pipeline: Raw Response → Z-score → RDM', fontsize=14, color='#e0e0e0', y=1.02)
fig.tight_layout()
fig.savefig('/media/lyxwxj/Data/common/Workspace/Slides/MachineLearningLectures/public/rsa_zscore.png',
            dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print('Saved rsa_zscore.png')

# ============================================================
# Figure 3: Z-score intuition
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(9, 4))

# Before z-score: raw spike counts (different baseline rates)
np.random.seed(12)
n_neurons_z = 4
n_stim_z = 8
R_raw = np.zeros((n_neurons_z, n_stim_z))
R_raw[0] = [12, 15, 10, 13, 11, 14, 9, 16]   # high-rate neuron
R_raw[1] = [2, 3, 1, 2, 1, 3, 2, 1]           # low-rate neuron
R_raw[2] = [8, 5, 12, 6, 10, 4, 11, 7]        # medium, variable
R_raw[3] = [5, 5, 5, 5, 5, 5, 5, 5]           # constant (no info)

ax = axes[0]
for i in range(n_neurons_z):
    ax.plot(range(n_stim_z), R_raw[i], 'o-', lw=2, markersize=6, label=f'Neuron {i+1}')
ax.set_title('Before Z-score: Raw spike counts', fontsize=12, color='#e0e0e0')
ax.set_xlabel('Stimulus', fontsize=10)
ax.set_ylabel('Spike count', fontsize=10)
ax.legend(fontsize=8, facecolor='#16213e', edgecolor='#555', labelcolor='#e0e0e0')
ax.grid(True, alpha=0.12, color='white')

# After z-score
R_z_mean = R_raw.mean(axis=1, keepdims=True)
R_z_std = R_raw.std(axis=1, keepdims=True)
R_z = (R_raw - R_z_mean) / R_z_std

ax = axes[1]
for i in range(n_neurons_z):
    ax.plot(range(n_stim_z), R_z[i], 'o-', lw=2, markersize=6, label=f'Neuron {i+1}')
ax.set_title('After Z-score: Centered & scaled', fontsize=12, color='#e0e0e0')
ax.set_xlabel('Stimulus', fontsize=10)
ax.set_ylabel('Z-score', fontsize=10)
ax.legend(fontsize=8, facecolor='#16213e', edgecolor='#555', labelcolor='#e0e0e0')
ax.grid(True, alpha=0.12, color='white')
ax.axhline(0, color='white', ls='--', lw=0.8, alpha=0.3)

fig.suptitle('Why Z-score? Removes baseline rate differences between neurons', fontsize=13, color='#e0e0e0', y=1.02)
fig.tight_layout()
fig.savefig('/media/lyxwxj/Data/common/Workspace/Slides/MachineLearningLectures/public/zscore_intuition.png',
            dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print('Saved zscore_intuition.png')

print('All figures generated successfully!')
