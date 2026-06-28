"""Generate covariance matrix visualizations for the slides."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch

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
# Figure 1: Four scatter plots showing different covariances
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(8, 7))
fig.suptitle('Covariance: How Two Variables Move Together', fontsize=14, color='#e0e0e0', y=0.97)

cases = [
    (0.0, 1.0, 1.0, 'Zero Covariance\n(Independent)'),
    (0.85, 1.0, 1.0, 'Positive Covariance\n(rise together)'),
    (-0.85, 1.0, 1.0, 'Negative Covariance\n(one rises, one falls)'),
    (0.0, 1.0, 2.5, 'Zero Covariance\n(different variances)'),
]

for ax, (rho, sig_x, sig_y, title) in zip(axes.flat, cases):
    cov_xy = rho * sig_x * sig_y
    cov = [[sig_x**2, cov_xy], [cov_xy, sig_y**2]]
    data = np.random.multivariate_normal([0, 0], cov, 200)

    ax.scatter(data[:, 0], data[:, 1], alpha=0.5, s=18, c='#5dade2', edgecolors='none')

    # Draw mean cross
    ax.axhline(0, color='#e74c3c', lw=0.8, ls='--', alpha=0.5)
    ax.axvline(0, color='#e74c3c', lw=0.8, ls='--', alpha=0.5)

    # Draw principal axes (eigenvectors scaled by sqrt of eigenvalues)
    vals, vecs = np.linalg.eigh(cov)
    for val, vec in zip(vals, vecs.T):
        arrow_len = 2 * np.sqrt(val)
        ax.annotate('', xy=(arrow_len * vec[0], arrow_len * vec[1]),
                     xytext=(0, 0),
                     arrowprops=dict(arrowstyle='->', color='#f39c12', lw=2))

    ax.set_title(title, fontsize=11, color='#e0e0e0')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.15, color='white')

    # Show covariance value
    ax.text(0.05, 0.95, f'cov = {cov_xy:+.2f}', transform=ax.transAxes,
            fontsize=9, va='top', color='#f39c12',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e', edgecolor='#f39c12', alpha=0.8))

fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig('/media/lyxwxj/Data/common/Workspace/Slides/MachineLearningLectures/public/cov_scatter.png',
            dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print('Saved cov_scatter.png')

# ============================================================
# Figure 2: Covariance ellipse visualization
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
fig.suptitle('Covariance Matrix Defines a Confidence Ellipse', fontsize=14, color='#e0e0e0', y=1.02)

ellipse_cases = [
    (0.9, 1.0, 1.0, 'Σ = [[1.0, 0.9],\n      [0.9, 1.0]]'),
    (0.0, 1.0, 2.0, 'Σ = [[1.0, 0.0],\n      [0.0, 2.0]]'),
    (-0.7, 1.5, 0.8, 'Σ = [[1.5, -0.7],\n      [-0.7, 0.8]]'),
]

for ax, (rho, sig_x, sig_y, title) in zip(axes, ellipse_cases):
    cov_xy = rho * sig_x * sig_y
    cov = [[sig_x**2, cov_xy], [cov_xy, sig_y**2]]
    data = np.random.multivariate_normal([0, 0], cov, 300)

    ax.scatter(data[:, 0], data[:, 1], alpha=0.35, s=14, c='#5dade2', edgecolors='none')

    # Draw 1-sigma and 2-sigma ellipses
    vals, vecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))

    for n_std, alpha_val in [(1, 0.6), (2, 0.3)]:
        width = 2 * n_std * np.sqrt(vals[1])
        height = 2 * n_std * np.sqrt(vals[0])
        ellipse = patches.Ellipse((0, 0), width, height, angle=angle,
                                   fill=False, edgecolor='#f39c12', lw=2, alpha=alpha_val)
        ax.add_patch(ellipse)

    # Draw eigenvectors
    for val, vec in zip(vals, vecs.T):
        arrow_len = 2 * np.sqrt(val)
        ax.annotate('', xy=(arrow_len * vec[0], arrow_len * vec[1]),
                     xytext=(0, 0),
                     arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2.5))

    ax.set_title(title, fontsize=10, color='#e0e0e0', fontfamily='monospace')
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-4.5, 4.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.12, color='white')

fig.tight_layout()
fig.savefig('/media/lyxwxj/Data/common/Workspace/Slides/MachineLearningLectures/public/cov_ellipse.png',
            dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print('Saved cov_ellipse.png')

# ============================================================
# Figure 3: Covariance vs Correlation
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
fig.suptitle('Covariance vs Correlation', fontsize=14, color='#e0e0e0', y=1.02)

# Left: same correlation, different scales
np.random.seed(7)
x = np.random.randn(200)
y = 0.8 * x + 0.6 * np.random.randn(200)

ax = axes[0]
ax.scatter(x, y, alpha=0.45, s=18, c='#5dade2', edgecolors='none', label='Original')
ax.scatter(x * 3, y, alpha=0.35, s=18, c='#e74c3c', edgecolors='none', label='x scaled by 3')
ax.set_title('Same Correlation, Different Covariance', fontsize=11, color='#e0e0e0')
ax.legend(fontsize=9, facecolor='#16213e', edgecolor='#555', labelcolor='#e0e0e0')
ax.grid(True, alpha=0.12, color='white')

cov_orig = np.cov(x, y)[0, 1]
cov_scaled = np.cov(x * 3, y)[0, 1]
corr_orig = np.corrcoef(x, y)[0, 1]
ax.text(0.05, 0.95, f'Original:   cov={cov_orig:.2f}, r={corr_orig:.2f}\nScaled x:  cov={cov_scaled:.2f}, r={corr_orig:.2f}',
        transform=ax.transAxes, fontsize=8.5, va='top', color='#f39c12', fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e', edgecolor='#f39c12', alpha=0.85))

# Right: correlation coefficient interpretation
ax = axes[1]
corrs = [0.0, 0.5, 0.9, -0.9]
colors_list = ['#5dade2', '#48c9b0', '#f39c12', '#e74c3c']
for rho, clr in zip(corrs, colors_list):
    cov = [[1, rho], [rho, 1]]
    d = np.random.multivariate_normal([0, 0], cov, 150)
    ax.scatter(d[:, 0], d[:, 1], alpha=0.4, s=14, c=clr, edgecolors='none', label=f'r = {rho:+.1f}')

ax.set_title('Correlation r ∈ [-1, 1]', fontsize=11, color='#e0e0e0')
ax.legend(fontsize=9, facecolor='#16213e', edgecolor='#555', labelcolor='#e0e0e0', loc='lower right')
ax.set_xlim(-3.5, 3.5)
ax.set_ylim(-3.5, 3.5)
ax.set_aspect('equal')
ax.grid(True, alpha=0.12, color='white')

fig.tight_layout()
fig.savefig('/media/lyxwxj/Data/common/Workspace/Slides/MachineLearningLectures/public/cov_vs_corr.png',
            dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print('Saved cov_vs_corr.png')

# ============================================================
# Figure 4: Covariance matrix as heatmap
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
fig.suptitle('Covariance Matrix Heatmaps', fontsize=14, color='#e0e0e0', y=1.02)

heatmap_cases = [
    ('Positive correlation', [[1.0, 0.8, 0.3],
                               [0.8, 1.0, 0.5],
                               [0.3, 0.5, 1.0]]),
    ('Negative correlation', [[1.0, -0.7, 0.1],
                               [-0.7, 1.0, -0.4],
                               [0.1, -0.4, 1.0]]),
    ('Independent', [[1.0, 0.0, 0.0],
                      [0.0, 2.0, 0.0],
                      [0.0, 0.0, 0.5]]),
]

for ax, (title, mat) in zip(axes, heatmap_cases):
    im = ax.imshow(mat, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{mat[i][j]:.1f}', ha='center', va='center',
                    fontsize=13, fontweight='bold', color='black' if abs(mat[i][j]) < 0.5 else 'white')
    ax.set_title(title, fontsize=11, color='#e0e0e0')
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(['$x_1$', '$x_2$', '$x_3$'], fontsize=11)
    ax.set_yticklabels(['$x_1$', '$x_2$', '$x_3$'], fontsize=11)

cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
cbar.ax.tick_params(colors='#b0b0b0')

fig.tight_layout()
fig.savefig('/media/lyxwxj/Data/common/Workspace/Slides/MachineLearningLectures/public/cov_heatmap.png',
            dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print('Saved cov_heatmap.png')

print('All figures generated successfully!')
