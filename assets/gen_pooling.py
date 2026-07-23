"""Generate pooling layer visualization for the lecture notes."""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import matplotlib
matplotlib.rcParams['font.size'] = 12

# ─── Max Pooling vs Average Pooling ───

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# Input feature map (4x4)
np.random.seed(42)
inp = np.array([
    [1, 3, 2, 1],
    [4, 6, 5, 2],
    [3, 1, 4, 3],
    [2, 5, 1, 2]
], dtype=float)

# Max pooling 2x2, stride 2
max_pool = np.array([
    [max(inp[i:i+2, j:j+2].flat) for j in range(0, 4, 2)]
    for i in range(0, 4, 2)
])

# Average pooling 2x2, stride 2
avg_pool = np.array([
    [np.mean(inp[i:i+2, j:j+2]) for j in range(0, 4, 2)]
    for i in range(0, 4, 2)
])

# Plot input
ax = axes[0]
im = ax.matshow(inp, cmap='Blues', vmin=0, vmax=6)
for i in range(4):
    for j in range(4):
        ax.text(j, i, f'{int(inp[i,j])}', ha='center', va='center', fontsize=16, fontweight='bold')
# Draw pooling windows
for wi, wj in [(0,0), (0,2), (2,0), (2,2)]:
    rect = patches.Rectangle((wj-0.5, wi-0.5), 2, 2, linewidth=2.5,
                              edgecolor='red', facecolor='red', alpha=0.08)
    ax.add_patch(rect)
ax.set_title('Input (4×4)\nwith 2×2 windows', fontsize=13)
ax.set_xticks(range(4)); ax.set_yticks(range(4))

# Plot max pooling
ax = axes[1]
im = ax.matshow(max_pool, cmap='Oranges', vmin=0, vmax=6)
for i in range(2):
    for j in range(2):
        ax.text(j, i, f'{int(max_pool[i,j])}', ha='center', va='center', fontsize=18, fontweight='bold')
        # Show which element was selected
        si, sj = i*2, j*2
        max_pos = np.unravel_index(inp[si:si+2, sj:sj+2].argmax(), (2, 2))
        rect = patches.Rectangle((sj+max_pos[1]-0.5, si+max_pos[0]-0.5), 1, 1,
                                  linewidth=2, edgecolor='red', facecolor='red', alpha=0.25)
        axes[0].add_patch(rect)
ax.set_title('Max Pooling (2×2)\nstride=2 → (2×2)', fontsize=13)
ax.set_xticks(range(2)); ax.set_yticks(range(2))

# Plot average pooling
ax = axes[2]
im = ax.matshow(avg_pool, cmap='Greens', vmin=0, vmax=6)
for i in range(2):
    for j in range(2):
        ax.text(j, i, f'{avg_pool[i,j]:.1f}', ha='center', va='center', fontsize=18, fontweight='bold')
ax.set_title('Average Pooling (2×2)\nstride=2 → (2×2)', fontsize=13)
ax.set_xticks(range(2)); ax.set_yticks(range(2))

fig.suptitle('Pooling Operations: Reduce Spatial Dimensions', fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig('/media/lyxwxj/Data/common/Workspace/Slides/MachineLearningLectures/assets/pooling_comparison.png',
            dpi=150, bbox_inches='tight')
plt.close()


# ─── Pooling with stride visualization ───

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Input 6x6
np.random.seed(7)
inp6 = np.random.randint(0, 10, (6, 6))

# Max pool 2x2, stride 2
pool6 = np.array([
    [max(inp6[i:i+2, j:j+2].flat) for j in range(0, 6, 2)]
    for i in range(0, 6, 2)
])

ax = axes[0]
ax.matshow(inp6, cmap='YlOrRd', vmin=0, vmax=10)
for i in range(6):
    for j in range(6):
        ax.text(j, i, f'{inp6[i,j]}', ha='center', va='center', fontsize=13, fontweight='bold')
# Draw pooling windows with different colors
colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336', '#00BCD4',
          '#795548', '#607D8B', '#E91E63']
idx = 0
for wi in range(0, 6, 2):
    for wj in range(0, 6, 2):
        rect = patches.Rectangle((wj-0.5, wi-0.5), 2, 2, linewidth=2.5,
                                  edgecolor=colors[idx % len(colors)],
                                  facecolor=colors[idx % len(colors)], alpha=0.12)
        ax.add_patch(rect)
        idx += 1
ax.set_title('Input (6×6) with 2×2 pool windows\nstride=2, no overlap', fontsize=13)
ax.set_xticks(range(6)); ax.set_yticks(range(6))

ax = axes[1]
ax.matshow(pool6, cmap='YlOrRd', vmin=0, vmax=10)
for i in range(3):
    for j in range(3):
        ax.text(j, i, f'{pool6[i,j]}', ha='center', va='center', fontsize=16, fontweight='bold')
ax.set_title('Output (3×3)\nspatial size halved', fontsize=13)
ax.set_xticks(range(3)); ax.set_yticks(range(3))

fig.suptitle('Max Pooling: 2×2 kernel, stride 2', fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig('/media/lyxwxj/Data/common/Workspace/Slides/MachineLearningLectures/assets/pooling_stride.png',
            dpi=150, bbox_inches='tight')
plt.close()

print("Done: pooling_comparison.png, pooling_stride.png saved.")
