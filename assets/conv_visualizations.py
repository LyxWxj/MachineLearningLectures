"""Generate convolution visualizations for the lecture notes."""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from scipy import ndimage
import matplotlib
matplotlib.rcParams['font.size'] = 11

# ─── 1. Convolution operation process ───

fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))

# Input image (5x5)
np.random.seed(42)
inp = np.array([
    [1, 0, 1, 0, 1],
    [0, 1, 0, 1, 0],
    [1, 0, 1, 0, 1],
    [0, 1, 0, 1, 0],
    [1, 0, 1, 0, 1]
], dtype=float)

kernel = np.array([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]
], dtype=float)

# Plot input
ax = axes[0]
ax.matshow(inp, cmap='Blues', vmin=0, vmax=1)
for i in range(5):
    for j in range(5):
        ax.text(j, i, f'{int(inp[i,j])}', ha='center', va='center', fontsize=14, fontweight='bold')
ax.set_title('Input (5×5)', fontsize=12)
ax.set_xticks(range(5)); ax.set_yticks(range(5))
ax.tick_params(labelsize=9)

# Plot kernel
ax = axes[1]
ax.matshow(kernel, cmap='RdBu_r', vmin=-4, vmax=4)
for i in range(3):
    for j in range(3):
        ax.text(j, i, f'{int(kernel[i,j])}', ha='center', va='center', fontsize=14, fontweight='bold')
ax.set_title('Kernel (3×3)', fontsize=12)
ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.tick_params(labelsize=9)

# Plot convolution step (highlight receptive field)
ax = axes[2]
ax.matshow(inp, cmap='Blues', vmin=0, vmax=1)
rect = patches.Rectangle((-0.5, -0.5), 3, 3, linewidth=3, edgecolor='red', facecolor='red', alpha=0.15)
ax.add_patch(rect)
for i in range(5):
    for j in range(5):
        c = 'red' if i < 3 and j < 3 else 'black'
        fw = 'bold' if i < 3 and j < 3 else 'normal'
        ax.text(j, i, f'{int(inp[i,j])}', ha='center', va='center', fontsize=14, fontweight=fw, color=c)
ax.set_title('Receptive field\n(0×1+1×1+0×0+1×1+1×(-4)+0×1+0×0+1×1+0×1 = -2)', fontsize=9)
ax.set_xticks(range(5)); ax.set_yticks(range(5))
ax.tick_params(labelsize=9)

# Plot output (3x3)
out = ndimage.convolve(inp, kernel, mode='constant', cval=0.0)
ax = axes[3]
ax.matshow(out, cmap='RdBu_r', vmin=-4, vmax=4)
for i in range(3):
    for j in range(3):
        ax.text(j, i, f'{int(out[i,j])}', ha='center', va='center', fontsize=14, fontweight='bold')
ax.set_title('Output (3×3)', fontsize=12)
ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.tick_params(labelsize=9)

fig.suptitle('Convolution Operation: Input × Kernel → Output', fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig('/media/lyxwxj/Data/common/Workspace/Slides/MachineLearningLectures/zh/src/assets/conv_operation.png',
            dpi=150, bbox_inches='tight')
plt.close()


# ─── 2. Different kernels and their effects ───

# Create a synthetic test image with edges and textures
np.random.seed(0)
img = np.zeros((128, 128))
# Circles
for cx, cy, r in [(32, 32, 20), (96, 96, 25), (64, 64, 15)]:
    yy, xx = np.ogrid[:128, :128]
    mask = (xx - cx)**2 + (yy - cy)**2 < r**2
    img[mask] = 1.0
# Rectangle
img[20:50, 80:120] = 0.7
# Diagonal stripe
for i in range(128):
    for j in range(max(0, i-5), min(128, i+5)):
        img[i, j] = max(img[i, j], 0.5)
# Add some noise
img += np.random.randn(128, 128) * 0.05
img = np.clip(img, 0, 1)

# Define kernels
kernels = {
    'Identity': np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
    'Gaussian\n(blur)': np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16.0,
    'Sobel X\n(vertical edge)': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
    'Sobel Y\n(horizontal edge)': np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
    'Laplacian\n(edge detection)': np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]),
    'Sharpen': np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
}

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()

for idx, (name, k) in enumerate(kernels.items()):
    filtered = ndimage.convolve(img, k, mode='constant', cval=0.0)
    # Clip for display
    if 'edge' in name.lower() or 'Laplacian' in name:
        filtered = np.abs(filtered)
        filtered = filtered / filtered.max()
    else:
        filtered = np.clip(filtered, 0, 1)

    axes[idx].imshow(filtered, cmap='gray')
    axes[idx].set_title(name, fontsize=12, fontweight='bold')
    axes[idx].axis('off')

fig.suptitle('Effects of Different Convolution Kernels', fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig('/media/lyxwxj/Data/common/Workspace/Slides/MachineLearningLectures/zh/src/assets/conv_kernels.png',
            dpi=150, bbox_inches='tight')
plt.close()


# ─── 3. Kernel visualization (numeric values) ───

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()

for idx, (name, k) in enumerate(kernels.items()):
    ax = axes[idx]
    vmax = max(abs(k.min()), abs(k.max()), 1)
    im = ax.matshow(k, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    for i in range(k.shape[0]):
        for j in range(k.shape[1]):
            val = k[i, j]
            ax.text(j, i, f'{val:.2f}' if val != 0 else '0',
                    ha='center', va='center', fontsize=11, fontweight='bold')
    ax.set_title(name, fontsize=11, fontweight='bold')
    ax.set_xticks(range(k.shape[1]))
    ax.set_yticks(range(k.shape[0]))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.suptitle('Convolution Kernel Values', fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig('/media/lyxwxj/Data/common/Workspace/Slides/MachineLearningLectures/zh/src/assets/conv_kernel_values.png',
            dpi=150, bbox_inches='tight')
plt.close()

print("Done: conv_operation.png, conv_kernels.png, conv_kernel_values.png saved.")
