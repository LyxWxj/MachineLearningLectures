"""Generate learning rate scheduler plots for the lecture notes."""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 12

# Step Decay
fig, ax = plt.subplots(figsize=(6, 3))
lrs_step = []
lr = 0.1
for epoch in range(100):
    lrs_step.append(lr)
    if epoch in [30, 60]:
        lr *= 0.1

ax.plot(range(100), lrs_step, linewidth=2, color='#2196F3')
ax.set_xlabel('Epoch')
ax.set_ylabel('Learning Rate')
ax.set_title('Step Decay (gamma=0.1, milestones=[30, 60])')
ax.set_yscale('log')
ax.grid(True, alpha=0.3)
ax.annotate('lr × 0.1', xy=(30, 0.1), xytext=(35, 0.15),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=10, color='red')
ax.annotate('lr × 0.1', xy=(60, 0.01), xytext=(65, 0.015),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=10, color='red')
fig.tight_layout()
fig.savefig('/media/lyxwxj/Data/common/Workspace/Slides/MachineLearningLectures/zh/src/assets/step_decay.png', dpi=150)
plt.close()

# Cosine Annealing
import numpy as np
fig, ax = plt.subplots(figsize=(6, 3))
T_max = 100
eta_min = 1e-4
eta_max = 0.1
epochs = np.arange(T_max)
lrs_cos = eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * epochs / T_max))

ax.plot(epochs, lrs_cos, linewidth=2, color='#FF5722')
ax.set_xlabel('Epoch')
ax.set_ylabel('Learning Rate')
ax.set_title('Cosine Annealing (T_max=100)')
ax.set_yscale('log')
ax.grid(True, alpha=0.3)
ax.annotate('lr_max', xy=(0, eta_max), xytext=(10, eta_max * 1.5),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=10, color='red')
ax.annotate('lr_min', xy=(T_max, eta_min), xytext=(80, eta_min * 2),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=10, color='red')
fig.tight_layout()
fig.savefig('/media/lyxwxj/Data/common/Workspace/Slides/MachineLearningLectures/zh/src/assets/cosine_annealing.png', dpi=150)
plt.close()

print("Done: step_decay.png and cosine_annealing.png saved.")
