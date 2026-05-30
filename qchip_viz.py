import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Data
models = ['FP32', 'INT8']
latency = [37.2, 93.3]
size = [9.8, 3.0]
colors = ["#791AA5", "#103099"]

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
fig.patch.set_facecolor('#0F1117')

for ax in axes:
    ax.set_facecolor('#0F1117')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('#444')
    ax.spines['left'].set_color('#444')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Chart 1: Latency
bars1 = axes[0].bar(models, latency, color=colors, width=0.4, zorder=3)
axes[0].set_title('Inference Latency (ms)\nLower is Better',
                  color='white', fontsize=13, pad=12)
axes[0].set_ylabel('milliseconds', color='#AAAAAA', fontsize=10)
axes[0].yaxis.grid(True, color='#222', zorder=0)
axes[0].set_ylim(0, max(latency) * 1.3)

for bar, val in zip(bars1, latency):
    axes[0].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 2,
                 f'{val} ms', ha='center', va='bottom',
                 color='white', fontsize=12, fontweight='bold')

# Annotation explaining the result
axes[0].text(0.5, 0.92,
             'INT8 slower: no VNNI on AMD Ryzen',
             transform=axes[0].transAxes,
             ha='center', color="#4C9BE8",
             fontsize=8, style='italic')

# Chart 2: Model Size
bars2 = axes[1].bar(models, size, color=colors, width=0.4, zorder=3)
axes[1].set_title('Model Size (MB)\nLower is Better',
                  color='white', fontsize=13, pad=12)
axes[1].set_ylabel('megabytes', color="#AAAAAAD8", fontsize=10)
axes[1].yaxis.grid(True, color='#222', zorder=0)
axes[1].set_ylim(0, max(size) * 1.3)

for bar, val in zip(bars2, size):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.1,
                 f'{val} MB', ha='center', va='bottom',
                 color='white', fontsize=12, fontweight='bold')

axes[1].text(0.5, 0.92,
             '69.5% smaller',
             transform=axes[1].transAxes,
             ha='center', color="#4C9BE8",
             fontsize=8, style='italic')

# Footer 
fig.text(0.5, 0.01,
         'Q-CHIP | Quantized Computer-vision Hardware Inspection Pipeline | AMD Ryzen 7 7435HS',
         ha='center', color='#666666', fontsize=8)

plt.suptitle('Q-CHIP Benchmark: FP32 vs INT8 on CPU',
             color='white', fontsize=15, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('qchip_benchmark.png', dpi=150,
            bbox_inches='tight', facecolor='#0F1117')
print("Saved: qchip_benchmark.png")