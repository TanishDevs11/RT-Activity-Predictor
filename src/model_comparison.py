import matplotlib.pyplot as plt
import numpy as np

# Data for comparison
labels = ['Accuracy', 'Precision (Active)', 'Recall (Active)']
baseline_scores = [0.757, 0.65, 0.81] # Unga old model scores (approx)
optimized_scores = [0.808, 0.73, 0.85] # Optuna optimized scores

x = np.arange(len(labels))  # label locations
width = 0.35  # width of the bars

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, baseline_scores, width, label='Baseline (75.7%)', color='skyblue')
rects2 = ax.bar(x + width/2, optimized_scores, width, label='Optimized (80.7%)', color='teal')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Model Performance Comparison: Baseline vs Optimized')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 1.0)
ax.legend()

# Bar mela numbers kaata
ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

fig.tight_layout()
plt.savefig('results/model_comparison_graph.png')
print("Comparison graph saved: results/model_comparison_graph.png")
plt.show()