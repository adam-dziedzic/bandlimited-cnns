import matplotlib
import matplotlib.pyplot as plt
import numpy as np


labels = ['1',
          '2',
          '4',
          '8',
          '10',
          '16',
          '20',
          '30',
          '40',
          '50']
acc = [81.89,
       81.53,
       82.90,
       84.12,
       85.65,
       85.24,
       84.68,
       85.34,
       84.78,
       84.99,
       ]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x, acc)
# rects2 = ax.bar(x + width/2, women_means, width, label='Women')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Clean Accuracy')
ax.set_xlabel('Ensemble Iterations')
ax.set_title('RobustNet Ensemble (RSE)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim((80, 90))
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)

fig.tight_layout()

plt.show()