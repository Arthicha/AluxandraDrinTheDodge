import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

uniform_data = np.random.rand(10, 12)
ax = sns.heatmap(uniform_data)
plt.show()
