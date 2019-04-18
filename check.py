import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt('data')
means = np.loadtxt('means')

plt.scatter(data[:, 0], data[:, 1])
plt.scatter(means[:, 0], means[:, 1], linewidths=2)
plt.show()
