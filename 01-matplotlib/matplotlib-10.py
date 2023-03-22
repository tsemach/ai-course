# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()  # Create a figure containing a single axes.
ax.plot([1, 2, 3, 4], [1, 4, 2, 3])  # Plot some data on the axes.
print(type(fig), type(ax))

# the two plot are overwrite one on the other
plt.plot([1, 2, 3, 4], [10, 40, 2, 3])  # Matplotlib plot.

