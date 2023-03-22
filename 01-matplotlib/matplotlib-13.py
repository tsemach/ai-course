# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2, 100)

plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro')
plt.axis([0, 6, 0, 20])
plt.show()