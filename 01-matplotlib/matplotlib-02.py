#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 22:04:57 2021

@author: tsemach
"""
#%%
import numpy as np
import matplotlib.pyplot as plt

# Compute the x and y coordinates for points on a sine curve
x = np.arange(0, 3 * np.pi, 0.1)
y = np.sin(x)

# Plot the points using matplotlib
plt.plot(x, y)
plt.show()  # You must call plt.show() to make graphics appear.
# %%
