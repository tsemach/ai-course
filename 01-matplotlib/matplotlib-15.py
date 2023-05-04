#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 21:50:25 2021

@author: tsemach
"""
#%%
import os
import numpy as np
import matplotlib.pyplot as plt  # To visualize
import pandas as pd  # To read data

#%%
X = np.arange(-100, 100, 0.01, dtype=np.float64)
Y = np.sin(X * np.pi / 180.) - np.power(X, 3)
# Y = np.sin(X * np.pi / 180.) - np.power(X, 3) + 3 * np.power(X, 2) 
# Y = np.power(X, 3) + 3*np.power(X, 2) + 1

plt.plot(X, Y, color='red')
plt.title('Y=sin(X) - power(X)/10', color='blue')
plt.show()

# %%
# Xlog = np.arange(0, 1, 0.001, dtype=np.float64)
# Y1 = -np.log(Xlog)
# Y2 = -np.log(1-Xlog)

# plt.plot(Xlog, Y1, color='magenta')
# plt.title('Y=-log(X)', color='blue')
# plt.show()
# plt.plot(Xlog, Y2, color='orange')
# plt.title('Y=-log(1-X)', color='blue')
# plt.show()

# # %%

# %%
