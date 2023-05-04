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
from sklearn.linear_model import LinearRegression

#%%

#%%
Xexp = np.arange(0, 1, 0.01, dtype=np.float64)
Y1 = -np.exp(Xexp)
Y2 = -np.exp(1-Xexp)

plt.plot(Xexp, Y1, color='red')
plt.title('Y=-ln(X)', color='blue')
plt.show()
plt.plot(Xexp, Y2, color='blue')
plt.title('Y=1-ln(X)', color='blue')
plt.show()

# %%
Xlog = np.arange(0, 1, 0.001, dtype=np.float64)
Y1 = -np.log(Xlog)
Y2 = -np.log(1-Xlog)

plt.plot(Xlog, Y1, color='magenta')
plt.title('Y=-log(X)', color='blue')
plt.show()
plt.plot(Xlog, Y2, color='orange')
plt.title('Y=-log(1-X)', color='blue')
plt.show()

# %%
