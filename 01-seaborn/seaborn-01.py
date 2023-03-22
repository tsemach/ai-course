#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 22:11:00 2021

@author: tsemach
"""
#%%
import numpy as np
import seaborn as sns

# sns.set()

# Compute the x and y coordinates for points on a sine curve
x = np.arange(0, 3 * np.pi, 0.1)
y = np.sin(x)

sns.pairplot(x,y)

# %%
