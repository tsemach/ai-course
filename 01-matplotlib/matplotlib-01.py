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
print(os.getcwd())
data = pd.read_csv('/home/tsemach/projects/machine-leaning/ai-course/tsemach-1/matplotlib/data.csv')  # load data set

X = data.iloc[:, 0].values.reshape(-1, 1)  # values converts it into a numpy array
Y = data.iloc[:, 1].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column

linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(X, Y)  # perform linear regression
Y_pred = linear_regressor.predict(X)  # make predictions
print(Y_pred)

#%%
plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()


# %%
