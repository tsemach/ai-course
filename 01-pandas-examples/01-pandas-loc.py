#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 19:34:27 2021

@author: tsemach

@from: https://towardsdatascience.com/how-to-use-loc-and-iloc-for-selecting-data-in-pandas-bd09cb4c3d79
"""
#%%
import os
import pandas as pd

print(os.getcwd())

#%%
print('cell: load data from data.csv')
df = pd.read_csv('data-1.csv', index_col=['Day'])
print(df)

#%%
print(f"cell: get Friday's temperature")
# get Friday's temperature
fri = df.loc['Fri', 'Temperature']
print(fri)

#%%
print('cell: get all rows of Temperature and Wind')
temperatue_rows = df.loc[:, ['Temperature', 'Wind']]
print(temperatue_rows)

#%%

