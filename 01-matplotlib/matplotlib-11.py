# -*- coding: utf-8 -*-
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
a = pd.DataFrame(np.random.rand(4, 5), columns = list('abcde'))
a_asarray = a.values

print(a_asarray[1:2])
plt.plot(np.arange(0,1,1/a_asarray[1:2][0].size), a_asarray[1:2][0])
plt.show()
# %%
