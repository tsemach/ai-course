#%%
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt  # To visualize

#%%
def polynom(Xnp):
  return np.sin(Xnp * np.pi / 180.) - np.power(Xnp, 3)

#%%
def normalize(data: torch.Tensor):
  s = data.max() - data.min()
  return (data - data.mean()) / s

#%%
def plotXY(x, y, title='y=sin(X) - power(X)/10'):
  plt.plot(Xtrain, Ytrain, color='red')
  plt.title(title, color='blue')
  plt.show()

#%%
XtrainNP = np.arange(-100, 100, 0.01, dtype=np.float64)
YtrainNP = polynom(XtrainNP)

Xtrain = normalize(torch.from_numpy(XtrainNP))
Ytrain = torch.from_numpy(YtrainNP)

XvalidNP = np.random.rand(6000) * 200 - 100
YvalidNP = polynom(XvalidNP)

Xvalid = normalize(torch.from_numpy(XvalidNP))
Yvalid = torch.from_numpy(YvalidNP)

train_ds = torch.stack((Xtrain, Ytrain), dim=1) 
valid_ds = torch.stack((Xvalid, Yvalid), dim=1)

#%%
batch_size = 100
train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=3, pin_memory=True)
valid_dl = DataLoader(valid_ds, batch_size*2, num_workers=3, pin_memory=True)

# %%
for batch in train_dl:
  print(batch[:, 0].size(), batch[:, 1].size())
  plotXY(batch[:, 0], batch[:, 1])
  break

#%%