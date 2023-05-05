#%%
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt  # To visualize

#%%
def polynom(Xnp):
  # return np.sin(Xnp * np.pi / 180.) - np.power(Xnp, 3)
  return np.sin(Xnp * np.pi / 180.)

def normalize(data: torch.Tensor):
  s = data.max() - data.min()
  return (data - data.mean()) / s

#%%
def plotXY(x, y, title='y=sin(X) - power(X)/10'):
  plt.plot(Xtrain, Ytrain, color='red')
  plt.title(title, color='blue')
  plt.show()

#%%
XtrainNP = np.arange(-180, 180, 0.01, dtype=np.float64)
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
class LR(nn.Module):

  def __init__(self, h_layer1_size, h_layer2_size):
    super().__init__()

    self.linear1 = nn.Linear(1, h_layer1_size) 
    self.linear2 = nn.Linear(h_layer1_size, h_layer2_size) 
    self.linear3 = nn.Linear(h_layer2_size, 1) 

  def forward(self, x):
    x = self.linear1(x)
    x = F.relu(x)
    x = self.linear2(x)
    x = F.relu(x)
    y = self.linear3(x)

    return y

#%%
torch.manual_seed(1)
model = LR(50, 100)

criterion = nn.MSELoss()  
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

#%%
print(model(torch.tensor([2.])))

#%%
epochs = 1
losses = []

for i in range(epochs):  

  for batch in train_dl:
    x_train = batch[:, 0]
    y_train = batch[:, 1]

    y_pred = model(x_train.reshape(100, -1).float())
    loss = criterion(y_pred.reshape(-1), y_train.float())
    print("epoch:", i, "loss:", loss.item())
    
    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#%%
plt.plot(range(0, len(losses)), losses)
plt.ylabel('Loss')
plt.xlabel('epoch')

#%%
