#%%
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
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
XtrainNP = np.arange(-90, 90, 0.01, dtype=np.float64)
YtrainNP = polynom(XtrainNP)

Xtrain = normalize(torch.from_numpy(XtrainNP))
Ytrain = torch.from_numpy(YtrainNP)

XvalidNP = np.random.rand(6000) * 180 - 90
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

  def __init__(self, layer_size):
    super().__init__()

    self.linear1 = nn.Linear(1, layer_size)
    self.linear2 = nn.Linear(layer_size, layer_size)
    self.linear3 = nn.Linear(layer_size, layer_size)
    self.linear4 = nn.Linear(layer_size, 1) 

  def forward(self, x):
    x = self.linear1(x)
    x = F.relu(x)
    x = self.linear2(x)
    x = F.relu(x)
    x = self.linear3(x)
    x = F.relu(x)
    y = self.linear4(x)

    return y

#%%
layer_size = 10000
torch.manual_seed(1)
model = LR(layer_size)
criterion = nn.MSELoss()  
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01) 

#%%
epochs = 1
t_losses = []
v_loeses = []

for i in range(epochs):  

  for batch in train_dl:
    x_train = batch[:, 0]
    y_train = batch[:, 1]

    y_pred = model(x_train.reshape(batch_size, -1).float())
    loss = criterion(y_pred.reshape(-1), y_train.float())
    print("epoch:", i, "train loss:", loss.item())
    
    t_losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  # else:
  #   with torch.no_grad():
  #     for batch in valid_dl:
  #       x_valid = batch[:, 0]
  #       y_valid = batch[:, 1]
        
  #       v_outs = model(x_valid.reshape(batch_size*2, -1).float())
  #       v_loss = criterion(v_outs.reshape(-1), y_valid.float())
  #       v_loeses.append(v_loss.item())
  #       print("epoch:", i, "validation loss:", loss.item())

    
#%%
plt.plot(range(0, len(t_losses)), t_losses)
plt.ylabel('Train Loss')
plt.xlabel('Train Epoch')

#%%
x_value = np.arange(-90, 90, 0.01, dtype=np.float64)
y_value = polynom(x_value)

with torch.no_grad():
  x_pred = normalize(torch.from_numpy(x_value))

  p_value = model(x_pred.reshape(len(x_pred), -1).float())
  p_value = p_value.view(-1).numpy()

print('prediction loss:', np.sum(np.abs(y_value - p_value)))

#%%      
plt.plot(x_value, y_value, label='polynom')
plt.xlabel('polynom =')
plt.plot(x_value, p_value, label='model')
plt.legend(loc="upper left")
plt.show()

#%%
