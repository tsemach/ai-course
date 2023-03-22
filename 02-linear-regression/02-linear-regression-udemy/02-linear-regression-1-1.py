# %%
import torch
import matplotlib as plt

# %% [markdown]
# We will define `w` and `b` as the parameter of the linear line as `y = w * x + b`

# %%
w = torch.tensor(3.0, requires_grad=True) # initial with random number
b = torch.tensor(1.0, requires_grad=True) # initial with random number

# %% [markdown]
# ### Define our perdiction model

# %%
def forward(x):
  y = w * x  + b
  
  return y

# %% [markdown]
# Make a prediction

# %%
x = torch.tensor(2.) # use x = 2.0, the expect value of y is 7 as 3 * 2 + 1 = 7 (w * x + b)
forward(x)

# %%
x = torch.tensor([[4], [7], [3]])
forward(x)

# %% [markdown]
# ### Use PyTorch `Linear` Class

# %%
from torch.nn import Linear

torch.manual_seed(1)

model = Linear(in_features=1, out_features=1) # in features are the weight or also called parameters
print(model.bias)
print(model.weight)

# %% [markdown]
# Make a prediction

# %%
x = torch.tensor([[2.], [3.3]]) # make a prediction for two numbers
y = model(x)
print(y)

# %% [markdown]
# Use custom module of Linear class

# %%
import torch.nn as nn

class LR(nn.Module):
  def __init__(self, in_size, out_size):
    super().__init__()

    self.linear = nn.Linear(in_size, out_size) 

  def forward(self, x):
    y = self.linear(x)

    return y

# %% [markdown]
# Check the model and print out the parameters

# %%
torch.manual_seed(1)
model = LR(1, 1)
print(list(model.parameters()))

# %% [markdown]
# Make a prediction

# %%
x = torch.tensor([[1.], [2.]])
print(model.forward(x)) 

# %% [markdown]
# ### Create the training set

# %%
import matplotlib.pyplot as plt

X = torch.randn(100, 1) * 10 # create vector of dim(100, 1) which is normal distrubute
plt.plot(X, X, 'o')             # plot X,Y use 'o' point for each coordinate

Y = X + 3 * torch.randn(100, 1) # create Y with little nois
plt.plot(X, Y, 'o')             # plot the new X,Y 
plt.xlabel('X')
plt.ylabel('Y')

# %% [markdown]
# Draw the line of the prediction

# %%
import torch.nn as nn

class LR(nn.Module):
  def __init__(self, in_size, out_size):
    super().__init__()

    self.linear = nn.Linear(in_size, out_size) 

  def forward(self, x):
    y = self.linear(x)

    return y

  def get_parameters(self):
    (w, b) = self.parameters()

    return (w[0][0].item(), b[0].item())  

# %%
torch.manual_seed(1)
model = LR(1, 1)
print(list(model.get_parameters()))

# %%
import numpy as np

def plot_fit(model: LR, title: str):
  plt.title(title)

  w, b = model.get_parameters()
  print(w, b)
  x1 = np.array([-30, 30])

  y1 = w * x1 + b

  plt.plot(x1, y1, 'r')
  plt.scatter(X, Y)
  plt.show()

# %%
plot_fit(model, 'Initial Model')

# %% [markdown]
# ### Training the Model

# %%
criterion = nn.MSELoss()  # define the lost fucntion
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01) # define the optimizer

# %%
epochs = 100
losses = []
y_train = Y

for i in range(epochs):
  y_pred = model.forward(X)
  loss = criterion(y_pred, y_train)
  print("epoch:", i, "loss:", loss.item())
  
  losses.append(loss.item())
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

# %%
plt.plot(range(epochs), losses)
plt.ylabel('Loss')
plt.xlabel('epoch')

# %%



