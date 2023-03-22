#%%
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np

#%%
torch.manual_seed(2)

# create dummy data for training
x_values = [i for i in range(11)]
x_train = np.array(x_values, dtype=np.float32)
x_train = x_train.reshape(-1, 1)

y_values = [2*i + 1 for i in x_values]
y_train = np.array(y_values, dtype=np.float32)
y_train = y_train.reshape(-1, 1)

x_train = (torch.randn(100, 1) * 10)
y_train = x_train + 3 * torch.randn(100, 1)

x_train = x_train.numpy()
y_train = y_train.numpy()

plt.plot(x_train, y_train, 'o')             # plot the new X,Y 
plt.xlabel('X')
plt.ylabel('Y')

#%%
class LR(torch.nn.Module):
  def __init__(self, inputSize, outputSize):
    super().__init__()
    self.linear = torch.nn.Linear(inputSize, outputSize)

  def forward(self, x):
    out = self.linear(x)

    return out

  def get_parameters(self):
    (w, b) = self.parameters()

    return (w[0][0].item(), b[0].item()) 

#%%
i_dim = 1        # takes variable 'x' 
o_dim = 1        # takes variable 'y'
learningRate = 0.01 
epochs = 100

model = LR(i_dim, o_dim)    
# if torch.cuda.is_available():
#     model.cuda()

#%%
def plot_fit(model: LR, title: str):
  plt.title(title)

  w, b = model.get_parameters()
  print(w, b)
  x1 = np.array([-30, 30])

  y1 = w * x1 + b

  plt.plot(x1, y1, 'r')
  plt.scatter(x_train, y_train)
  plt.xlabel('X')
  plt.ylabel('Y')
  plt.show()

#%%
# %% [markdown]
### Plot the fitting
plot_fit(model, 'Initial Model')

#%%
criterion = torch.nn.MSELoss() 
optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

#%%
for epoch in range(epochs):  
  inputs = Variable(torch.from_numpy(x_train))
  labels = Variable(torch.from_numpy(y_train))

  # clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
  optimizer.zero_grad()

  # get output from the model, given the inputs
  # outputs = model(inputs)
  outputs = model.forward(inputs)

  # get loss for the predicted output
  loss = criterion(outputs, labels)

  # get gradients w.r.t to parameters
  loss.backward()

  # update parameters
  optimizer.step()

  print('epoch {}, loss {}'.format(epoch, loss.item()))

#%%
with torch.no_grad(): # we don't need gradients in the testing phase
  if torch.cuda.is_available():
    predicted = model(Variable(torch.from_numpy(x_train).cuda())).cpu().data.numpy()
  else:
    predicted = model(Variable(torch.from_numpy(x_train))).data.numpy()

plt.plot(x_train, y_train, 'go', label='True data', alpha=0.5)
plt.plot(x_train, predicted, '--', label='Predictions', alpha=0.5)
plt.legend(loc='best')
plt.show()

#%%
import matplotlib.pyplot as plt

plt.plot(x_train, y_train, 'go', label='True data', alpha=0.5)
plt.plot(x_train, predicted, '--', label='Predictions', alpha=0.5)

# plt.plot(range(epochs), losses)
plt.ylabel('Loss')
plt.xlabel('epoch')

#%%

