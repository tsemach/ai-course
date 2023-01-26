
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: pytorch
#     language: python
#     name: python3
# ---

# ## Tensors Operations
#
# Tensors are the building block of all PyTotch operations.\
# At its core, PyTorch is a library for processing tensors. A tensor is a number, vector, matrix, or any n-dimensional array. Let's create a tensor with a single number.\
# \
# `Note`: all elements of a tensors require same data type.
#

#%%
import torch

a = torch.tensor(4.)
print(a.shape)  # note: a is a scalar, there is no dimension to scalars.
a.dtype

#%%
v = torch.tensor([4., 5, 6, 7]) # all of the tensors are set to the same data type
print(v.dtype)
v

v = torch.tensor([1, 2, 3, 4, 5, 6]) 
print(v[2:5]) # print items in indexies 2,3,4

v = torch.tensor([1, 2, 3, 4, 5, 6]) 
print(v[2:]) # print from index 2 until the end

v = torch.arange(2, 7) # create a vector of 2,3 .. 6 => tensor([2, 3, 4, 5, 6])
v = torch.arange(2, 7, 2) # create a vector => tensor([2, 4, 6])
v

x = torch.arange(18).view(3, 2, 3) # create a 3D matrix
print(x[1, 0:1, 1:2]) # slice from matrix 1, row 0:1, column 1:2 (excluded)
print(x[1, :, :]) # slice from matrix 1, all, all column 1:2 (excluded)
x

v = torch.FloatTensor([1, 2, 3, 4, 5, 6]) # create a float tensor
print(v.dtype)

v = torch.FloatTensor([1, 2, 3, 4, 5, 6]) # create a float tensor
print(v.view(3, 2)) # rearange to a (3, 2) matrix

v = torch.FloatTensor([1, 2, 3, 4, 5, 6]) # create a float tensor
print(v.view(3, -1)) # -1 => infer the number of columns, rearange to a (3, 2) matrix, 

# Vector Operations

# +
v1 = torch.tensor([1, 2, 3])  # create a vector of [1, 2, 3]
v2 = torch.tensor([1, 2, 3]) # create a vector of [1, 2, 3]

v1 * v2   # element wise multiplication => tensor([1, 4, 9])
v1 * 5    # multiply by scalar => tensor([ 5, 10, 15])
torch.dot(v1, v2)         # dot product, v1[0] * v2[0] + v1[1] * v2[1] ... + v1[n] * v2[n] => 14
torch.linspace(0, 10, 5)  # dive the spcae of 0:10 to 5 piceses => tensor([ 0.0000,  2.5000,  5.0000,  7.5000, 10.0000])

# +
x = torch.linspace(0, 10, 100)  # dive the spcae of 0:10 to 100 piceses
y = torch.exp(x)                # y = exponent of x

import matplotlib.pyplot as plt

plt.plot(x.numpy(), y.numpy())

# +
import matplotlib.pyplot as plt

# x = torch.arange(0, 1000., 1)
x = torch.linspace(0, 10, 100)
y1 = torch.exp(x)
y2 = torch.exp(torch.max(x) - x)

figure, axis = plt.subplots(1, 2)

axis[0].plot(x, y1)
axis[0].set_title("exp(x)")
  
# For Cosine Function
axis[1].plot(x, y2)
axis[1].set_title("max(x) = x")
    
# Combine all the operations and display
plt.show()

# -

torch.arange(18)
ms = torch.arange(18).view(3, 3, 2)
print(ms)
print('max:', torch.max(ms))
print(ms[1, 1:2, 0:2])

# +
x = torch.linspace(0, 10, 100)

y1 = torch.exp(x)
y2 = torch.exp(torch.max(x) - x)

plt.plot(x.numpy(), y1.numpy())
plt.plot(x.numpy(), y2.numpy())
# -

v = torch.tensor([1, 2, 3, 4, 5, 6]) # create a float tensor
print(v.size())

m = torch.tensor([[5., 6],
                  [7, 8],
                  [9, 10]])
print(m.shape)                  
m                  

a = torch.tensor([0, 3, 5, 5, 5, 2]).view(2, 3) # create a matrix of dim(2,3)
b = torch.tensor([3, 4, 3, -2, 4, -2]).view(3, 2) # create a matrix of dim(3, 2)
print(a @ b) # multiply a and b
print(torch.matmul(a, b)) # multiply a and b

# `Note`: matrix dimension must be kept on all rows and columns 

d3 = torch.tensor([
  [[111, 112, 113],
   [121, 122, 123]],
  [[211, 212, 213],
   [221, 222, 223.]]])
print(d3.shape)
d3   

# ## Tensor operations and gradients
# We can combine tensors with the usual arithmetic operations. Let's look at an example:

# create tensors.
x = torch.tensor(3.)
w = torch.tensor(4., requires_grad=True)
b = torch.tensor(5., requires_grad=True)
x, w, b

# We've created three tensors: `x`, `w`, and `b`, all numbers. `w` and `b` have an additional parameter `requires_grad` set to `True`. We'll see what it does in just a moment. 
#
# Let's create a new tensor `y` by combining these tensors.

# arithmetic operations
y = w * x + b
y

# As expected, `y` is a tensor with the value `3 * 4 + 5 = 17`. What makes PyTorch unique is that we can automatically compute the derivative of `y` w.r.t. the tensors that have `requires_grad` set to `True` i.e. w and b. This feature of PyTorch is called _autograd_ (automatic gradients).
#
# To compute the derivatives, we can invoke the `.backward` method on our result `y`.

# compute derivatives
y.backward()

# The derivatives of `y` with respect to the input tensors are stored in the `<tensor>.grad` property of the respective tensors.

# display gradients, x = 3, w = 4, b = 5
print('dy/dx:', x.grad)   # we didn't defined x with gradiant
print('dy/dw:', w.grad)   # the gradient of y with respect of w => w * 1 + 0 = 3
print('dy/db:', b.grad)   # the gradient of y with respect of b => w * 0 + 1 = 1 (we trait b as the variable of the derivitve)

# As expected, `dy/dw` has the same value as `x`, i.e., `3`, and `dy/db` has the value `1`. Note that `x.grad` is `None` because `x` doesn't have `requires_grad` set to `True`. 
#
# The "grad" in `w.grad` is short for _gradient_, which is another term for derivative. The term _gradient_ is primarily used while dealing with vectors and matrices.
