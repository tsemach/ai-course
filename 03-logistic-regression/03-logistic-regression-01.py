
#%% Imports
import torch
import torchvision
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
%matplotlib inline

#%%
# + colab={"base_uri": "https://localhost:8080/", "height": 389, "referenced_widgets": ["d7bc8179054c448c8decb471100d8edf", "3e158b4f6fc045a799a5d2848b7ef224", "7de6379c81224840ab318d1c5dbe2573", "8d85fb898cc4482794a273f9f40a7716", "ab0c7dba3efe488e962a95f77fdff901", "5e3647b0ec134d6cb7c76347435756d1", "41db56c1758644e28cc891ed17b9111e", "b278158fce8d4b619f02689a2d671560", "c755370c4edf4661b591e61de544ad8f", "392aac024b8845c58ab75d00893b4e42", "c50520b03cd0415c931144f086b2007a", "4ea3550eaeab40969e6a835721528c35", "12d3ec215dcc4b6794e3bbdef44602ce", "28ef332369de4d358909a586a1743c61", "332d4f3a80d84f07ad997aa55afd7f45", "69e7677bfdb3471d925d0a3eca1df5a7", "b4213321d97e4f54b356d0b466253e4c", "0e03dd08224f4e8c9c2f8d975358cdb7", "90fba2e9dcee4b5eaf94776dcd05b01a", "ef3735b9a82347edb1c193fcc1dc1ec2", "b91d4ce036c24e6fb29c9e3a2914e0bd", "96b3cf3c2b8c41ae832e3f33dcf13453", "57ea6daaa8484c59b3930783d0a9cf20", "043acb1db455426e8f56a61d6f39b222", "02a48991f3f2405db5dd2968af3177bb", "7113a3efba564850899ad46ec33a1ce0", "be465e46f1fa44c69c9dd8ff2d173543", "ef4e2e0ed447424b8695c1a66cec0210", "b612c47359234c3a8d59430c65605047", "7ffd56bd6fdb48efb48c338dfe983bfb", "eab337afee944438b7e9db9a2b7fd61b", "876cca63cea043f99904cd7a28f67e26"]} executionInfo={"elapsed": 2911, "status": "ok", "timestamp": 1606577890091, "user": {"displayName": "Aakash N S", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiIWFHtan62vtW1gz2Bv2bxL3rppefcadxzEVxRKQ=s64", "userId": "03254185060287524023"}, "user_tz": -330} id="q1Skz7dlfTTV" outputId="cfc70a7f-bbb0-4c05-e160-24c0f43bc0c7"
# Download training dataset
dataset = MNIST(root='data', download=True)

# + [markdown] id="Koi-fVbAfTTV"
# When this statement is executed for the first time, it downloads the data to the `data/` directory next to the notebook and creates a PyTorch `Dataset`. On subsequent executions, the download is skipped as the data is already downloaded. Let's check the size of the dataset.

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 843, "status": "ok", "timestamp": 1606577913840, "user": {"displayName": "Aakash N S", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiIWFHtan62vtW1gz2Bv2bxL3rppefcadxzEVxRKQ=s64", "userId": "03254185060287524023"}, "user_tz": -330} id="W9mWyKqAfTTV" outputId="c3ad6519-90c4-4584-d314-805c68cf6a7a"
print(len(dataset))
#%%
# + [markdown] id="6pRKB16NfTTW"
# The dataset has 60,000 images that we'll use to train the model. There is also an additional test set of 10,000 images used for evaluating models and reporting metrics in papers and reports. We can create the test dataset using the `MNIST` class by passing `train=False` to the constructor.

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 881, "status": "ok", "timestamp": 1606577951429, "user": {"displayName": "Aakash N S", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiIWFHtan62vtW1gz2Bv2bxL3rppefcadxzEVxRKQ=s64", "userId": "03254185060287524023"}, "user_tz": -330} id="zLqZQOlWfTTW" outputId="858232eb-e985-45ea-dff9-261c6e17567e"
test_dataset = MNIST(root='data/', train=False)
len(test_dataset)

# + [markdown] id="bgJru-f7fTTW"
# Let's look at a sample element from the training dataset.

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 748, "status": "ok", "timestamp": 1606577960430, "user": {"displayName": "Aakash N S", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiIWFHtan62vtW1gz2Bv2bxL3rppefcadxzEVxRKQ=s64", "userId": "03254185060287524023"}, "user_tz": -330} id="4ZmSZeFGfTTW" outputId="f38133b3-f015-4ab9-9773-3d196e17f9a1"
print(dataset[0])
#%%

# + [markdown] id="2oBdGAc4fTTX"
# It's a pair, consisting of a 28x28px image and a label. The image is an object of the class `PIL.Image.Image`, which is a part of the Python imaging library [Pillow](https://pillow.readthedocs.io/en/stable/). We can view the image within Jupyter using [`matplotlib`](https://matplotlib.org/), the de-facto plotting and graphing library for data science in Python.

# + executionInfo={"elapsed": 769, "status": "ok", "timestamp": 1606578019026, "user": {"displayName": "Aakash N S", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiIWFHtan62vtW1gz2Bv2bxL3rppefcadxzEVxRKQ=s64", "userId": "03254185060287524023"}, "user_tz": -330} id="edyBIgXPfTTX"
import matplotlib.pyplot as plt
# %matplotlib inline

#%%
# + [markdown] id="TSVQL5vJfTTX"
# The statement `%matplotlib inline` indicates to Jupyter that we want to plot the graphs within the notebook. Without this line, Jupyter will show the image in a popup. Statements starting with `%` are called magic commands and are used to configure the behavior of Jupyter itself. You can find a full list of magic commands here: https://ipython.readthedocs.io/en/stable/interactive/magics.html .
#
# Let's look at a couple of images from the dataset.

# + colab={"base_uri": "https://localhost:8080/", "height": 282} executionInfo={"elapsed": 764, "status": "ok", "timestamp": 1606578085723, "user": {"displayName": "Aakash N S", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiIWFHtan62vtW1gz2Bv2bxL3rppefcadxzEVxRKQ=s64", "userId": "03254185060287524023"}, "user_tz": -330} id="3cxwpTQLfTTX" outputId="0e67e177-1d1e-489e-f45e-3353d7960076"
image, label = dataset[0]
plt.imshow(image, cmap='gray')
print('Label:', label)

#%%
# + colab={"base_uri": "https://localhost:8080/", "height": 282} executionInfo={"elapsed": 798, "status": "ok", "timestamp": 1606578094615, "user": {"displayName": "Aakash N S", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiIWFHtan62vtW1gz2Bv2bxL3rppefcadxzEVxRKQ=s64", "userId": "03254185060287524023"}, "user_tz": -330} id="35ELPbM7fTTX" outputId="0bfafb5a-46c1-465d-aeff-b7ee7ff601bc"
image, label = dataset[10]
plt.imshow(image, cmap='gray')
print('Label:', label)

#%%
# + [markdown] id="5vO2Gh9RfTTX"
# It's evident that these images are relatively small in size, and recognizing the digits can sometimes be challenging even for the human eye. While it's useful to look at these images, there's just one problem here: PyTorch doesn't know how to work with images. We need to convert the images into tensors. We can do this by specifying a transform while creating our dataset.

# + executionInfo={"elapsed": 757, "status": "ok", "timestamp": 1606578158015, "user": {"displayName": "Aakash N S", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiIWFHtan62vtW1gz2Bv2bxL3rppefcadxzEVxRKQ=s64", "userId": "03254185060287524023"}, "user_tz": -330} id="FBnad_UDfTTX"
import torchvision.transforms as transforms

# + [markdown] id="084T0emtfTTX"
# PyTorch datasets allow us to specify one or more transformation functions that are applied to the images as they are loaded. The `torchvision.transforms` module contains many such predefined functions. We'll use the `ToTensor` transform to convert images into PyTorch tensors.

# + executionInfo={"elapsed": 927, "status": "ok", "timestamp": 1606578184520, "user": {"displayName": "Aakash N S", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiIWFHtan62vtW1gz2Bv2bxL3rppefcadxzEVxRKQ=s64", "userId": "03254185060287524023"}, "user_tz": -330} id="hYvBQ3LzfTTX"
# MNIST dataset (images and labels)
dataset = MNIST(root='data/',
                train=True,
                transform=transforms.ToTensor())

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 781, "status": "ok", "timestamp": 1606578201993, "user": {"displayName": "Aakash N S", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiIWFHtan62vtW1gz2Bv2bxL3rppefcadxzEVxRKQ=s64", "userId": "03254185060287524023"}, "user_tz": -330} id="nCZ4h7jMfTTX" outputId="d182f619-8e57-4864-ec67-8f2b08c78fc7"
img_tensor, label = dataset[0]
print(type(dataset[0][0]))
print(img_tensor.shape, label)

#%%
# + [markdown] id="Io317DfSfTTX"
# The image is now converted to a 1x28x28 tensor. The first dimension tracks color channels. The second and third dimensions represent pixels along the height and width of the image, respectively. Since images in the MNIST dataset are grayscale, there's just one channel. Other datasets have images with color, in which case there are three channels: red, green, and blue (RGB).
#
# Let's look at some sample values inside the tensor.

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 792, "status": "ok", "timestamp": 1606578295334, "user": {"displayName": "Aakash N S", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiIWFHtan62vtW1gz2Bv2bxL3rppefcadxzEVxRKQ=s64", "userId": "03254185060287524023"}, "user_tz": -330} id="prZFgdtefTTX" outputId="835d4232-380d-456a-fafd-8f8474766ba0"
print(img_tensor[:, 10:15, 10:15])
print(torch.max(img_tensor), torch.min(img_tensor))

#%%
# + [markdown] id="_2HjQ5P5fTTX"
# The values range from 0 to 1, with `0` representing black, `1` white, and the values in between different shades of grey. We can also plot the tensor as an image using `plt.imshow`.

# + colab={"base_uri": "https://localhost:8080/", "height": 265} executionInfo={"elapsed": 828, "status": "ok", "timestamp": 1606578335519, "user": {"displayName": "Aakash N S", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiIWFHtan62vtW1gz2Bv2bxL3rppefcadxzEVxRKQ=s64", "userId": "03254185060287524023"}, "user_tz": -330} id="Fw-NvsqpfTTX" outputId="5b128f82-14dc-467b-c704-a2795dce6c72"
# Plot the image by passing in the 28x28 matrix
plt.imshow(img_tensor[0,10:15,10:15], cmap='gray');
#%%

# + [markdown] id="WudqBr3pfTTX"
# Note that we need to pass just the 28x28 matrix to `plt.imshow`, without a channel dimension. We also pass a color map (`cmap=gray`) to indicate that we want to see a grayscale image.
# + [markdown] id="qSKTZ8aifTTX"
# ## Training and Validation Datasets
#
# While building real-world machine learning models, it is quite common to split the dataset into three parts:
#
# 1. **Training set** - used to train the model, i.e., compute the loss and adjust the model's weights using gradient descent.
# 2. **Validation set** - used to evaluate the model during training, adjust hyperparameters (learning rate, etc.), and pick the best version of the model.
# 3. **Test set** - used to compare different models or approaches and report the model's final accuracy.
#
# In the MNIST dataset, there are 60,000 training images and 10,000 test images. The test set is standardized so that different researchers can report their models' results against the same collection of images.
#
# Since there's no predefined validation set, we must manually split the 60,000 images into training and validation datasets. Let's set aside 10,000 randomly chosen images for validation. We can do this using the `random_spilt` method from PyTorch.

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 802, "status": "ok", "timestamp": 1606578530215, "user": {"displayName": "Aakash N S", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiIWFHtan62vtW1gz2Bv2bxL3rppefcadxzEVxRKQ=s64", "userId": "03254185060287524023"}, "user_tz": -330} id="1dcE0OPBfTTX" outputId="be74ca6a-d40e-4ccf-cc24-ea6231a6e097"
from torch.utils.data import random_split

train_ds, val_ds = random_split(dataset, [50000, 10000])
print(len(train_ds), len(val_ds))

#%%
print(type(train_ds))
print('train_ds len:', len(train_ds))
print(train_ds)

print(type(val_ds))
print('val_ds:', len(val_ds))
print(val_ds)

#%%
# + [markdown] id="CACcvMnZfTTX"
# It's essential to choose a random sample for creating a validation set. Training data is often sorted by the target labels, i.e., images of 0s, followed by 1s, followed by 2s, etc. If we create a validation set using the last 20% of images, it would only consist of 8s and 9s. In contrast, the training set would contain no 8s or 9s. Such a training-validation would make it impossible to train a useful model.
#
# We can now create data loaders to help us load the data in batches. We'll use a batch size of 128.
#
#
#

# + executionInfo={"elapsed": 846, "status": "ok", "timestamp": 1606578689940, "user": {"displayName": "Aakash N S", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiIWFHtan62vtW1gz2Bv2bxL3rppefcadxzEVxRKQ=s64", "userId": "03254185060287524023"}, "user_tz": -330} id="pD_oD7BTfTTX"
from torch.utils.data import DataLoader

batch_size = 128

train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size)
#%%

print('train_loader len:', len(train_loader))
print('val_loader len:', len(val_loader))
#%%

# + [markdown] id="NCtbVhkOfTTX"
# We set `shuffle=True` for the training data loader to ensure that the batches generated in each epoch are different. This randomization helps generalize & speed up the training process. On the other hand, since the validation data loader is used only for evaluating the model, there is no need to shuffle the images.
#

# + [markdown] id="HEyJshGxfTTY"
# ### Save and upload your notebook
#
# Whether you're running this Jupyter notebook online or on your computer, it's essential to save your work from time to time. You can continue working on a saved notebook later or share it with friends and colleagues to let them execute your code. [Jovian](https://jovian.ai/platform-features) offers an easy way of saving and sharing your Jupyter notebooks online.

# + executionInfo={"elapsed": 3385, "status": "ok", "timestamp": 1606578705431, "user": {"displayName": "Aakash N S", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiIWFHtan62vtW1gz2Bv2bxL3rppefcadxzEVxRKQ=s64", "userId": "03254185060287524023"}, "user_tz": -330} id="QWDMN_nsfTTY"
# Install the library
# !pip install jovian --upgrade --quiet

# + executionInfo={"elapsed": 616, "status": "ok", "timestamp": 1606578706947, "user": {"displayName": "Aakash N S", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiIWFHtan62vtW1gz2Bv2bxL3rppefcadxzEVxRKQ=s64", "userId": "03254185060287524023"}, "user_tz": -330} id="HS6PzY2KfTTY"
# import jovian

# + colab={"base_uri": "https://localhost:8080/", "height": 139} executionInfo={"elapsed": 20491, "status": "ok", "timestamp": 1606578738611, "user": {"displayName": "Aakash N S", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiIWFHtan62vtW1gz2Bv2bxL3rppefcadxzEVxRKQ=s64", "userId": "03254185060287524023"}, "user_tz": -330} id="LkpRo2TffTTY" outputId="9b371bfd-b0e2-4f18-c82a-0ef65ec442ef"
#jovian.commit(project='03-logistic-regression-live')

# + [markdown] id="rUFBNaPFfTTY"
# `jovian.commit` uploads the notebook to your Jovian account, captures the Python environment, and creates a shareable link for your notebook, as shown above. You can use this link to share your work and let anyone (including you) run your notebooks and reproduce your work.

# + [markdown] id="Fy-lajKQfTTY"
# ## Model
#
# Now that we have prepared our data loaders, we can define our model.
#
# * A **logistic regression** model is almost identical to a linear regression model. It contains weights and bias matrices, and the output is obtained using simple matrix operations (`pred = x @ w.t() + b`).
#
# * As we did with linear regression, we can use `nn.Linear` to create the model instead of manually creating and initializing the matrices.
#
# * Since `nn.Linear` expects each training example to be a vector, each `1x28x28` image tensor is _flattened_ into a vector of size 784 `(28*28)` before being passed into the model.
#
# * The output for each image is a vector of size 10, with each element signifying the probability of a particular target label (i.e., 0 to 9). The predicted label for an image is simply the one with the highest probability.

# + executionInfo={"elapsed": 1078, "status": "ok", "timestamp": 1606579070048, "user": {"displayName": "Aakash N S", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiIWFHtan62vtW1gz2Bv2bxL3rppefcadxzEVxRKQ=s64", "userId": "03254185060287524023"}, "user_tz": -330} id="zzVbD7pifTTY"
import torch.nn as nn

input_size = 28*28
num_classes = 10

# Logistic regression model
model = nn.Linear(input_size, num_classes)
#%%

# + [markdown] id="2ilamccMfTTY"
# Of course, this model is a lot larger than our previous model in terms of the number of parameters. Let's take a look at the weights and biases.

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 765, "status": "ok", "timestamp": 1606579085171, "user": {"displayName": "Aakash N S", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiIWFHtan62vtW1gz2Bv2bxL3rppefcadxzEVxRKQ=s64", "userId": "03254185060287524023"}, "user_tz": -330} id="HAj8gVdUfTTY" outputId="3569f384-390a-4555-949f-a85b79a22ed0"
print("model.weight.shape", model.weight.shape)
print("model.weight", model.weight)
print('')
# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1187, "status": "ok", "timestamp": 1606579093000, "user": {"displayName": "Aakash N S", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiIWFHtan62vtW1gz2Bv2bxL3rppefcadxzEVxRKQ=s64", "userId": "03254185060287524023"}, "user_tz": -330} id="fRnviQ34fTTY" outputId="25bf0abf-427c-4079-8cc2-0d03eb6af0b2"
print("model.bias.shape", model.bias.shape)
print("model.bias", model.bias)
#%%

# + [markdown] id="Is6yNefafTTY"
# Although there are a total of 7850 parameters here, conceptually, nothing has changed so far. Let's try and generate some outputs using our model. We'll take the first batch of 100 images from our dataset and pass them into our model.

# + colab={"base_uri": "https://localhost:8080/", "height": 517} executionInfo={"elapsed": 645, "status": "error", "timestamp": 1606579153232, "user": {"displayName": "Aakash N S", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiIWFHtan62vtW1gz2Bv2bxL3rppefcadxzEVxRKQ=s64", "userId": "03254185060287524023"}, "user_tz": -330} id="qIuxYOy3fTTY" outputId="a58c1ab0-bddb-456a-d9a4-d27d9b6d8332"
for images, labels in train_loader:
    print("labels", labels)
    print("labels.spape", labels.shape)
    print("images.shape:", images.shape)
    # outputs = model(images)
    # print("outputs:", outputs)
    break
#%%
print("labels.spape", labels.shape)
print("images.shape:", images.shape)
xb = images
xb = xb.reshape(-1, 784)
print("images.shape:", xb.shape)

#%%
# + executionInfo={"elapsed": 1083, "status": "ok", "timestamp": 1606579733133, "user": {"displayName": "Aakash N S", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiIWFHtan62vtW1gz2Bv2bxL3rppefcadxzEVxRKQ=s64", "userId": "03254185060287524023"}, "user_tz": -330} id="c2HWol7qfTTY"
class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, xb):
        print("[MnistModel] forward is called")
        xb = xb.reshape(-1, 784)
        out = self.linear(xb)
        return out

model = MnistModel()
#%%

# + [markdown] id="mZKn3GBFfTTY"
# Inside the `__init__` constructor method, we instantiate the weights and biases using `nn.Linear`. And inside the `forward` method, which is invoked when we pass a batch of inputs to the model, we flatten the input tensor and pass it into `self.linear`.
#
# `xb.reshape(-1, 28*28)` indicates to PyTorch that we want a *view* of the `xb` tensor with two dimensions. The length along the 2nd dimension is 28\*28 (i.e., 784). One argument to `.reshape` can be set to `-1` (in this case, the first dimension) to let PyTorch figure it out automatically based on the shape of the original tensor.
#
# Note that the model no longer has `.weight` and `.bias` attributes (as they are now inside the `.linear` attribute), but it does have a `.parameters` method that returns a list containing the weights and bias.

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 887, "status": "ok", "timestamp": 1606579744253, "user": {"displayName": "Aakash N S", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiIWFHtan62vtW1gz2Bv2bxL3rppefcadxzEVxRKQ=s64", "userId": "03254185060287524023"}, "user_tz": -330} id="voZF2M_3nVnI" outputId="1feeba1e-2ad7-4d32-9eb6-56e12f587a51"
model.linear

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 751, "status": "ok", "timestamp": 1606579783293, "user": {"displayName": "Aakash N S", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiIWFHtan62vtW1gz2Bv2bxL3rppefcadxzEVxRKQ=s64", "userId": "03254185060287524023"}, "user_tz": -330} id="JciejPo9fTTY" outputId="c3ea0a15-d0bb-4d44-f755-1d1ad4fe50ff"
print(model.linear.weight.shape, model.linear.bias.shape)
print('model.parameters()', model.parameters())
print(list(model.parameters()))
#%%

# make a prediction of one (first) batch of images,
# NOTE: Forwarrd is call on MnistModel every time a module is called
for images, labels in train_loader:
    print("images.shape", images.shape)
    outputs = model(images) # images are reshaped by forward method of MnistModel
    break

print('outputs.shape : ', outputs.shape)
print('Sample outputs :\n', outputs[:2].data)
#%%

# + [markdown] id="9v407S1jfTTZ"
# For each of the 100 input images, we get 10 outputs, one for each class. As discussed earlier, we'd like these outputs to represent probabilities. Each output row's elements must lie between 0 to 1 and add up to 1, which is not the case.
#
# To convert the output rows into probabilities, we use the softmax function, which has the following formula:
#
# ![softmax](https://i.imgur.com/EAh9jLN.png)
#
# First, we replace each element `yi` in an output row by `e^yi`, making all the elements positive.
#
# ![](https://www.montereyinstitute.org/courses/DevelopmentalMath/COURSE_TEXT2_RESOURCE/U18_L1_T1_text_final_6_files/image001.png)
#
#
#
# Then, we divide them by their sum to ensure that they add up to 1. The resulting vector can thus be interpreted as probabilities.
#
# While it's easy to implement the softmax function (you should try it!), we'll use the implementation that's provided within PyTorch because it works well with multidimensional tensors (a list of output rows in our case).

# + executionInfo={"elapsed": 781, "status": "ok", "timestamp": 1606580126906, "user": {"displayName": "Aakash N S", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiIWFHtan62vtW1gz2Bv2bxL3rppefcadxzEVxRKQ=s64", "userId": "03254185060287524023"}, "user_tz": -330} id="Wc7d6WD0fTTZ"
import torch.nn.functional as F

# The softmax function is included in the `torch.nn.functional` package and requires us to specify a dimension along which the function should be applied.
print('outputs[:2]:', outputs[:2])
probs = F.softmax(outputs, dim=1)

# Look at sample probabilities
print("probs.shape:", probs.shape)
print("Sample probabilities:", probs[:2].data)

# Add up the probabilities of an output row
print("Sum: ", torch.sum(probs[0]).item())
#%%

# + [markdown] id="k2cgqWGVfTTZ"
# Finally, we can determine the predicted label for each image by simply choosing the index of the element with the highest probability in each output row. We can do this using `torch.max`, which returns each row's largest element and the corresponding index.

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1108, "status": "ok", "timestamp": 1606580284840, "user": {"displayName": "Aakash N S", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiIWFHtan62vtW1gz2Bv2bxL3rppefcadxzEVxRKQ=s64", "userId": "03254185060287524023"}, "user_tz": -330} id="1o9a4hAufTTZ" outputId="ffe611f9-df20-4bbc-fb1c-6178faac90cc"
max_probs, preds_idx = torch.max(probs, dim=1)
print('preds_idx.shape:', preds_idx.shape)
print('preds:', preds_idx)
print('max_probs.shape:', max_probs.shape)
print('max_probs:', max_probs)
#%%

# + executionInfo={"elapsed": 967, "status": "ok", "timestamp": 1606580510348, "user": {"displayName": "Aakash N S", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiIWFHtan62vtW1gz2Bv2bxL3rppefcadxzEVxRKQ=s64", "userId": "03254185060287524023"}, "user_tz": -330} id="11hDOTR1fTTZ"
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))
#%%

# The `==` operator performs an element-wise comparison of two tensors with the same shape and returns a tensor of the same shape, containing `True` for unequal elements and `False` for equal elements. Passing the result to `torch.sum` returns the number of labels that were predicted correctly. Finally, we divide by the total number of images to get the accuracy.
# Note that we don't need to apply  softmax to the outputs since its results have the same relative order. This is because `e^x` is an increasing function, i.e., if `y1 > y2`, then `e^y1 > e^y2`. The same holds after averaging out the values to get the softmax.
# Let's calculate the accuracy of the current model on the first batch of data.
print(accuracy(outputs, labels))
#%%

# Accuracy is an excellent way for us (humans) to evaluate the model. However, we can't use it as a loss function for optimizing our model using gradient descent for the following reasons:
# 1. It's not a differentiable function. `torch.max` and `==` are both non-continuous and non-differentiable operations, so we can't use the accuracy for computing gradients w.r.t the weights and biases.
# 2. It doesn't take into account the actual probabilities predicted by the model, so it can't provide sufficient feedback for incremental improvements.

# For these reasons, accuracy is often used as an **evaluation metric** for classification, but not as a loss function. A commonly used loss function for classification problems is the **cross-entropy**, which has the following formula:
# While it looks complicated, it's actually quite simple:
# * For each output row, pick the predicted probability for the correct label. E.g., if the predicted probabilities for an image are `[0.1, 0.3, 0.2, ...]` and the correct label is `1`, we pick the corresponding element `0.3` and ignore the rest.
# * Then, take the [logarithm](https://en.wikipedia.org/wiki/Logarithm) of the picked probability. If the probability is high, i.e., close to 1, then its logarithm is a very small negative value, close to 0. And if the probability is low (close to 0), then the logarithm is a very large negative value. We also multiply the result by -1, which results is a large postive value of the loss for poor predictions.
#
# ![](https://www.intmath.com/blog/wp-content/images/2019/05/log10.png)
#
# * Finally, take the average of the cross entropy across all the output rows to get the overall loss for a batch of data.
#
# Unlike accuracy, cross-entropy is a continuous and differentiable function. It also provides useful feedback for incremental improvements in the model (a slightly higher probability for the correct label leads to a lower loss). These two factors make cross-entropy a better choice for the loss function.
#
# As you might expect, PyTorch provides an efficient and tensor-friendly implementation of cross-entropy as part of the `torch.nn.functional` package. Moreover, it also performs softmax internally, so we can directly pass in the model's outputs without converting them into probabilities.

# + executionInfo={"elapsed": 804, "status": "ok", "timestamp": 1606580980198, "user": {"displayName": "Aakash N S", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiIWFHtan62vtW1gz2Bv2bxL3rppefcadxzEVxRKQ=s64", "userId": "03254185060287524023"}, "user_tz": -330} id="OqWUQpANfTTZ"
loss_fn = F.cross_entropy

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1109, "status": "ok", "timestamp": 1606580980798, "user": {"displayName": "Aakash N S", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiIWFHtan62vtW1gz2Bv2bxL3rppefcadxzEVxRKQ=s64", "userId": "03254185060287524023"}, "user_tz": -330} id="SuESG0hbfTTZ" outputId="10804014-e2d7-4b7c-c56d-b56761ffb464"
# Loss for current batch of data
# loss_fn include softmax in it
loss = loss_fn(outputs, labels)
print(loss)
#%%

# We know that cross-entropy is the negative logarithm of the predicted probability of the correct label averaged over all training samples. Therefore, one way to interpret the resulting number e.g. `2.23` is look at `e^-2.23` which is around `0.1` as the predicted probability of the correct label, on average. *The lower the loss, The better the model.*

# ## Training the model
#
# Now that we have defined the data loaders, model, loss function and optimizer, we are ready to train the model. The training process is identical to linear regression, with the addition of a "validation phase" to evaluate the model in each epoch. Here's what it looks like in pseudocode:
#
# ```
# for epoch in range(num_epochs):
#     # Training phase
#     for batch in train_loader:
#         # Generate predictions
#         # Calculate loss
#         # Compute gradients
#         # Update weights
#         # Reset gradients
#
#     # Validation phase
#     for batch in val_loader:
#         # Generate predictions
#         # Calculate loss
#         # Calculate metrics (accuracy etc.)
#     # Calculate average validation loss & metrics
#
#     # Log epoch, loss & metrics for inspection
# ```
#
# Some parts of the training loop are specific the specific problem we're solving (e.g. loss function, metrics etc.) whereas others are generic and can be applied to any deep learning problem.
#
# We'll include the problem-independent parts within a function called `fit`, which will be used to train the model. The problem-specific parts will be implemented by adding new methods to the `nn.Module` class.
def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    optimizer = opt_func(model.parameters(), lr)
    history = [] # for recording epoch-wise results

    for epoch in range(epochs):

        # Training Phase
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)

    return history
#%%

def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    print('evaluate type(outputs):', type(outputs), 'len:', len(outputs), type(val_loader))
    print('evaluate outputs[0]:', outputs[0])

    return model.validation_epoch_end(outputs)
#%%

# Finally, let's redefine the `MnistModel` class to include additional methods `training_step`, `validation_step`, `validation_epoch_end`, and `epoch_end` used by `fit` and `evaluate`.
class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, xb):
        # print("[MnistModel] forward is called")
        xb = xb.reshape(-1, 784)
        out = self.linear(xb)
        return out

    def training_step(self, batch):
        # print("[MnistModel] training_step called")
        images, labels = batch
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss

        return loss

    def validation_step(self, batch):
        # print("[MnistModel] validation_step called")
        images, labels = batch
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy

        return {'val_loss': loss, 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        print("[MnistModel] validation_epoch_end called")
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies

        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))

model = MnistModel()
#%%

# Before we train the model, let's see how the model performs on the validation set with the initial set of randomly initialized weights & biases.
result0 = evaluate(model, val_loader)
print(result0)
#%%

# The initial accuracy is around 10%, which one might expect from a randomly initialized model (since it has a 1 in 10 chance of getting a label right by guessing randomly).
# We are now ready to train the model. Let's train for five epochs and look at the results.
history1 = fit(5, 0.001, model, train_loader, val_loader)
#%%

# That's a great result! With just 5 epochs of training, our model has reached an accuracy of over 80% on the validation set. Let's see if we can improve that by training for a few more epochs. Try changing the learning rates and number of epochs in each of the cells below.
history2 = fit(5, 0.001, model, train_loader, val_loader)
#%%

history3 = fit(5, 0.001, model, train_loader, val_loader)
#%%

history4 = fit(5, 0.001, model, train_loader, val_loader)
#%%

history5 = fit(5, 0.001, model, train_loader, val_loader)
#%%

# While the accuracy does continue to increase as we train for more epochs, the improvements get smaller with every epoch. Let's visualize this using a line graph.
history = [result0] + history1 + history2 + history3 + history4 + history5
accuracies = [result['val_acc'] for result in history]
plt.plot(accuracies, '-x')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Accuracy vs. No. of epochs');
#%%

# It's quite clear from the above picture that the model probably won't cross the accuracy threshold of 90% even after training for a very long time. One possible reason for this is that the learning rate might be too high. The model's parameters may be "bouncing" around the optimal set of parameters for the lowest loss. You can try reducing the learning rate and training for a few more epochs to see if it helps.
# The more likely reason that **the model just isn't powerful enough**. If you remember our initial hypothesis, we have assumed that the output (in this case the class probabilities) is a **linear function** of the input (pixel intensities), obtained by perfoming a matrix multiplication with the weights matrix and adding the bias. This is a fairly weak assumption, as there may not actually exist a linear relationship between the pixel intensities in an image and the digit it represents. While it works reasonably well for a simple dataset like MNIST (getting us to 85% accuracy), we need more sophisticated models that can capture non-linear relationships between image pixels and labels for complex tasks like recognizing everyday objects, animals etc.
# Let's save our work using `jovian.commit`. Along with the notebook, we can also record some metrics from our training.
jovian.log_metrics(val_acc=history[-1]['val_acc'], val_loss=history[-1]['val_loss'])

# + colab={"base_uri": "https://localhost:8080/", "height": 104} executionInfo={"elapsed": 3324, "status": "ok", "timestamp": 1606582746421, "user": {"displayName": "Aakash N S", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiIWFHtan62vtW1gz2Bv2bxL3rppefcadxzEVxRKQ=s64", "userId": "03254185060287524023"}, "user_tz": -330} id="y0qr0ZNAfTTa" outputId="f9b1f120-be74-418d-fe2a-77769d51d499"
jovian.commit(project='03-logistic-regression', environment=None)
#%%

# ## Testing with individual images
# While we have been tracking the overall accuracy of a model so far, it's also a good idea to look at model's results on some sample images. Let's test out our model with some images from the predefined test dataset of 10000 images. We begin by recreating the test dataset with the `ToTensor` transform.
# Define test dataset
test_dataset = MNIST(root='data/',
                     train=False,
                     transform=transforms.ToTensor())

# + [markdown] id="Sw9J_viefTTa"
# Here's a sample image from the dataset.

# + colab={"base_uri": "https://localhost:8080/", "height": 299} executionInfo={"elapsed": 784, "status": "ok", "timestamp": 1606582938197, "user": {"displayName": "Aakash N S", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiIWFHtan62vtW1gz2Bv2bxL3rppefcadxzEVxRKQ=s64", "userId": "03254185060287524023"}, "user_tz": -330} id="yzEdFHq6fTTa" outputId="4e0faa6b-789f-46c4-9c25-27441665844c"
img, label = test_dataset[0]
plt.imshow(img[0], cmap='gray')
print('Shape:', img.shape)
print('Label:', label)
#%%

# + [markdown] id="QgwbsDUjfTTa"
# Let's define a helper function `predict_image`, which returns the predicted label for a single image tensor.

# + executionInfo={"elapsed": 802, "status": "ok", "timestamp": 1606583060675, "user": {"displayName": "Aakash N S", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiIWFHtan62vtW1gz2Bv2bxL3rppefcadxzEVxRKQ=s64", "userId": "03254185060287524023"}, "user_tz": -330} id="9FfeYQaOfTTa"
def predict_image(img, model):
    print("img: type(img)", type(img), img.shape)
    xb = img.unsqueeze(0)
    print("xb: type(xp)", type(xb), xb.shape)
    yb = model(xb)
    _, preds = torch.max(yb, dim=1)
    return preds[0].item()
#%%

# `img.unsqueeze` simply adds another dimension at the begining of the 1x28x28 tensor, making it a 1x1x28x28 tensor, which the model views as a batch containing a single image.
# Let's try it out with a few images.
img, label = test_dataset[0]
plt.imshow(img[0], cmap='gray')
print('Label:', label, ', Predicted:', predict_image(img, model))
#%%

img, label = test_dataset[10]
plt.imshow(img[0], cmap='gray')
print('Label:', label, ', Predicted:', predict_image(img, model))
#%%

img, label = test_dataset[193]
plt.imshow(img[0], cmap='gray')
print('Label:', label, ', Predicted:', predict_image(img, model))

# + colab={"base_uri": "https://localhost:8080/", "height": 282} executionInfo={"elapsed": 790, "status": "ok", "timestamp": 1606583155559, "user": {"displayName": "Aakash N S", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiIWFHtan62vtW1gz2Bv2bxL3rppefcadxzEVxRKQ=s64", "userId": "03254185060287524023"}, "user_tz": -330} id="eetYJdoFfTTb" outputId="da8c6613-355a-4479-cb49-574236d2de5c"
img, label = test_dataset[1839]
plt.imshow(img[0], cmap='gray')
print('Label:', label, ', Predicted:', predict_image(img, model))

# Identifying where our model performs poorly can help us improve the model, by collecting more training data, increasing/decreasing the complexity of the model, and changing the hypeparameters.
# As a final step, let's also look at the overall loss and accuracy of the model on the test set.
# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1572, "status": "ok", "timestamp": 1606583301081, "user": {"displayName": "Aakash N S", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiIWFHtan62vtW1gz2Bv2bxL3rppefcadxzEVxRKQ=s64", "userId": "03254185060287524023"}, "user_tz": -330} id="6bdjaeG-fTTb" outputId="c4479ab7-ee92-46d4-ac0d-9217d6873f43"
test_loader = DataLoader(test_dataset, batch_size=256)
result = evaluate(model, test_loader)
print('result of test set:', result)
#%%

# We expect this to be similar to the accuracy/loss on the validation set. If not, we might need a better validation set that has similar data and distribution as the test set (which often comes from real world data).
# ## Saving and loading the model
# Since we've trained our model for a long time and achieved a resonable accuracy, it would be a good idea to save the weights and bias matrices to disk, so that we can reuse the model later and avoid retraining from scratch. Here's how you can save the model.
torch.save(model.state_dict(), 'mnist-logistic.pth')

# The `.state_dict` method returns an `OrderedDict` containing all the weights and bias matrices mapped to the right attributes of the model.
# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 781, "status": "ok", "timestamp": 1606583449734, "user": {"displayName": "Aakash N S", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiIWFHtan62vtW1gz2Bv2bxL3rppefcadxzEVxRKQ=s64", "userId": "03254185060287524023"}, "user_tz": -330} id="elbp48SCfTTb" outputId="84f92a49-910c-4466-8b96-ab267026f3ac"
model.state_dict()

# + [markdown] id="Qc9kRTDpfTTb"
# To load the model weights, we can instante a new object of the class `MnistModel`, and use the `.load_state_dict` method.

# + executionInfo={"elapsed": 845, "status": "ok", "timestamp": 1606583501874, "user": {"displayName": "Aakash N S", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiIWFHtan62vtW1gz2Bv2bxL3rppefcadxzEVxRKQ=s64", "userId": "03254185060287524023"}, "user_tz": -330} id="OO666r7_1rbW"
model2 = MnistModel()

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 776, "status": "ok", "timestamp": 1606583510607, "user": {"displayName": "Aakash N S", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiIWFHtan62vtW1gz2Bv2bxL3rppefcadxzEVxRKQ=s64", "userId": "03254185060287524023"}, "user_tz": -330} id="aUvYwe4Q1six" outputId="63adc692-57b8-458d-a926-dc969bf491db"
model2.state_dict()

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1715, "status": "ok", "timestamp": 1606583526111, "user": {"displayName": "Aakash N S", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiIWFHtan62vtW1gz2Bv2bxL3rppefcadxzEVxRKQ=s64", "userId": "03254185060287524023"}, "user_tz": -330} id="gHac9s6e1vyS" outputId="bd0ef06a-ded0-4d91-c0a1-620b310bc89e"
evaluate(model2, test_loader)

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 751, "status": "ok", "timestamp": 1606583546422, "user": {"displayName": "Aakash N S", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiIWFHtan62vtW1gz2Bv2bxL3rppefcadxzEVxRKQ=s64", "userId": "03254185060287524023"}, "user_tz": -330} id="bvR1g8ggfTTb" outputId="2cb76bb5-b26a-472c-d2b3-3f872ebb8c7a"
model2.load_state_dict(torch.load('mnist-logistic.pth'))
model2.state_dict()

# + [markdown] id="zhjBm4BMfTTb"
# Just as a sanity check, let's verify that this model has the same loss and accuracy on the test set as before.

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1668, "status": "ok", "timestamp": 1606583555728, "user": {"displayName": "Aakash N S", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiIWFHtan62vtW1gz2Bv2bxL3rppefcadxzEVxRKQ=s64", "userId": "03254185060287524023"}, "user_tz": -330} id="UynZ4aSLfTTb" outputId="ed0184c3-6239-4610-b707-09b2e4cbd49d"
test_loader = DataLoader(test_dataset, batch_size=256)
result = evaluate(model2, test_loader)
result

# + [markdown] id="BRMNaQUWfTTb"
# As a final step, we can save and commit our work using the `jovian` library. Along with the notebook, we can also attach the weights of our trained model, so that we can use it later.

# + id="p1cuCZ6afTTb"
jovian.commit(project='03-logistic-regression', environment=None, outputs=['mnist-logistic.pth'])
