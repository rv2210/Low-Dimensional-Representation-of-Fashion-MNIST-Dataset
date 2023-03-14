import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import seaborn as sn
import pandas as pd
import tensorflow as tf
from tensorflow import keras

#Loading Train and Validation Datasets from TorchVision
transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.5,),(0.5,),)])


Train_data = datasets.FashionMNIST(root = 'HW2/Fashion_MNIST_Datasets/FashionMNIST',
                                    train = True, download = True, transform = transforms.ToTensor())

#print(type(train_data))
#print(len(train_data))

#print("Training dataset = ", train_data)

Validation_data = datasets.FashionMNIST(root = 'HW2/Fashion_MNIST_Datasets/FashionMNIST',
                                        download = True, transform = transforms.ToTensor())

#print(type(validation_data))
#print(len(validation_data))

#print("\nValidation dataset = ", validation_data)

input_data_size= 28 * 28 #Each input image size is 28 by 28 pixels so total of 784 pixels 
output_data_size= 10     #Output is a number 0-9 which shows what the fashion item is in the image



class Softmax(nn.Module):
    def __init__(self, input_data_size, output_data_size):
        super(Softmax, self).__init__()
        self.linear = nn.Linear(input_data_size, output_data_size, bias= False)
    
    def forward(self, X):
        z = self.linear(X)
        return z

model = Softmax(input_data_size, output_data_size)
#print('\nModel = ', model)

X = Train_data[0][0]

#print(len(X))

X = X.view(-1, 28*28)   #Flattening the numpy array 

#print(X)
model(X)

learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
#optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
criterion = nn.CrossEntropyLoss()
train_data_loader = torch.utils.data.DataLoader(dataset = Train_data, batch_size = 50)
validation_data_loader = torch.utils.data.DataLoader(dataset = Validation_data, batch_size = 2500)

#print(train_data_loader)

model_output = model(X)
softmax = nn.Softmax(dim = 1)
probability = softmax(model_output)
print(probability)


epochs = 10
loss = []
accuracy = []

#print(len(validation_data))
Num_of_tests = len(Validation_data)

def test_data_training(n_epoch):
    
    for epoch in range(n_epoch):
        for x,y in train_data_loader:
            optimizer.zero_grad()
            z = model(x.view(-1, 28*28))
            loss_item = criterion(z, y)
            loss_item.backward()
            optimizer.step()
        
        correct = 0
        for x_test, y_test in validation_data_loader:
            z = model(x_test.view(-1, 28*28))
            _, yhat = torch.max(z.data, dim = 1)
            correct += (yhat == y_test).sum().item()
        
        accuracy_item = correct/Num_of_tests
        loss.append(loss_item.data)
        accuracy.append(accuracy_item)

test_data_training(epochs)

def plot(model):
    W = model.state_dict()['linear.weight'].data
    w_min = W.min().item()
    w_max = W.max().item()
    fig, axes = plt.subplots(2,5)
    fig.subplots_adjust(hspace = 0.01, wspace = 0.1)
    for i, ax in enumerate(axes.flat):
        if i < 10 :

            ax.set_xlabel("Class : {0}".format(i))
            ax.imshow(W[i,:].view(28, 28), vmin = w_min, vmax = w_max, cmap = 'gray')

            ax.set_xticks([])
            ax.set_yticks([])

    plt.show()


def show_data(data_sample):
    plt.imshow(data_sample[0].numpy().reshape(28, 28), cmap = 'gray')
    plt.show()




fig, ax1 = plt.subplots()
color = 'tab:green'
ax1.plot(loss, color = color)
ax1.set_xlabel('epoch', color = color)
ax1.set_ylabel('total loss',color=color)
ax1.tick_params(axis='y', color=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('accuracy', color=color)
ax2.plot( accuracy, color=color)
ax2.tick_params(axis='y', color=color)
fig.tight_layout()

plot(model)


(x_train, y_train), (_, _) = keras.datasets.fashion_mnist.load_data()
x_train = x_train[:3000]
y_train = y_train[:3000]



x_mnist = np.reshape(x_train, [x_train.shape[0], x_train.shape[1] * x_train.shape[2]])


tsne = TSNE(n_components = 2, verbose = 1, random_state = 123)
z = tsne.fit_transform(x_mnist)
df = pd.DataFrame()
df["y"] = y_train
df["component-1"] = z[:,0]
df["component-2"] = z[:,1]

sn.scatterplot(x = "component-1", y = "component-2", hue = df.y.tolist(), palette = sn.color_palette("hls", 10), data = df).set(title = "Fashion-MNIST data Cluster visualization")

plt.show()