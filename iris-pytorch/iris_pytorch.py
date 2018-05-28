import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3, hl)
        self.fc2 = nn.Linear(hl, hl)
        self.fc3 = nn.Linear(hl, 3)

    def forward(self, x):
        global xtrained
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        xtrained = x
        return x

def load(file_name):
    data = pd.read_csv(file_name)
    data = data[['sepal_length', 'sepal_width', 'petal_width', 'species']]
    data.loc[data['species']=='Iris-setosa', 'species']=0
    data.loc[data['species']=='Iris-versicolor', 'species']=1
    data.loc[data['species']=='Iris-virginica', 'species']=2
    data = data.apply(pd.to_numeric)
    return data

def train(epoch):
    X = Variable(torch.Tensor(xtrain).float())
    Y = Variable(torch.Tensor(ytrain).long())

    optimizer.zero_grad()
    out = net(X)
    loss = criterion(out, Y)
    loss.backward()
    optimizer.step()

    if (epoch) % 50 == 0:
        print ('Epoch [%d/%d] Loss: %.4f' %(epoch+1, num_epoch, loss.data))

# training data
datatrain = load('iris_train.csv')
datatrain_array = datatrain.as_matrix()
xtrain = datatrain_array[:,:3]
ytrain = datatrain_array[:,3]
# test data
datatest = load('iris_test.csv')
datatest_array = datatest.as_matrix()
xtest = datatest_array[:,:3]
ytest = datatest_array[:,3]

xtrained = None

torch.manual_seed(1234)

hl = 3
num_epoch = 10000

net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

for epoch in range(num_epoch):
    train(epoch)

xtrained_backup = xtrained.detach().numpy()

X = Variable(torch.Tensor(xtest).float())
Y = torch.Tensor(ytest).long()
out = net(X)
_, predicted = torch.max(out.data, 1)

print('Accuracy of the network %d %%' % (100 * torch.sum(Y==predicted) / 30))

fig = plt.figure()
ax1 = fig.add_subplot(211, projection='3d')
ax2 = fig.add_subplot(212, projection='3d')

ax1.scatter(datatrain['sepal_length'], datatrain['sepal_width'], datatrain['petal_width'], c=ytrain)
ax2.scatter(xtrained_backup[:,0], xtrained_backup[:,1], xtrained_backup[:,2], c=ytrain)

plt.show()