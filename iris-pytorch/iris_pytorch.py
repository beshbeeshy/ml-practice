import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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

# training data
datatrain = pd.read_csv('iris_train.csv')
datatrain = datatrain[['sepal_length', 'sepal_width', 'petal_width', 'species']]
datatrain.loc[datatrain['species']=='Iris-setosa', 'species']=0
datatrain.loc[datatrain['species']=='Iris-versicolor', 'species']=1
datatrain.loc[datatrain['species']=='Iris-virginica', 'species']=2
datatrain = datatrain.apply(pd.to_numeric)
datatrain_array = datatrain.as_matrix()
xtrain = datatrain_array[:,:3]
ytrain = datatrain_array[:,3]
# test data
datatest = pd.read_csv('iris_test.csv')
datatest = datatest[['sepal_length', 'sepal_width', 'petal_width', 'species']]
datatest.loc[datatest['species']=='Iris-setosa', 'species']=0
datatest.loc[datatest['species']=='Iris-versicolor', 'species']=1
datatest.loc[datatest['species']=='Iris-virginica', 'species']=2
datatest = datatest.apply(pd.to_numeric)
datatest_array = datatest.as_matrix()
xtest = datatest_array[:,:3]
ytest = datatest_array[:,3]

xtrained = None

torch.manual_seed(1234)

hl = 3
num_epoch = 10000

def train(epoch):
    X = Variable(torch.Tensor(xtrain).float())
    Y = Variable(torch.Tensor(ytrain).long())

    optimizer.zero_grad()
    out = net(X)
    loss = criterion(out, Y)
    loss.backward()
    optimizer.step()

    if (epoch) % 50 == 0:
        print ('Epoch [%d/%d] Loss: %.4f' 
            %(epoch+1, num_epoch, loss.data))

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

ax1.scatter(
    datatrain['sepal_length'],
    datatrain['sepal_width'],
    datatrain['petal_width'],
    c=ytrain
)

ax2.scatter(
    xtrained_backup[:,0],
    xtrained_backup[:,1],
    xtrained_backup[:,2],
    c=ytrain
)

plt.show()