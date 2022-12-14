import sys
sys.path.append('../src')
sys.path.append('../data')
import numpy as np
import scipy.io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data

torch.manual_seed(1)

## Train the error model using direct data
data = np.loadtxt('../data/L96_c_10_direct.dat')
input_training = torch.from_numpy(data[::100,0].reshape(-1,1))
input_training = input_training.float()
output_training = torch.from_numpy(data[::100,1].reshape(-1,1))
output_training = output_training.float()
x, y = Variable(input_training), Variable(output_training)

net = torch.nn.Sequential(
        torch.nn.Linear(1, 5),
        torch.nn.Sigmoid(),
        torch.nn.Linear(5, 5),
        torch.nn.Sigmoid(),
        torch.nn.Linear(5, 1),
    )

optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

BATCH_SIZE = 16
EPOCH = 100

torch_dataset = Data.TensorDataset(x, y)

loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True, num_workers=2,)

# start training
for epoch in range(EPOCH):
    print("Epoch " + str(epoch))
    for step, (batch_x, batch_y) in enumerate(loader): # for each training step
        b_x = Variable(batch_x)
        b_y = Variable(batch_y)
        prediction = net(b_x)     # input x and predict based on x
        loss = loss_func(prediction, b_y)
        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients

torch.save(net, 'L96_c_10_direct_trained.pt')

## Plot the trained model
plt.rcParams.update({'font.size': 16})
plt.figure()
plt.plot(x.data.numpy(), y.data.numpy(), 'o', color = "#e31a1c", alpha=0.2, label='Training data')
prediction = net(x)     # input x and predict based on x
x_nn, y_nn = zip(*sorted(zip(x.data.numpy(), prediction.data.numpy())))
plt.plot(x_nn, y_nn, '-', color='#1f78b4', lw=2, alpha=1.0, label='Neural network')
plt.xlabel('X')
plt.ylabel(r'$\delta(X)$')
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig('L96_c_10_direct_trained.pdf')
plt.close()
