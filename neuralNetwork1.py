import torch
from torch import nn
from lorenzSystem import System
import numpy as np

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
print(f'Using {device}.')

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_tanh_stack = nn.Sequential(
            nn.Linear(63, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 3)
        )
    def forward(self, x):
        x = self.flatten(x)
        return self.linear_tanh_stack(x)

model = NeuralNetwork().to(device)
#print(model)

#L1Loss is Mean Absolute Error loss function
lossFunction = nn.L1Loss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
system = System(10, 28, 8/3, 100)

def train(model, loss_fn, optimizer):

    #sets the model to training mode
    model.train()

    values, times = system.values(1000)
    xarray, yarray, zarray = values

    for i in range(16,len(times),10):
        x_gt, y_gt, z_gt = xarray[i], yarray[i], zarray[i]
        x_in, y_in, z_in = [], [], []
        for j in range(1,17):
            x_in.append(xarray[i - (j*10)])
            y_in.append(yarray[i - (j*10)])
            z_in.append(zarray[i - (j*10)])
        for k in range(1,6):
            x_in.append(xarray[i - k])
            y_in.append(yarray[i - k])
            z_in.append(zarray[i - k])

        X = torch.tensor([[*x_in, *y_in, *z_in]], dtype=torch.float32)
        y = torch.tensor([[x_gt, y_gt, z_gt]], dtype=torch.float32)

        X, y  = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def getPredictionsFullData(x_gt, y_gt, z_gt):
    x_pred = np.zeros(160)
    y_pred = np.zeros(160)
    z_pred = np.zeros(160)
    for i in range(160, len(x_gt)):
        x_in, y_in, z_in = [], [], []
        for j in range(1, 17):
            x_in.append(x_gt[i - (j * 10)])
            y_in.append(y_gt[i - (j * 10)])
            z_in.append(z_gt[i - (j * 10)])
        for k in range(1, 6):
            x_in.append(x_gt[i - k])
            y_in.append(y_gt[i - k])
            z_in.append(z_gt[i - k])
        input = torch.tensor([[*x_in, *y_in, *z_in]], dtype=torch.float32)
        results = model.forward(input).detach().numpy()
        x_pred = np.append(x_pred, results[0][0])
        y_pred = np.append(y_pred, results[0][1])
        z_pred = np.append(z_pred, results[0][2])
    return x_pred, y_pred, z_pred

def getRecrusivePrediction(x_gt, y_gt, z_gt, lengthOfOutput, keepOriginal = False):
    x, y, z = x_gt, y_gt, z_gt

    for i in range(len(x_gt), lengthOfOutput):
        x_in, y_in, z_in = [], [], []
        for j in range(1, 17):
            x_in.append(x[i - (j * 10)])
            y_in.append(y[i - (j * 10)])
            z_in.append(z[i - (j * 10)])
        for k in range(1, 6):
            x_in.append(x[i - k])
            y_in.append(y[i - k])
            z_in.append(z[i - k])
        input = torch.tensor([[*x_in, *y_in, *z_in]], dtype=torch.float32)
        results = model.forward(input).detach().numpy()
        x = np.append(x, results[0][0])
        y = np.append(y, results[0][1])
        z = np.append(z, results[0][2])
    if not keepOriginal:
        x = x[len(x_gt):]
        y = y[len(y_gt):]
        z = z[len(z_gt):]
    return x, y, z

try:
    model.load_state_dict(torch.load('model.pth', weights_only=True))
    print('Model found, and loaded.')
except:
    print('No model found, making new one.')

if __name__ == '__main__':

    #if running this program, train.
    epochs = 4000

    try:
        for t in range(epochs):
            print(f'Epoch {t}')
            train(model, lossFunction, optimizer)
    finally:
        torch.save(model.state_dict(), 'model.pth')



