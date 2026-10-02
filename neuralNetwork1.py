import torch
from torch import nn
from torch.utils.data import DataLoader
from lorenzSystem import System
import numpy as np

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_tanh_stack = nn.Sequential(
            nn.Linear(4, 128),
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
system = System(10, 28, 8/3)

def train(model, loss_fn, optimizer):

    #sets the model to training mode
    model.train()

    values, times = system.values(100)
    xarray, yarray, zarray = values

    for i in range(1,len(times)):

        #gather inputs and target outputs (inputs: previous steps x, y, z, and current time. Outputs: current x, y, z)
        x_gt, y_gt, z_gt = xarray[i], yarray[i], zarray[i]
        x_in, y_in, z_in = xarray[i-1], yarray[i-1], zarray[i-1]
        t_in = times[i]

        X = torch.tensor([[x_in, y_in, z_in, t_in]], dtype=torch.float32)
        y = torch.tensor([[x_gt, y_gt, z_gt]], dtype=torch.float32)

        X, y  = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def getPredictionsFullData(x_gt, y_gt, z_gt, times):
    x_pred = np.array([0])
    y_pred = np.array([0])
    z_pred = np.array([0])
    for i in range(1, len(times)):
        input = torch.tensor([[x_gt[i - 1], y_gt[i - 1], z_gt[i - 1], times[i]]], dtype=torch.float32)
        results = model.forward(input).detach().numpy()
        x_pred = np.append(x_pred, results[0][0])
        y_pred = np.append(y_pred, results[0][1])
        z_pred = np.append(z_pred, results[0][2])
    return x_pred, y_pred, z_pred

try:
    model.load_state_dict(torch.load('model.pth', weights_only=True))
    print('Model found, and loaded.')
except:
    print('No model found, making new one.')

if __name__ == '__main__':

    #if running this program, train.
    epochs = 2000
    for t in range(epochs):
        print(f'Epoch {t}')
        train(model, lossFunction, optimizer)


    torch.save(model.state_dict(), 'model.pth')



