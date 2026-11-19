import torch
from torch import nn
import numpy as np

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
print(f'Using {device}.')

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_tanh_stack = nn.Sequential(
            nn.Linear(3, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
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

lossFunction = nn.L1Loss() #L1Loss is Mean Absolute Error loss function
optimizerFunction = torch.optim.SGD(model.parameters(), lr=1e-4)


def train(model, trainingData):

    #sets the model to training mode
    model.train()

    loss_fn = lossFunction
    optimizer = optimizerFunction

    xarray, yarray, zarray, times = trainingData

    for i in range(1,len(times), 10):
        X = torch.tensor([[xarray[i-1], yarray[i-1], zarray[i-1]]], dtype=torch.float32)
        X = X.to(device)

        x_gt, y_gt, z_gt = xarray[i], yarray[i], zarray[i]
        y = torch.tensor([[x_gt, y_gt, z_gt]], dtype=torch.float32)
        y = y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)
        #print(f'Pred: {pred}, actual: {y}, loss: {loss}')

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def getPredictionsFullData(x_gt, y_gt, z_gt, times):
    precision = int(1/(times[1] - times[0]))
    x_pred = np.array([])
    y_pred = np.array([])
    z_pred = np.array([])
    for i in range(1, len(x_gt)):
        X = torch.tensor([[x_gt[i-1], y_gt[i-1], z_gt[i-1]]], dtype=torch.float32)
        X = X.to(device)
        results = model.forward(X).detach().numpy()
        x_pred = np.append(x_pred, results[0][0])
        y_pred = np.append(y_pred, results[0][1])
        z_pred = np.append(z_pred, results[0][2])
    outputTimes = times[1:]
    return x_pred, y_pred, z_pred, outputTimes

def getRecrusivePrediction(x_gt, y_gt, z_gt, times, lengthOfOutput, keepOriginal = False):
    precision = int(1 / (times[1] - times[0]))
    x, y, z = x_gt, y_gt, z_gt
    for i in range(len(x_gt), lengthOfOutput+len(x_gt)):
        X = torch.tensor([[x[-1], y[-1], z[-1]]], dtype=torch.float32)
        X = X.to(device)
        results = model.forward(X).detach().numpy()
        x = np.append(x, results[0][0])
        y = np.append(y, results[0][1])
        z = np.append(z, results[0][2])
    if keepOriginal:
        newTimes = times
    else:
        x = x[len(x_gt):]
        y = y[len(y_gt):]
        z = z[len(z_gt):]
        newTimes = np.linspace(times[-1]+(1/precision), times[-1] + (lengthOfOutput/precision), lengthOfOutput) #endpoint = False to keep values spaced correctly
    return x, y, z, newTimes

def save():
    torch.save(model.state_dict(), 'model2.pth')

try:
    model.load_state_dict(torch.load('model2.pth', weights_only=True))
    print('Model found, and loaded.')
except:
    print('No model found, making new one.')



