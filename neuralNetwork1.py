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
            nn.Linear(63, 128),
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

def prepareInputData(xarray, yarray, zarray, index, precision, backCount = 16, pointsPerInteger = 10, curveCount = 5):
    x_in, y_in, z_in = [], [], []
    for j in range(1, backCount + 1):
        x_in.append(xarray[index - (j * int(precision/pointsPerInteger))])
        y_in.append(yarray[index - (j * int(precision/pointsPerInteger))])
        z_in.append(zarray[index - (j * int(precision/pointsPerInteger))])
    for k in range(1, curveCount + 1):
        x_in.append(xarray[index - k])
        y_in.append(yarray[index - k])
        z_in.append(zarray[index - k])

    X = torch.tensor([[*x_in, *y_in, *z_in]], dtype=torch.float32)

    X = X.to(device)

    return X


def train(model, trainingData):

    #sets the model to training mode
    model.train()

    loss_fn = lossFunction
    optimizer = optimizerFunction

    xarray, yarray, zarray, times = trainingData

    for i in range(160,len(times),10):
        X = prepareInputData(xarray, yarray, zarray, i, 100)
        #print(X)
        x_gt, y_gt, z_gt = xarray[i], yarray[i], zarray[i]
        y = torch.tensor([[x_gt, y_gt, z_gt]], dtype=torch.float32)
        y = y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)
        #print(f'Pred: {pred}, actual: {y}, loss: {loss}')

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def getPredictionsFullData(x_gt, y_gt, z_gt):
    x_pred = np.full(160, np.nan).astype(float)
    y_pred = np.full(160, np.nan).astype(float)
    z_pred = np.full(160, np.nan).astype(float)
    for i in range(160, len(x_gt)):
        input = prepareInputData(x_gt, y_gt, z_gt, i, 100)
        results = model.forward(input).detach().numpy()
        x_pred = np.append(x_pred, results[0][0])
        y_pred = np.append(y_pred, results[0][1])
        z_pred = np.append(z_pred, results[0][2])
    return x_pred, y_pred, z_pred

def getRecrusivePrediction(x_gt, y_gt, z_gt, lengthOfOutput, keepOriginal = False):
    x, y, z = x_gt, y_gt, z_gt
    for i in range(len(x_gt), lengthOfOutput):
        input = prepareInputData(x, y, z, i, 100)
        results = model.forward(input).detach().numpy()
        x = np.append(x, results[0][0])
        y = np.append(y, results[0][1])
        z = np.append(z, results[0][2])
    if not keepOriginal:
        x = np.concatenate((np.full(len(x_gt), np.nan).astype(float), x[len(x_gt):]))
        y = np.concatenate((np.full(len(y_gt), np.nan).astype(float), y[len(y_gt):]))
        z = np.concatenate((np.full(len(z_gt), np.nan).astype(float), z[len(z_gt):]))
    return x, y, z

try:
    model.load_state_dict(torch.load('model.pth', weights_only=True))
    print('Model found, and loaded.')
except:
    print('No model found, making new one.')



