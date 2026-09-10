import torch
from torch import nn
from torch.utils.data import DataLoader
from lorenzSystem import System

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_tanh_stack = nn.Sequential(
            nn.Linear(1, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 3)
        )
    def forward(self, x):
        return self.linear_tanh_stack(x)

model = NeuralNetwork().to(device)

lossFunction = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
system = System(10, 28, 8/3)

def train():
    times, positions = system.values(500)
    for x, y in times, positions:
        x, y = x.to(device), y.to(device)

        pred = model(x)
        loss = lossFunction(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

if __name__ == '__main__':
    try:
        model.load_state_dict(torch.load('model.pth', weights_only=True))
    except:
        pass

    epochs = 5
    for t in range(epochs):
        train()

    torch.save(model.state_dict(), 'model.pth')



