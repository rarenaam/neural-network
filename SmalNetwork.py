import torch
import torch.nn as nn

class Network(nn.Module):
  def __init__(self):
    super(Network, self).__init__()

    self.hidden = nn.Linear(2, 5)

    self.relu = nn.ReLU()

    self.output = nn.Linear(5, 1)

  def forward(self, x):
    x = self.hidden(x)
    x = self.relu(x)
    x = self.output(x)
    return x

model = Network()
print(model)
