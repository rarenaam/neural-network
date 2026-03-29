import math
import random

data = [0.5, 0.8]

class Raw_network:
  def __init__(self):
    self.w_hidden = [[random.uniform(-1, 1) for _ in range(2)] for _ in range (5)]
    self.b_hidden = [random.uniform(-1, 1) for _ in range(5)]

    self.w_output = [random.uniform(-1, 1) for _ in range(5)]
    self.b_output = random.uniform(-1, 1)

  def forward(self, n):
    hidden_outputs = []
    for i in range(5):
      z = (self.w_hidden[i][0] * n[0]) + (self.w_hidden[i][1] * n[1]) + self.b_hidden[i]
      hidden_outputs.append(max(0, z))

    total = 0
    for i in range(5):
      total += hidden_outputs[i] * self.w_output[i]

    y_out = total + self.b_output
    return y_out

model = Raw_network()
print(model.forward(data))
