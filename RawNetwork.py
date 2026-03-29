import time
import random

epoch_range = 10000

data = []
for i in range(1000):
    a = random.uniform(0, 10)
    b = random.uniform(0, 10)
    data.append([[a, b], a + b])


class Raw_network:
    def __init__(self):
        self.w_hidden = [[random.uniform(-1, 1) for _ in range(2)] for _ in range(5)]
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

    def train(self, n, target, learning_rate=0.001):
        hidden_inputs = []
        hidden_outputs = []

        for i in range(5):
            z = (self.w_hidden[i][0] * n[0]) + (self.w_hidden[i][1] * n[1]) + self.b_hidden[i]
            hidden_inputs.append(z)
            hidden_outputs.append(max(0, z))

        total = 0
        for i in range(5):
            total += hidden_outputs[i] * self.w_output[i]

        prediction = total + self.b_output
        error = prediction - target
        d_output = error

        for i in range(5):
            gradient_w_out = d_output * hidden_outputs[i]
            self.w_output[i] -= learning_rate * gradient_w_out

        self.b_output -= learning_rate * d_output

        for i in range(5):
            if hidden_inputs[i] > 0:
                d_hidden = d_output * self.w_output[i]

                self.w_hidden[i][0] -= learning_rate * d_hidden * n[0]
                self.w_hidden[i][1] -= learning_rate * d_hidden * n[1]
                self.b_hidden[i] -= learning_rate * d_hidden


model = Raw_network()
start_time = time.time()
print(f"start training Raw_network...")
print()

for epoch in range(epoch_range):
    for item in data:
        data_input = item[0]
        data_target = item[1]

        model.train(data_input, data_target)

    if epoch % (epoch_range/10) == 0:
        current_prediction = model.forward(data_input)
        print(f"step {epoch}: prediction = {data_input[0]:.7f} + {data_input[1]:.7f} = {current_prediction:.5f}")
stop_time = time.time()
time = stop_time - start_time

test_data = data[0][0]
test_sum = data[0][1]
result = model.forward(test_data)
print(f"the network trained for {time:.5f}s")
print(f"\nsum:                {test_data[0]:.2f} + {test_data[1]:.2f} = {test_sum:.2f}")
print(f"network prediction: {test_data[0]:.2f} + {test_data[1]:.2f} = {result:.2f}")

# user input
while True:
    max_decimal = 0
    print()
    User_data = []
    for i in range(2):
        User_input = input(f"give number {i + 1}: ")
        if "." in User_input:
            number = len(User_input.split(".")[1])
            if number > max_decimal:
                max_decimal = number
        User_data.append(float(User_input))
    User_result = model.forward(User_data)
    print(f"The network predicts {User_result:.{max_decimal}f}")
