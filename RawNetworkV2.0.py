import time
import random


class Raw_network:
    def __init__(self):
        self.w_hidden = [[random.uniform(-1, 1) for _ in range(3)] for _ in range(30)]
        self.b_hidden = [random.uniform(-1, 1) for _ in range(30)]

        self.w_output = [random.uniform(-1, 1) for _ in range(30)]
        self.b_output = random.uniform(-1, 1)

    def forward(self, n):
        hidden_outputs = []
        for i in range(30):
            z = (self.w_hidden[i][0] * n[0]) + (self.w_hidden[i][1] * n[1]) + (self.w_hidden[i][2] * n[2]) + self.b_hidden[i]
            hidden_outputs.append(max(0, z))

        total = 0
        for i in range(30):
            total += hidden_outputs[i] * self.w_output[i]

        y_out = total + self.b_output
        return y_out

    def train(self, n, target, learning_rate=0.00005):
        hidden_inputs = []
        hidden_outputs = []

        for i in range(30):
            z = (self.w_hidden[i][0] * n[0]) + (self.w_hidden[i][1] * n[1]) + (self.w_hidden[i][2] * n[2]) + self.b_hidden[i]
            hidden_inputs.append(z)
            hidden_outputs.append(max(0, z))

        total = 0
        for i in range(30):
            total += hidden_outputs[i] * self.w_output[i]

        prediction = total + self.b_output
        error = prediction - target
        d_output = error

        for i in range(30):
            gradient_w_out = d_output * hidden_outputs[i]
            self.w_output[i] -= learning_rate * gradient_w_out

        self.b_output -= learning_rate * d_output

        for i in range(30):
            if hidden_inputs[i] > 0:
                d_hidden = d_output * self.w_output[i]

                self.w_hidden[i][0] -= learning_rate * d_hidden * n[0]
                self.w_hidden[i][1] -= learning_rate * d_hidden * n[1]
                self.w_hidden[i][2] -= learning_rate * d_hidden * n[2]
                self.b_hidden[i] -= learning_rate * d_hidden


model = Raw_network()

choise = input("Type 'p' for pretrained or 't' for self training: ")
if choise.lower() == 'p':
    model.w_hidden = [[-0.2233787320708149, 0.08372058993517084], [-0.1547591595765616, -0.44099712627194987],
                      [-0.8346030453002322, -0.07329752419394216], [-0.023123247526250248, -0.04212844881688215],
                      [1.033207396147921, 1.0194723983112182]]
    model.b_hidden = [-0.8236753623884338, 0.0488254015807843, 0.6939556711247872, 0.6384182136955142,
                      -0.720413246448629]
    model.w_output = [-0.3347480932531945, -0.5666903086783834, -1.9764369509067372e-13, -0.6883360834039357,
                      0.9524548875967469]
    model.b_output = 1.1256074104584177
    print("\n[INFO] neurons loaded")
else:

    epoch_range = 10000

    data = []
    for i in range(500):
        a = random.uniform(0, 10)
        b = random.uniform(0, 10)
        data.append([[a, b, 0], a + b])
    for i in range(500):
        a = random.uniform(0, 10)
        b = random.uniform(0, 10)
        data.append([[a, b, 1], a * b])

    start_time = time.time()
    print(f"start training Raw_network...")
    print()

    for epoch in range(epoch_range):

        random.shuffle(data)

        for item in data:
            data_input = item[0]
            data_target = item[1]

            model.train(data_input, data_target)

        if epoch % (epoch_range / 10) == 0:
            current_prediction = model.forward(data_input)
            if data_input[2] == 1:
                print(f"step {epoch}: prediction = {data_input[0]:.7f} * {data_input[1]:.7f} = {current_prediction:.5f}")
            else:
                print(f"step {epoch}: prediction = {data_input[0]:.7f} + {data_input[1]:.7f} = {current_prediction:.5f}")
    stop_time = time.time()
    duration = stop_time - start_time

    test_sum = data[0][0]
    test_result = data[0][1]
    result = model.forward(test_sum)
    print(f"the network trained for {duration:.5f}s")
    if test_sum[2] == 1:
        print(f"\nsum:                {test_sum[0]:.2f} * {test_sum[1]:.2f} = {test_result:.2f}")
        print(f"network prediction: {test_sum[0]:.2f} * {test_sum[1]:.2f} = {result:.2f}")
    else:
        print(f"\nsum:                {test_sum[0]:.2f} + {test_sum[1]:.2f} = {test_result:.2f}")
        print(f"network prediction: {test_sum[0]:.2f} + {test_sum[1]:.2f} = {result:.2f}")

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
    PorS = int(input("give 1 for a product and a 0 for a sum"))
    User_data.append(PorS)
    User_result = model.forward(User_data)
    print(f"The network predicts {User_result:.{max_decimal}f}")
