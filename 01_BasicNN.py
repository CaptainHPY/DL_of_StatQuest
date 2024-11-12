import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD

import matplotlib.pyplot as plt
import seaborn as sns

class BasicNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.w00 = nn.Parameter(torch.tensor(1.7), requires_grad=False)
        self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad=False)
        self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad=False)

        self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad=False)
        self.b10 = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.w11 = nn.Parameter(torch.tensor(2.7), requires_grad=False)

        self.final_bias = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def forward(self, input):

        input_to_top_relu = input * self.w00 + self.b00
        top_relu_out = F.relu(input_to_top_relu)
        scaled_top_relu_output = top_relu_out * self.w01

        input_to_bottom_relu = input * self.w10 + self.b10
        bottom_relu_output = F.relu(input_to_bottom_relu)
        scaled_bottom_relu_output = bottom_relu_output * self.w11

        input_to_final_relu = scaled_top_relu_output + scaled_bottom_relu_output + self.final_bias

        output = F.relu(input_to_final_relu)

        return output
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

inputs = torch.tensor([0., 0.5, 1.]).to(device)
labels = torch.tensor([0., 1., 0.]).to(device)

model = BasicNN()

model.to(device)

optimizer = SGD([model.final_bias], lr=0.1)

print("Final bias before training: " + str(model.final_bias.data) + "\n")

for epoch in range(100):

    total_loss = 0

    for iteration in range(len(inputs)):

        input_i = inputs[iteration]
        label_i = labels[iteration]

        output_i = model(input_i)

        loss = (output_i - label_i) ** 2

        loss.backward()

        total_loss += float(loss)

    if(total_loss < 0.0001):
        print("Num steps: " + str(epoch))
        break

    optimizer.step()
    optimizer.zero_grad()

    print("Step: " + str(epoch) + " Final bias: " + str(model.final_bias.data) + "\n")

print("Final bias after optimization: " + str(model.final_bias.data))

input_doses = torch.linspace(start=0, end=1, steps=11).to(device)
print(input_doses)

output_values = model(input_doses)

sns.set(style="whitegrid")

sns.lineplot(x=input_doses.cpu(), y=output_values.cpu().detach(), color="green", linewidth=2.5)

plt.xlabel("Dose")
plt.ylabel("Effectiveness")
plt.show()
