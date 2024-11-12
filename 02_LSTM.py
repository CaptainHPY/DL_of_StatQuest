import torch
import torch.nn as nn
from torch.optim import Adam

from torch.utils.data import DataLoader, TensorDataset

class LSTM(nn.Module):

    def __init__(self):

        super().__init__()

        mean = torch.tensor(0.0)
        std = torch.tensor(1.0)

        self.wlr1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wlr2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.blr1 = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        self.wpr1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wpr2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.bpr1 = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        self.wp1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wp2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.bp1 = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        self.wo1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wo2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.bo1 = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def lstm_unit(self, input_value, long_memory, short_memory):

        long_remember_percent = torch.sigmoid(self.wlr1 * short_memory + self.wlr2 * input_value + self.blr1)

        potential_remember_percent = torch.sigmoid(self.wpr1 * short_memory + self.wpr2 * input_value + self.bpr1)
        potential_memory = torch.tanh(self.wp1 * short_memory + self.wp2 * input_value + self.bp1)

        updated_long_memory = long_remember_percent * long_memory + potential_remember_percent * potential_memory

        output_percent = torch.sigmoid(self.wo1 * short_memory + self.wo2 * input_value + self.bo1)

        updated_short_memory = output_percent * torch.tanh(updated_long_memory)

        return ([updated_short_memory, updated_long_memory])
    
    def forward(self, input):

        long_memory = 0
        short_memory = 0
        day1 = input[0]
        day2 = input[1]
        day3 = input[2]
        day4 = input[3]

        long_memory, short_memory = self.lstm_unit(day1, long_memory, short_memory)
        long_memory, short_memory = self.lstm_unit(day2, long_memory, short_memory)
        long_memory, short_memory = self.lstm_unit(day3, long_memory, short_memory)
        long_memory, short_memory = self.lstm_unit(day4, long_memory, short_memory)

        return short_memory
    
    def configure_optimizers(self):

        return Adam(self.parameters(), lr=0.1)
    
    def training_step(self, batch):

        input_i, label_i = batch
        output_i = self.forward(input_i[0].to(device))
        label_i = label_i.to(device)
        loss = (output_i - label_i) ** 2

        return loss, label_i
    
model = LSTM()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

print("Now let's compare the observed and predicted values...")

print("Company A: Observed = 0, Predicted =", model(torch.tensor([0., 0.5, 0.25, 1.]).to(device)).detach())

print("Company B: Observed = 1, Predicted =", model(torch.tensor([1., 0.5, 0.25, 1.]).to(device)).detach())

inputs = torch.tensor([[0., 0.5, 0.25, 1.], [1., 0.5, 0.25, 1.]]).to(device)
labels = torch.tensor([0., 1.]).to(device)

dataset = TensorDataset(inputs, labels)
dataloader = DataLoader(dataset)

optimizer = model.configure_optimizers()

for epoch in range(300):

    for batch in dataloader:

        loss, label_i = model.training_step(batch)
        loss.backward()

        if(epoch % 10 == 0):
            if(label_i == 0):
                print("Step: " + str(epoch) + " Loss of Company A: " + str(loss))
            else:
                print("Step: " + str(epoch) + " Loss of Company B: " + str(loss))

    optimizer.step()
    optimizer.zero_grad()

print("Now let's compare the observed and predicted values again...")

print("Company A: Observed = 0, Predicted =", model(torch.tensor([0., 0.5, 0.25, 1.]).to(device)).detach())

print("Company B: Observed = 1, Predicted =", model(torch.tensor([1., 0.5, 0.25, 1.]).to(device)).detach())
