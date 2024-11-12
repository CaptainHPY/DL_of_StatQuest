import torch
import torch.nn as nn

from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class WordEmbedding(nn.Module):

    def __init__(self):

        super().__init__()

        self.input_to_hidden = nn.Linear(in_features=4, out_features=2, bias=False)
        self.hidden_to_output = nn.Linear(in_features=2, out_features=4, bias=False)

        self.loss = nn.CrossEntropyLoss()

    def forward(self, input):

        hidden = self.input_to_hidden(input)
        output_values = self.hidden_to_output(hidden)

        return output_values
    
    def configure_optimizers(self):

        return Adam(self.parameters(), lr=0.1)
    
    def training_step(self, batch):

        inputs_i, labels_i = batch
        labels_i = labels_i.to(device)
        outputs_i = self.forward(inputs_i).to(device)
        loss = self.loss(outputs_i, labels_i)

        return loss

model = WordEmbedding()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

optimizer = model.configure_optimizers()

data = {
    "w1": model.input_to_hidden.weight.detach().cpu()[0].numpy(),
    "w2": model.input_to_hidden.weight.detach().cpu()[1].numpy(),
    "token": ["Troll2", "is", "great", "Gymkata"],
    "input": ["input1", "input2", "input3", "input4"]
}
df = pd.DataFrame(data)
print(df)

sns.scatterplot(data=df, x="w1", y="w2")

plt.text(df.w1[0], df.w2[0], df.token[0], 
         horizontalalignment='left', 
         size='medium', 
         color='black', 
         weight='semibold')

plt.text(df.w1[1], df.w2[1], df.token[1], 
         horizontalalignment='left', 
         size='medium', 
         color='black', 
         weight='semibold')

plt.text(df.w1[2], df.w2[2], df.token[2], 
         horizontalalignment='left', 
         size='medium', 
         color='black', 
         weight='semibold')

plt.text(df.w1[3], df.w2[3], df.token[3], 
         horizontalalignment='left', 
         size='medium', 
         color='black', 
         weight='semibold')

plt.show()

inputs = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]).to(device)

labels = torch.tensor([[0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.], [0., 1., 0., 0.]]).to(device)

dataset = TensorDataset(inputs, labels)
dataloader = DataLoader(dataset)

for epoch in range(100):

    for batch in dataloader:

        loss = model.training_step(batch)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        
data = {
    "w1": model.input_to_hidden.weight.detach().cpu()[0].numpy(),
    "w2": model.input_to_hidden.weight.detach().cpu()[1].numpy(),
    "token": ["Troll2", "is", "great", "Gymkata"],
    "input": ["input1", "input2", "input3", "input4"]
}
df = pd.DataFrame(data)
print(df)

sns.scatterplot(data=df, x="w1", y="w2")

plt.text(df.w1[0]-0.2, df.w2[0]+0.1, df.token[0], 
         horizontalalignment='left', 
         size='medium', 
         color='black', 
         weight='semibold')

plt.text(df.w1[1], df.w2[1], df.token[1], 
         horizontalalignment='left', 
         size='medium', 
         color='black', 
         weight='semibold')

plt.text(df.w1[2], df.w2[2], df.token[2], 
         horizontalalignment='left', 
         size='medium', 
         color='black', 
         weight='semibold')

plt.text(df.w1[3]-0.3, df.w2[3]-0.1, df.token[3], 
         horizontalalignment='left', 
         size='medium', 
         color='black', 
         weight='semibold')

plt.show()

softmax = nn.Softmax(dim=0)

print(torch.round(softmax(model(torch.tensor([1., 0., 0., 0.]).to(device))), decimals=2))

print(torch.round(softmax(model(torch.tensor([0., 1., 0., 0.]).to(device))), decimals=2))

print(torch.round(softmax(model(torch.tensor([0., 0., 1., 0.]).to(device))), decimals=2))

print(torch.round(softmax(model(torch.tensor([0., 0., 0., 1.]).to(device))), decimals=2))
