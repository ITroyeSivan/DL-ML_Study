import torch
import torch.nn as nn

from torch.optim import Adam
from torch.distributions.uniform import Uniform
from torch.utils.data import TensorDataset, DataLoader

import lightning as L

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class WordEmbedding(L.LightningModule):

    def __init__(self):

        super().__init__()

        L.seed_everything(seed=42)

        min_value = -0.5
        max_value = 0.5

        self.input1_w1 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.input1_w2 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.input2_w1 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.input2_w2 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.input3_w1 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.input3_w2 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.input4_w1 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.input4_w2 = nn.Parameter(Uniform(min_value, max_value).sample())

        self.output1_w1 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.output1_w2 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.output2_w1 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.output2_w2 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.output3_w1 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.output3_w2 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.output4_w1 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.output4_w2 = nn.Parameter(Uniform(min_value, max_value).sample())

        self.loss = nn.CrossEntropyLoss()

    def forward(self, input):

        input = input[0]

        inputs_to_top_hidden = ((input[0] * self.input1_w1) +
                                (input[1] * self.input2_w1) +
                                (input[2] * self.input3_w1) +
                                (input[3] * self.input4_w1))

        inputs_to_bottom_hidden = ((input[0] * self.input1_w2) +
                                   (input[1] * self.input2_w2) +
                                   (input[2] * self.input3_w2) +
                                   (input[3] * self.input4_w2))

        output1 = ((inputs_to_top_hidden * self.output1_w1) +
                   (inputs_to_bottom_hidden * self.output1_w2))
        output2 = ((inputs_to_top_hidden * self.output2_w1) +
                   (inputs_to_bottom_hidden * self.output2_w2))
        output3 = ((inputs_to_top_hidden * self.output3_w1) +
                   (inputs_to_bottom_hidden * self.output3_w2))
        output4 = ((inputs_to_top_hidden * self.output4_w1) +
                   (inputs_to_bottom_hidden * self.output4_w2))

        output_presoftmax = torch.stack([output1, output2, output3, output4])

        return(output_presoftmax)


    def configure_optimizers(self):

        return Adam(self.parameters(), lr=0.1)

    def training_step(self, batch, batch_idx):

        input_i, labels_i = batch
        output_i = self.forward(input_i)
        loss = self.loss(output_i, labels_i[0])

        return loss

def main():
    inputs = torch.tensor([[1., 0., 0., 0.],
                           [0., 1., 0., 0.],
                           [0., 0., 1., 0.],
                           [0., 0., 0., 1.]])
    labels = torch.tensor([[0., 1., 0., 0.],
                           [0., 0., 1., 0.],
                           [0., 0., 0., 1.],
                           [0., 1., 0., 0.]])
    dataset = TensorDataset(inputs, labels)
    dataloader = DataLoader(dataset)

    model = WordEmbedding()

    print("初始参数为：")
    data = {
        "w1": [model.input1_w1.item(),
               model.input2_w1.item(),
               model.input3_w1.item(),
               model.input4_w1.item()
               ],
        "w2": [model.input1_w2.item(),
               model.input2_w2.item(),
               model.input3_w2.item(),
               model.input4_w2.item()],
        "token": ["Troll2", "is", "great", "Gymkata"],
        "input": ["input1", "input2", "input3", "input4"]
    }
    df = pd.DataFrame(data)
    print(df)
    # sns.scatterplot(data=df, x="w1", y="w2")
    # for i in range(4):
    #     plt.text(df.w1[i], df.w2[i], df.token[i],
    #              horizontalalignment='left',
    #              size='medium',
    #              color='black',
    #              weight='semibold')
    # plt.show()

    trainer = L.Trainer(max_epochs=100)
    trainer.fit(model, train_dataloaders=dataloader)

    print("训练后的参数为：")
    data = {
        "w1": [model.input1_w1.item(),
               model.input2_w1.item(),
               model.input3_w1.item(),
               model.input4_w1.item()
               ],
        "w2": [model.input1_w2.item(),
               model.input2_w2.item(),
               model.input3_w2.item(),
               model.input4_w2.item()],
        "token": ["Troll2", "is", "great", "Gymkata"],
        "input": ["input1", "input2", "input3", "input4"]
    }
    df = pd.DataFrame(data)
    print(df)
    sns.scatterplot(data=df, x="w1", y="w2")
    for i in range(len(df.w1)):
        plt.text(df.w1[i], df.w2[i], df.token[i],
                 horizontalalignment='left',
                 size='medium',
                 color='black',
                 weight='semibold')
    plt.show()

    softmax = nn.Softmax(dim=0)
    print(torch.round(softmax(model(torch.tensor([[1., 0., 0., 0.]]))),
          decimals=2)) # torch.round对张量中每个元素进行四舍五入，小数点两位

if __name__ == '__main__':
    main()