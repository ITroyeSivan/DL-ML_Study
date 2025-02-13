import torch
import torch.nn as nn

from torch.optim import Adam
from torch.distributions.uniform import Uniform
from torch.utils.data import TensorDataset, DataLoader

import lightning as L

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class WordEmbeddingWithLinear(L.LightningModule):

    def __init__(self):

        super().__init__()

        L.seed_everything(seed=42)

        # nn.Linear()为隐藏层中的两个节点分别创建4个权重
        self.input_to_hidden = nn.Linear(in_features=4, out_features=2, bias=False)
        self.hidden_to_output = nn.Linear(in_features=2, out_features=4, bias=False)

        self.loss = nn.CrossEntropyLoss()

    def forward(self, input):

        hidden = self.input_to_hidden(input)
        output_values = self.hidden_to_output(hidden)

        return(output_values)


    def configure_optimizers(self):

        return Adam(self.parameters(), lr=0.1)

    def training_step(self, batch, batch_idx):

        input_i, labels_i = batch
        output_i = self.forward(input_i)
        loss = self.loss(output_i, labels_i)

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

    model = WordEmbeddingWithLinear()

    print("初始参数为：")
    data = {
        "w1": model.input_to_hidden.weight.detach()[0].numpy(), # 使用nn.Linear()创造权重，所以我们通过weight访问它们
        "w2": model.input_to_hidden.weight.detach()[1].numpy(),
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
        "w1": model.input_to_hidden.weight.detach()[0].numpy(),  # 使用nn.Linear()创造权重，所以我们通过weight访问它们
        "w2": model.input_to_hidden.weight.detach()[1].numpy(),
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

    softmax = nn.Softmax(dim=1) # dim=0将sofmax应用于行，dim=1将sofmax应用于列
    print(torch.round(softmax(model(torch.tensor([[1., 0., 0., 0.]]))),
          decimals=2)) # torch.round对张量中每个元素进行四舍五入，小数点两位

    # 使用nn.Embedding()来加载和使用预训练的词嵌入值
    # 首先创建nn.Embedding类，并通过.from_pretrained将预训练的权重传递过去
    # nn.Embedding希望权重以列的方式存在，.T将行转为列
    word_embeddings = nn.Embedding.from_pretrained(model.input_to_hidden.weight.T)
    print(word_embeddings(torch.tensor(0)))
    # 或者也可以通过将token映射到其索引的字典来简化这个过程
    vocab = {
        'Troll2': 0,
        'is': 1,
        'great': 2,
        'Gymkata': 3
    }
    print(word_embeddings(torch.tensor(vocab['Troll2'])))

if __name__ == '__main__':
    main()