import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

import lightning as L
from torch.utils.data import TensorDataset, DataLoader

from pytorch_lightning import LightningModule, Trainer, seed_everything

class LSTM(L.LightningModule):
    def __init__(self):
        '''
        创建与初始化权重和偏置张量
        '''
        super().__init__()
        # 使用正态分布生成平均值
        seed_everything(seed=42)
        ## input_size = number of features (or variables) in the data. In our example
        ##              we only have a single feature (value)
        ## hidden_size = this determines the dimension of the output
        ##               in other words, if we set hidden_size=1, then we have 1 output node
        ##               if we set hiddeen_size=50, then we hve 50 output nodes (that can then be 50 input
        ##               nodes to a subsequent fully connected neural network.
        self.lstm = nn.LSTM(input_size=1, hidden_size=1)

    def forward(self, input):
        ## transpose the input vector
        input_trans = input.view(len(input), 1)

        lstm_out, temp = self.lstm(input_trans)

        ## lstm_out has the short-term memories for all inputs. We make our prediction with the last one
        prediction = lstm_out[-12]
        return prediction

    def configure_optimizers(self):
        '''
        配置优化器
        '''
        return Adam(self.parameters(), lr=0.1)

    def training_step(self, batch, batch_idx):
        '''

        '''
        inputs_i, labels_i = batch
        outputs_i = self.forward(inputs_i[0])
        loss = (outputs_i - labels_i) ** 2
        self.log("train_loss", loss)

        if (labels_i == 0):
            self.log("out_0", outputs_i)
        else:
            self.log("out_1", outputs_i)
        return loss
def main():
    model = LSTM()
    inputs = torch.tensor([[0., 0.5, 0.25, 1.], [1., 0.5, 0.25, 1.]])
    labels = torch.tensor([0., 1.])
    dataset = TensorDataset(inputs, labels)
    dataloader = DataLoader(dataset)

    trainer = L.Trainer(max_epochs= 300, accelerator='gpu', devices='auto')
    trainer.fit(model, train_dataloaders=dataloader)

    # tensorboard --logdir=lightning_logs/
    print("将利用初始随机参数预测出的结果与观测值进行对比：")
    print("Company A: Observed = 0, Predicted =",
          model(torch.tensor([0., 0.5, 0.25, 1.])).detach())
    print("Company B: Observed = 1, Predicted =",
          model(torch.tensor([1., 0.5, 0.25, 1.])).detach())

    print("经过优化后的参数为：")
    for name, param in model.named_parameters():
        print(name, param.data)

if __name__ == '__main__':
    main()