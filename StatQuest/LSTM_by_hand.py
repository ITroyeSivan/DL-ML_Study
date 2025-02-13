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
        # 分布的平均值
        mean = torch.tensor(0.0)
        # 分布的标准差
        std = torch.tensor(1.0)

        self.wlr1 = nn.Parameter(torch.normal(mean, std), requires_grad=True)
        self.wlr2 = nn.Parameter(torch.normal(mean, std), requires_grad=True)
        self.blr1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

        self.wpr1 = nn.Parameter(torch.normal(mean, std), requires_grad=True)
        self.wpr2 = nn.Parameter(torch.normal(mean, std), requires_grad=True)
        self.bpr1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

        self.wp1 = nn.Parameter(torch.normal(mean, std), requires_grad=True)
        self.wp2 = nn.Parameter(torch.normal(mean, std), requires_grad=True)
        self.bp1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

        self.wo1 = nn.Parameter(torch.normal(mean, std), requires_grad=True)
        self.wo2 = nn.Parameter(torch.normal(mean, std), requires_grad=True)
        self.bo1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

    def lstm_unit(self, input_value, long_memory, short_memory):
        '''
        进行LSTM运算
        '''
        # long term memory也被称为cell state
        # short term memory也被称为hidden state

        # 1）第一层决定了当前Long-term memory应该被记住的百分比
        long_remember_percent = torch.sigmoid((short_memory * self.wlr1) +
                                              (input_value * self.wlr2) +
                                              self.blr1)

        # 2)第二层创造了一个新的潜在的Long-term memory，
        # 并且决定了将其添加到当前Long-term memory的百分比。
        potential_remember_percent = torch.sigmoid((short_memory * self.wpr1) +
                                                   (input_value * self.wpr2) +
                                                   self.bpr1)
        potential_memory = torch.tanh((short_memory * self.wp1) +
                                      (input_value * self.wp2) +
                                      self.bp1)
        # 利用上面两层的结果更新Long-term memory
        updated_long_memory = ((long_memory * long_remember_percent) +
                               (potential_remember_percent * potential_memory))

        # 3）第三层创造了一个新的潜在的Short-term memory，
        # 并且决定了其被记住和用作输出的百分比。
        output_percent = torch.sigmoid((short_memory * self.wo1) +
                                       (input_value * self.wo2) +
                                       self.bo1)
        updated_short_memory = torch.tanh(updated_long_memory) * output_percent
        # 最后，返回并更新长、短期记忆
        return([updated_long_memory, updated_short_memory])

    def forward(self, input):
        '''
        展开LSTM
        '''
        long_memory = 0
        short_memory = 0
        day1 = input[0]
        day2 = input[1]
        day3 = input[2]
        day4 = input[3]

        long_memory, short_memory  = self.lstm_unit(day1, long_memory, short_memory)
        long_memory, short_memory = self.lstm_unit(day2, long_memory, short_memory)
        long_memory, short_memory = self.lstm_unit(day3, long_memory, short_memory)
        long_memory, short_memory = self.lstm_unit(day4, long_memory, short_memory)

        return short_memory

    def configure_optimizers(self):
        '''
        配置优化器
        '''
        return Adam(self.parameters())

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

    trainer = L.Trainer(max_epochs= 2000, accelerator='gpu', devices='auto')
    trainer.fit(model, train_dataloaders=dataloader)

    # 找到最近的checkpoints文件。
    # Lightning提供的此功能可以使我们继续增加额外的epoch而不用重新开始。
    path_to_checkpoint = trainer.checkpoint_callback.best_model_path
    print("最近的检查点为" + path_to_checkpoint)
    trainer = L.Trainer(max_epochs=5000, accelerator='gpu', devices='auto')
    trainer.fit(model, train_dataloaders=dataloader, ckpt_path=path_to_checkpoint)

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