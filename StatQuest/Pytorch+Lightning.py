import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD

import lightning as L
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt
import seaborn as sns

class BasicLightningTrain(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.w00 = nn.Parameter(torch.tensor(1.7), requires_grad=False)
        self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad=False)
        self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad=False)

        self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad=False)
        self.b10 = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.w11 = nn.Parameter(torch.tensor(2.7), requires_grad=False)

        self.final_bias = nn.Parameter(torch.tensor(0.), requires_grad=True)

        self.learning_rate = 0.01

    def forward(self, input):
        input_to_top_relu = input * self.w00 + self.b00
        top_relu_output = F.relu(input_to_top_relu)
        scaled_top_relu_output = top_relu_output * self.w01

        input_to_bottom_relu = input * self.w10 + self.b10
        bottom_relu_output = F.relu(input_to_bottom_relu)
        scaled_bottom_relu_output = bottom_relu_output * self.w11

        input_to_final_relu = scaled_top_relu_output + scaled_bottom_relu_output + self.final_bias

        output = F.relu(input_to_final_relu)

        return output
    def configure_optimizers(self):
        return SGD(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        inputs_i, labels_i = batch
        outputs_i = self.forward(inputs_i)
        loss = (outputs_i - labels_i) ** 2
        return loss

def main():
    input_doses = torch.linspace(0, 1, 11)
    inputs = torch.tensor([0., 0.5, 1.] * 100)
    lables = torch.tensor([0., 1., 0.] * 100)
    dataset = TensorDataset(inputs, lables)
    dataloader = DataLoader(dataset)
    model = BasicLightningTrain()
    trainer = L.Trainer(max_epochs=35, accelerator='gpu', devices='auto')
    tuner = L.pytorch.tuner.tuning.Tuner(trainer)
    lr_find_results = tuner.lr_find(model,
                                    train_dataloaders=dataloader,
                                    min_lr=0.001,
                                    max_lr=1.0,
                                    early_stop_threshold=None)
    new_lr = lr_find_results.suggestion()
    print(f"lr_find() suggests {new_lr:.5f} for the learning rate.")
    model.learning_rate = new_lr

    trainer.fit(model, train_dataloaders=dataloader)

    print(model.final_bias.data)

    output_values = model(input_doses)

    sns.set(style="whitegrid")

    sns.lineplot(x=input_doses,
                 y=output_values.detach(),
                 color='green',
                 linewidth=2.5
                 )
    plt.ylabel('Effectiveness')
    plt.xlabel('Dose')
    plt.show()

if __name__ == '__main__':
    main()