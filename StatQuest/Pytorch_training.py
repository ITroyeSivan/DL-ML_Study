import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD

import matplotlib.pyplot as plt
import seaborn as sns

# Basic继承Pytorch的名为module的类
class BasicNN_train(nn.Module):
    def __init__(self):
        super().__init__()
        self.w00 = nn.parameter.Parameter(torch.tensor(1.7), requires_grad=False)
        self.b00 = nn.parameter.Parameter(torch.tensor(-0.85), requires_grad=False)
        self.w01 = nn.parameter.Parameter(torch.tensor(-40.8), requires_grad=False)

        self.w10 = nn.parameter.Parameter(torch.tensor(12.6), requires_grad=False)
        self.b10 = nn.parameter.Parameter(torch.tensor(0.0), requires_grad=False)
        self.w11 = nn.parameter.Parameter(torch.tensor(2.7), requires_grad=False)

        self.final_bias = nn.parameter.Parameter(torch.tensor(0.), requires_grad=True)
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

def main():
    input_doses = torch.linspace(0, 1, 11)

    model = BasicNN_train()

    inputs = torch.tensor([0., 0.5, 1.])
    lables = torch.tensor([0., 1., 0.])
    optimizer = SGD(model.parameters(), lr=0.1)
    print("Final Bias, before optimization: " + str(model.final_bias.data) + "\n")
    for epoch in range(100):
        total_loss = 0
        for iteration in range(len(inputs)):
            input_i = inputs[iteration]
            lables_i = lables[iteration]
            output_i = model(input_i)
            loss = (output_i - lables_i) ** 2
            loss.backward() #计算损失函数的导数
            total_loss += float(loss)
        if (total_loss < 0.0001):
            print("Num steps: " + str(epoch))
            break
        optimizer.step()
        optimizer.zero_grad()
        print("Step: " + str(epoch) + " Final Bias: " + str(model.final_bias.data) + "'\n")
    print("Final Bias, after optimization: " + str(model.final_bias.data) + "\n")

    output_values = model(input_doses)

    sns.set(style="whitegrid")
    sns.lineplot(x=input_doses,
                 # seaborn不知道如何处理梯度，所以用detach将其剥离
                 y=output_values.detach(),
                 color='green',
                 linewidth=2.5)
    plt.ylabel('Effectiveness')
    plt.xlabel('Dose')
    plt.show()

if __name__ == '__main__':
    main()
