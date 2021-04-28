# -*- coding: utf-8 -*- 
# Author: Jacky
# Creation Date: 2021/3/8


# Importing the libraries
import torch
import torch.nn as nn
torch.set_default_tensor_type(torch.DoubleTensor)


# Creating the architecture of the Neural Network
class AutoEncoder(nn.Module):
    def __init__(self, dim_in, dim_hidden):
        """
        AutoEncoder初始化
        """
        super(AutoEncoder, self).__init__()
        # 线性方程和激励函数
        self.linear1 = nn.Linear(dim_in, dim_hidden)
        self.linear2 = nn.Linear(dim_hidden, dim_in)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        神经网络forward函数
        :param x: 输入，Input
        :return: y_pred
        """
        y_encode = self.linear1(x)
        y_pred = self.sigmoid(self.linear2(y_encode))
        return y_pred
