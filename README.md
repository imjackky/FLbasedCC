# Federated Learning based Content Caching

#### 介绍
基于联邦学习的高效边缘缓存研究


#### 系统架构
在我们的系统模型中，中央服务器维护一个共享的全局模型。每个连接的用户都将基于本地训练数据集来计算从服务器下载的全局模型的更新，并将更新和推荐列表返回至服务器。接下来，服务器使用联邦平均算法聚合所有用户端更新以构建改进的全局模型，之后改进的模型将再次发送到用户端。本课题基于真实世界数据集MovieLens进行仿真。


#### 安装包

1.  pytorch
2.  numpy
3.  pandas等

#### 使用说明
data文件夹解压ml-100k和ml-1m，运行main

#### 算法实现
实现了基于联邦学习的边缘缓存算法，同时探讨了random caching、thompson sampling、m-e-greedy、oracle等算法
