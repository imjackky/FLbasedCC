# -*- coding: utf-8 -*- 
# Author: Jacky
# Creation Date: 2021/4/13

# -*- coding: utf-8 -*-
# Author: Jacky
# Creation Date: 2021/3/20


import copy
import time
import math
import numpy as np
from tqdm import tqdm
import torch
from itertools import chain
import matplotlib.pyplot as plt

from options import args_parser
from dataset_processing import sampling, average_weights
from user_cluster_recommend import recommend, Oracle_recommend, recommend_randomN
from local_update import LocalUpdate, cache_hit_ratio
from model import AutoEncoder
from utils import exp_details, ModelManager, count_top_items
from Thompson_Sampling import thompson_sampling
from data_set import convert

if __name__ == '__main__':

    # 开始时间
    start_time = time.time()
    # args & 输出实验参数
    args = args_parser()
    exp_details(args)
    # gpu or cpu
    if args.gpu: torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'

    # load sample users_group_train users_group_test
    sample, users_group_train, users_group_test = sampling(args)
    data_set = np.array(sample)

    # build model
    global_model = AutoEncoder(int(max(data_set[:, 1])), 10)

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # all epoch weights
    # 保存每个回合，每个client的weights，非全局weights
    w_all_epochs = dict([(k, []) for k in range(args.epochs)])
    # 保存每个回合的全局weights
    global_w_epochs = dict([(k, []) for k in range(args.epochs)])

    # Training loss
    train_loss = []

    # 设置用户id和训练数据
    client_id = 0
    dataset_id = 0

    for epoch in tqdm(range(args.epochs)):
        # 本地模型的weights和losses
        local_weights, local_losses = [], []
        # 开始
        print(f'\n | Global Training Round : {epoch + 1} |\n')

        global_model.train()
        # 不引入联邦学习，只训练特定的client
        idxs_clients = [client_id]

        for idx in idxs_clients:
            local_model = LocalUpdate(args=args, dataset=data_set,
                                      idxs=users_group_train[idx][epoch])
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), client_idx=idx + 1, global_round=epoch + 1)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            w_all_epochs[epoch].append(w['linear1.weight'].tolist())

        # update global weights
        global_weights = average_weights(local_weights)
        # update global weights
        global_model.load_state_dict(global_weights)
        # save global weights
        global_w_epochs[epoch].append(global_weights['linear1.weight'].tolist())

        # train loss
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        print(f' \nAvg Training Stats after {epoch + 1} global rounds:')
        print(f'Training Loss : {np.mean(np.array(train_loss))}')

    # # 查看某个clinet，某个回合结束后的缓存命中率
    # # 用来验证我们的用户推荐算法的有效性
    # # Caching size
    # cachesize = args.cachesize
    # # 设置通信回合
    # global_round = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # # Recommend movies
    # # FPCC / Oracle
    # # dictionary index: client idx
    # recommend_movies = dict([(k, []) for k in cachesize])
    # recommend_movies_randomN = dict([(k, []) for k in cachesize])
    # Oracle_recommend_movies = dict([(k, []) for k in cachesize])
    # # cache efficiency
    # # FPCC / Oracle caching
    # cache_efficiency = np.zeros((len(global_round), len(cachesize)))
    # cache_efficiency_randomN = np.zeros(len(cachesize))
    # Oracle_cache_efficiency = np.zeros(len(cachesize))
    #
    # # 根据选择的用户及其数据得到训练集和测试集
    # test_dataset = data_set[users_group_test[client_id][dataset_id]]
    # train_dataset = data_set[users_group_train[client_id][dataset_id]]
    # user_movie_i = convert(train_dataset, max(sample['movie_id']))
    # # FPCC_randomN Oracle
    # recommend_movies_i_randomN = recommend_randomN(train_dataset)
    # for i in range(len(cachesize)):
    #     c = cachesize[i]
    #     # FPCC_randomN
    #     recommend_movies_randomN[c] = count_top_items(c, recommend_movies_i_randomN)
    #     cache_efficiency_randomN[i] = cache_hit_ratio(test_dataset, recommend_movies_randomN[c])
    #     # Oracle
    #     Oracle_recommend_movies[c] = list(Oracle_recommend(train_dataset, c))
    #     Oracle_cache_efficiency[i] = cache_hit_ratio(test_dataset, Oracle_recommend_movies[c])
    # j = 0
    # for communication_round in global_round:
    #     print(f'\nCommunication Round : {communication_round}')
    #     print(f'Caching Efficiency vs Cache Size of client {client_id}')
    #     recommend_movies_i = recommend(user_movie_i, train_dataset, w_all_epochs[communication_round][0])
    #     for i in range(len(cachesize)):
    #         c = cachesize[i]
    #         # FPCC
    #         recommend_movies[c] = count_top_items(c, recommend_movies_i)
    #         cache_efficiency[j][i] = cache_hit_ratio(test_dataset, recommend_movies[c])
    #     j = j+1
    #
    # # plt cache hit ratio
    # plt.figure(figsize=(6, 6))
    # # 设置坐标轴范围、名称
    # plt.xlim(50 - 5, 400 + 5)
    # plt.ylim(0, 50)
    # plt.xlabel('Cache Size')
    # plt.ylabel('Cache Efficiency')
    # plt.title('Cache Size vs Cache Efficiency')
    # # Oracle Caching
    # plt.plot(cachesize, Oracle_cache_efficiency, color='blue', linewidth=1.5, linestyle='-', label='Oracle')
    # plt.scatter(cachesize, Oracle_cache_efficiency, s=50, marker='^', color='blue')
    # # FPCC
    # # plt.plot(cachesize, cache_efficiency[2], color='red', linewidth=1.5, linestyle='-', label='FPCC_9')
    # # plt.scatter(cachesize, cache_efficiency[2], s=50, marker='o', color='red')
    # plt.plot(cachesize, cache_efficiency[1], color='green', linewidth=1.5, linestyle='-', label='FPCC_5')
    # plt.scatter(cachesize, cache_efficiency[1], s=50, marker='o', color='green')
    # # plt.plot(cachesize, cache_efficiency[0], color='greenyellow', linewidth=1.5, linestyle='-', label='FPCC_0')
    # # plt.scatter(cachesize, cache_efficiency[0], s=50, marker='o', color='greenyellow')
    # # FPCC_randomN
    # plt.plot(cachesize, cache_efficiency_randomN, color='yellow', linewidth=1.5, linestyle='-', label='FPCC_RandomN')
    # plt.scatter(cachesize, cache_efficiency_randomN, s=50, marker='o', color='yellow')
    # plt.legend()
    # # plt.savefig(f"./save/{args.dataset}-CE-CS-{args.frac}.png")
    # plt.show()

    ##################################
    # 观察缓存命中率随着训练回合的变化
    ##################################
    cachesize = 50
    print(f'\n Caching Efficiency vs Communication Rounds of client {client_id}')
    recommend_movies_ssize = dict([(k, []) for k in np.arange(1, args.epochs + 1)])
    cache_efficiency_ssize = np.zeros(args.epochs + 1)

    # # 所有测试集集合
    # test_dataset_idxs = list(chain.from_iterable(users_group_test[client_id][0: args.epochs + 1]))
    # test_dataset = data_set[test_dataset_idxs]

    for global_round in np.arange(1, args.epochs + 1):
        # 测试集
        test_dataset_idxs = list(chain.from_iterable(users_group_test[client_id][0: global_round]))
        # test_dataset_idxs = users_group_test[client_id][global_round - 1]
        test_dataset = data_set[test_dataset_idxs]
        # 训练集
        train_idxs = list(chain.from_iterable(users_group_train[client_id][0: global_round]))
        # train_idxs = users_group_train[client_id][global_round - 1]
        train_dataset_i = data_set[train_idxs]
        user_movie_i = convert(train_dataset_i, max(sample['movie_id']))
        t = int(pow(global_round, 2))
        # t = global_round
        recommend_list = recommend(user_movie_i, train_dataset_i, w_all_epochs[global_round - 1][0], t)
        recommend_movies_ssize[global_round] = count_top_items(cachesize, recommend_list)
        # print 选择缓存电影结果
        print(f' \nThe selected 50 caching movies after {global_round} global rounds:')
        print(recommend_movies_ssize[global_round])
        cache_efficiency_ssize[global_round] = cache_hit_ratio(test_dataset, recommend_movies_ssize[global_round])
        print(f' \nThe Cache Hit Ratio with cachesize 50 after {global_round} global rounds:')
        print(f'Cache Hit Ratio : {cache_efficiency_ssize[global_round]}')

    # plt cache efficiency
    plt.figure(figsize=(8, 4))
    # 设置坐标轴范围、名称
    plt.xlim(0, 20)
    plt.ylim(0, 10)
    plt.xticks([0, 5, 10, 15, 20])
    plt.yticks([0, 5, 10])
    plt.xlabel('Communication Round')
    plt.ylabel('Cache Efficiency')
    plt.title('Cache Efficiency vs Communication Round')
    # FPCC
    plt.plot(range(args.epochs + 1), cache_efficiency_ssize, color='red', linewidth=1.5, linestyle='-', label='FPCC')
    plt.scatter(range(args.epochs + 1), cache_efficiency_ssize, s=50, marker='o', color='red')
    plt.legend()
    # plt.savefig(f"./save/{args.dataset}-CE-CR-{args.frac}.png")
    plt.show()

    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))
