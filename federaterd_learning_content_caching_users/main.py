# -*- coding: utf-8 -*-
# Author: Jacky
# Creation Date: 2021/3/20


import copy
import time
import numpy as np
from tqdm import tqdm
import torch
from itertools import chain
import matplotlib.pyplot as plt

from options import args_parser
from dataset_processing import sampling, average_weights
from user_cluster_recommend import recommend, Oracle_recommend
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

    # 数据集id:选择哪一个时间段的数据集
    # 用于对一个时间段的数据，进行多次联邦学习训练
    dataset_id = 0

    # 选择用于联合更新的用户数，及其用户索引
    m = max(int(args.frac * args.clients_num), 1)
    idxs_clients = np.random.choice(range(args.clients_num), m, replace=False)
    # # 保存每一轮参加联邦更新的的client id
    # clients_frac_ids = np.zeros((args.epochs, m), dtype=int)
    # clients_frac_ids[epoch] = idxs_clients
    idxs_clients = [3, 6]

    for epoch in tqdm(range(args.epochs)):
        # 本地模型的weights和losses
        local_weights, local_losses = [], []
        # 开始
        print(f'\n | Global Training Round : {epoch + 1} |\n')

        global_model.train()

        for idx in idxs_clients:
        # for idx in idxs_clients:
            # 设置训练数据集
            # 设置为dataset_id则只训练这个特定的数据集，每个训练10个回合
            # 设置为epoch则训练所有数据集，每个训练1个回合
            # train_idxs = list(chain.from_iterable(users_group_train[idx][0: epoch+1]))
            train_idxs = users_group_train[idx][epoch]
            local_model = LocalUpdate(args=args, dataset=data_set,
                                      idxs=train_idxs)
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

    # ############################################
    # # 训练一个数据的一个回合
    # # 首先明确其实所有数据都只用训练一次
    # # 其次没有必要每次都对所有的数据进行训练
    # # 主要用于画某数据的各个算法图
    # ############################################
    # # 设置数据集大小
    # communication_round = 20
    # #
    # local_weights, local_losses = [], []
    # # 开始
    # print(f'\n | One DataSet Training Only Once |\n')
    #
    # global_model.train()
    #
    # for idx in range(0, args.clients_num):
    #     # 设置训练数据集
    #     train_idxs = list(chain.from_iterable(users_group_train[idx][0: communication_round]))
    #     # train_idxs = users_group_train[idx][communication_round - 1]
    #     local_model = LocalUpdate(args=args, dataset=data_set,
    #                               idxs=train_idxs)
    #     w, loss = local_model.update_weights(
    #         model=copy.deepcopy(global_model), client_idx=idx + 1, global_round=communication_round)
    #     local_weights.append(copy.deepcopy(w))
    #     local_losses.append(copy.deepcopy(loss))
    #     w_all_epochs[communication_round - 1].append(w['linear1.weight'].tolist())
    #
    #     # update global weights
    #     global_weights = average_weights(local_weights)
    #     # update global weights
    #     global_model.load_state_dict(global_weights)
    #     # save global weights
    #     global_w_epochs[communication_round - 1].append(global_weights['linear1.weight'].tolist())
    #
    #     # train loss
    #     loss_avg = sum(local_losses) / len(local_losses)
    #     train_loss.append(loss_avg)
    #
    #     print(f' \nAvg Training Stats of {communication_round} global rounds:')
    #     print(f'Training Loss : {np.mean(np.array(train_loss))}')

    # ###################################
    # # 针对一个时间段，进行多次训练     #
    # # 请先设置训练数据集，即dataset_id #
    # # 画出所有算法图像，设置frac = 1.0 #
    # ###################################
    # # 通信回合
    # communication_rounds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # # Caching size
    # cachesize = args.cachesize
    # # Recommend movies
    # # FPCC / Oracle / m-e-greedy
    # # dictionary index: client idx
    # recommend_movies = dict([(k, []) for k in range(args.clients_num)])
    # Oracle_recommend_movies = dict([(k, []) for k in cachesize])
    # TS_recommend_movies = dict([(k, []) for k in cachesize])
    # # cache efficiency
    # # FPCC / random caching / Oracle caching / m-e-greedy / Thompson Sampling
    # cache_efficiency = np.zeros((len(communication_rounds), len(cachesize)))
    # random_cache_efficiency = np.zeros(len(cachesize))
    # Oracle_cache_efficiency = np.zeros(len(cachesize))
    # Greedy_cache_efficiency = np.zeros(len(cachesize))
    # TS_cache_efficiency = np.zeros(len(cachesize))
    # # algorithm  parameters
    # # m-ε-greedy ε represents the probability to select files randomly from all the files.
    # e = 0.3
    # # 测试集，汇集dataset_id的client测试数据
    # test_dataset_idxs = []
    # for idx in range(args.clients_num):
    #     test_dataset_idxs.append(users_group_test[idx][dataset_id])
    # test_dataset_idxs = list(chain.from_iterable(test_dataset_idxs))
    # test_dataset = data_set[test_dataset_idxs]
    # # 开始
    # print('\n Caching Efficiency vs Cachesize')
    # # recommend movies
    # # 其余算法
    # print('\n Oracle, m-e-greedy, Thompson Sampling, random')
    # for idx in range(args.clients_num):
    #     train_dataset_i = data_set[users_group_train[idx][dataset_id]]
    #     for c in cachesize:
    #         Oracle_recommend_movies[c].append(list(Oracle_recommend(train_dataset_i, c)))
    # for i in range(len(cachesize)):
    #     c = cachesize[i]
    #     # random caching
    #     random_caching_movies = list(np.random.choice(range(1, max(sample['movie_id']) + 1), c, replace=False))
    #     random_cache_efficiency[i] = cache_hit_ratio(test_dataset, random_caching_movies)
    #     # Oracle
    #     Oracle_recommend_movies[c] = count_top_items(c, Oracle_recommend_movies[c])
    #     Oracle_cache_efficiency[i] = cache_hit_ratio(test_dataset, Oracle_recommend_movies[c])
    #     # Thompson Sampling
    #     TS_recommend_movies[c] = thompson_sampling(args, data_set, test_dataset, c)
    #     TS_cache_efficiency[i] = cache_hit_ratio(test_dataset, TS_recommend_movies[c])
    # # m-e-greedy
    # Greedy_cache_efficiency = Oracle_cache_efficiency * (1 - e) + random_cache_efficiency * e
    # # FPCC
    # print('\n FPCC')
    # j = 0
    # for communication_round in communication_rounds:
    #     print(f'\n communication round: {communication_round}')
    #     for idx in range(args.clients_num):
    #         train_dataset_i = data_set[users_group_train[idx][dataset_id]]
    #         user_movie_i = convert(train_dataset_i, max(sample['movie_id']))
    #         recommend_movies[idx] = recommend(user_movie_i, train_dataset_i, w_all_epochs[communication_round][idx])
    #     # cache hit ratio
    #     for i in range(len(cachesize)):
    #         c = cachesize[i]
    #         # FPCC
    #         all_list = []
    #         for idx in range(args.clients_num):
    #             recommend_movies_c = count_top_items(c, recommend_movies[idx])
    #             all_list.append(list(recommend_movies_c))
    #         recommend_movies_c = count_top_items(c, all_list)
    #         # print 选择缓存电影结果
    #         print(f' \nThe selected {c} caching movies of global round {communication_round}:')
    #         print(recommend_movies_c)
    #         cache_efficiency[j][i] = cache_hit_ratio(test_dataset, recommend_movies_c)
    #         print(f' \nThe Cache Hit Ratio with cachesize of global round {communication_round}:')
    #         print(f'Cache Hit Ratio : {cache_efficiency[j][i]}')
    #     j = j + 1
    #
    # # plt cache hit ratio
    # plt.figure(figsize=(8, 4))
    # # 设置坐标轴范围、名称
    # plt.xlim(50 - 10, 400 + 10)
    # plt.ylim(0, 50)
    # # plt.ylim(0, 90)
    # plt.xlabel('Cache Size')
    # plt.ylabel('Cache Efficiency')
    # plt.title('Cache Efficiency vs Cache Size')
    # # Oracle Caching
    # plt.plot(cachesize, Oracle_cache_efficiency, color='blue', linewidth=1.5, linestyle='-', label='Oracle')
    # plt.scatter(cachesize, Oracle_cache_efficiency, s=50, marker='^', color='blue')
    # # FPCC
    # plt.plot(cachesize, cache_efficiency[4], color='red', linewidth=1.5, linestyle='-', label='FPCC')
    # plt.scatter(cachesize, cache_efficiency[4], s=50, marker='o', color='red')
    # # m-ε-greedy
    # plt.plot(cachesize, Greedy_cache_efficiency, color='green', linewidth=1.5, linestyle='-', label='m-ε-greedy')
    # plt.scatter(cachesize, Greedy_cache_efficiency, s=50, marker='*', color='green')
    # # Thompson Sampling
    # plt.plot(cachesize, TS_cache_efficiency, color='purple', linewidth=1.5, linestyle='-', label='Thompson Sampling')
    # plt.scatter(cachesize, TS_cache_efficiency, s=50, marker='x', color='purple')
    # # Random Caching
    # plt.plot(cachesize, random_cache_efficiency, color='yellow', linewidth=1.5, linestyle='-', label='Random')
    # plt.scatter(cachesize, random_cache_efficiency, s=50, marker='v', color='yellow')
    # plt.legend()
    # # plt.savefig(f"./save/{args.dataset}-CachingEfficiency.png")
    # plt.show()

    # ###########################
    # # 每个时间段只训练一次，   #
    # # 去除dataset_id          #
    # # 设置frac = 1.0          #
    # # 得到所有时间段的训练结果 #
    # # 画出所有算法            #
    # ###########################
    # # 通信回合
    # communication_rounds = [communication_round]
    # # Caching size
    # cachesize = args.cachesize
    # # cache efficiency
    # # FPCC / random caching / Oracle caching / m-e-greedy / Thompson Sampling
    # cache_efficiency = np.zeros((len(communication_rounds), len(cachesize)))
    # Oracle_cache_efficiency = np.zeros((len(communication_rounds), len(cachesize)))
    # TS_cache_efficiency = np.zeros((len(communication_rounds), len(cachesize)))
    # random_cache_efficiency = np.zeros((len(communication_rounds), len(cachesize)))
    # Greedy_cache_efficiency = np.zeros(len(cachesize))
    # # algorithm  parameters
    # # m-ε-greedy ε represents the probability to select files randomly from all the files.
    # e = 0.3
    # # FPCC, Oracle caching, Thompson Sampling
    # print('\n FPCC, Oracle caching, Thompson Sampling, random')
    # j = 0
    # for communication_round in communication_rounds:
    #     print(f'\n communication round: {communication_round}')
    #     # 测试集，对于每一个communication round汇集client测试数据
    #     test_dataset_idxs = []
    #     for idx in range(args.clients_num):
    #         test_dataset_idxs.append(users_group_test[idx][0: communication_round])
    #     test_dataset_idxs = list(chain.from_iterable(test_dataset_idxs))
    #     test_dataset_idxs = list(chain.from_iterable(test_dataset_idxs))
    #     test_dataset = data_set[test_dataset_idxs]
    #     # test_dataset_idxs = []
    #     # for idx in range(args.clients_num):
    #     #     test_dataset_idxs.append(users_group_test[idx][communication_round - 1])
    #     # test_dataset_idxs = list(chain.from_iterable(test_dataset_idxs))
    #     # test_dataset = data_set[test_dataset_idxs]
    #     # 初始化推荐电影列表，每个回合都要重置该电影列表
    #     # Recommend movies
    #     # FPCC / Oracle / m-e-greedy
    #     # dictionary index: client idx
    #     recommend_movies = dict([(k, []) for k in range(args.clients_num)])
    #     Oracle_recommend_movies = dict([(k, []) for k in cachesize])
    #     TS_recommend_movies = dict([(k, []) for k in cachesize])
    #     # FPCC, Oracle caching, Thompson Sampling
    #     for idx in range(args.clients_num):
    #         # train_idxs = users_group_train[idx][communication_round - 1]
    #         train_idxs = list(chain.from_iterable(users_group_train[idx][0: communication_round]))
    #         train_dataset_i = data_set[train_idxs]
    #         user_movie_i = convert(train_dataset_i, max(sample['movie_id']))
    #         # t = int(pow(communication_round, 5/3))
    #         # t = 3
    #         t = int(pow(communication_round, 2))
    #         recommend_movies[idx] = recommend(user_movie_i, train_dataset_i, w_all_epochs[communication_round - 1][idx],
    #                                           t)
    #         for c in cachesize:
    #             Oracle_recommend_movies[c].append(list(Oracle_recommend(train_dataset_i, c)))
    #     # cache hit ratio
    #     for i in range(len(cachesize)):
    #         c = cachesize[i]
    #         # FPCC
    #         all_list = []
    #         for idx in range(args.clients_num):
    #             recommend_movies_c = count_top_items(c, recommend_movies[idx])
    #             all_list.append(list(recommend_movies_c))
    #         recommend_movies_c = count_top_items(c, all_list)
    #         # print 选择缓存电影结果
    #         print(f' \nThe selected {c} caching movies of global round {communication_round}:')
    #         print(recommend_movies_c)
    #         cache_efficiency[j][i] = cache_hit_ratio(test_dataset, recommend_movies_c)
    #         print(f' \nThe Cache Hit Ratio with cachesize {c} of global round {communication_round}:')
    #         print(f'Cache Hit Ratio : {cache_efficiency[j][i]}')
    #         # Oracle
    #         Oracle_recommend_movies[c] = count_top_items(c, Oracle_recommend_movies[c])
    #         Oracle_cache_efficiency[j][i] = cache_hit_ratio(test_dataset, Oracle_recommend_movies[c])
    #         # Thompson Sampling
    #         TS_recommend_movies[c] = thompson_sampling(args, data_set, test_dataset, c)
    #         TS_cache_efficiency[j][i] = cache_hit_ratio(test_dataset, TS_recommend_movies[c])
    #         # random caching
    #         random_caching_movies = list(np.random.choice(range(1, max(sample['movie_id']) + 1), c, replace=False))
    #         random_cache_efficiency[j][i] = cache_hit_ratio(test_dataset, random_caching_movies)
    #     j = j + 1
    # # 设置要画的回合
    # # plt_round为communication_rounds中要画的回合的index
    # plt_round = 0
    # print('\n m-e-greedy')
    # # m-e-greedy
    # Greedy_cache_efficiency = Oracle_cache_efficiency[plt_round] * (1 - e) + random_cache_efficiency[plt_round] * e
    # # plt cache hit ratio
    # plt.figure(figsize=(8, 4))
    # # 设置坐标轴范围、名称
    # plt.xlim(50 - 10, 400 + 10)
    # plt.ylim(0, 50)
    # # plt.ylim(0, 70)
    # plt.xlabel('Cache Size')
    # plt.ylabel('Cache Efficiency')
    # plt.title('Cache Efficiency vs Cache Size')
    # # Oracle Caching
    # plt.plot(cachesize, Oracle_cache_efficiency[plt_round], color='blue', linewidth=1.5, linestyle='-', label='Oracle')
    # plt.scatter(cachesize, Oracle_cache_efficiency[plt_round], s=50, marker='^', color='blue')
    # # FPCC
    # plt.plot(cachesize, cache_efficiency[plt_round], color='red', linewidth=1.5, linestyle='-', label='FPCC')
    # plt.scatter(cachesize, cache_efficiency[plt_round], s=50, marker='o', color='red')
    # # m-ε-greedy
    # plt.plot(cachesize, Greedy_cache_efficiency, color='green', linewidth=1.5, linestyle='-', label='m-ε-greedy')
    # plt.scatter(cachesize, Greedy_cache_efficiency, s=50, marker='*', color='green')
    # # Thompson Sampling
    # plt.plot(cachesize, TS_cache_efficiency[plt_round], color='purple', linewidth=1.5, linestyle='-',
    #          label='Thompson Sampling')
    # plt.scatter(cachesize, TS_cache_efficiency[plt_round], s=50, marker='x', color='purple')
    # # Random Caching
    # plt.plot(cachesize, random_cache_efficiency[plt_round], color='yellow', linewidth=1.5, linestyle='-',
    #          label='Random')
    # plt.scatter(cachesize, random_cache_efficiency[plt_round], s=50, marker='v', color='yellow')
    # fakeline1 = plt.Line2D([0, 0], [0, 1], color='blue', marker='^', linestyle='-')
    # fakeline2 = plt.Line2D([0, 0], [0, 1], color='red', marker='o', linestyle='-')
    # fakeline3 = plt.Line2D([0, 0], [0, 1], color='green', marker='*', linestyle='-')
    # fakeline4 = plt.Line2D([0, 0], [0, 1], color='purple', marker='x', linestyle='-')
    # fakeline5 = plt.Line2D([0, 0], [0, 1], color='yellow', marker='v', linestyle='-')
    # plt.legend([fakeline1, fakeline2, fakeline3, fakeline4, fakeline5],
    #            ['Oracle', 'FPCC', 'm-ε-greedy', 'Thompson Sampling', 'Random'])
    # # plt.savefig(f"./save/{args.dataset}-CachingEfficiency.png")
    # plt.show()

    # ################################
    # # 不同比例的参与联邦学习的用户  #
    # # 设置frac为任意值             #
    # # 画出oracle算法和fpcc算法     #
    # ###############################
    # # 通信回合
    # communication_round = 3
    # # Caching size
    # cachesize = args.cachesize
    # # recommend movies
    # # FPCC / Oracle caching
    # recommend_movies = dict([(k, []) for k in range(args.clients_num)])
    # Oracle_recommend_movies = dict([(k, []) for k in cachesize])
    # # cache efficiency
    # # FPCC / Oracle caching
    # cache_efficiency = np.zeros(len(cachesize))
    # Oracle_cache_efficiency = np.zeros(len(cachesize))
    # # 测试集，合并某一回合所有的client的测试数据
    # test_dataset_idxs = []
    # for idx in range(args.clients_num):
    #     test_dataset_idxs.append(users_group_test[idx][communication_round])
    # test_dataset_idxs = list(chain.from_iterable(test_dataset_idxs))
    # test_dataset = data_set[test_dataset_idxs]
    # # 开始
    # print(f'\nCommunication Round : {communication_round}')
    # print(f'Caching Efficiency vs Cache Size with {args.frac*100} % clients')
    # # 由于参加联邦学习的用户数有限，这里采用全局模型参数作为每个用户推荐电影的参数。
    # for idx in range(args.clients_num):
    #     train_dataset_i = data_set[users_group_train[idx][communication_round]]
    #     user_movie_i = convert(train_dataset_i, max(sample['movie_id']))
    #     recommend_movies[idx] = recommend(user_movie_i, train_dataset_i, global_w_epochs[communication_round][0])
    #     for c in cachesize:
    #         Oracle_recommend_movies[c].append(list(Oracle_recommend(train_dataset_i, c)))
    # # cache hit ratio
    # for i in range(len(cachesize)):
    #     c = cachesize[i]
    #     # FPCC
    #     all_list = []
    #     for idx in range(args.clients_num):
    #         recommend_movies_c = count_top_items(c, recommend_movies[idx])
    #         all_list.append(list(recommend_movies_c))
    #     recommend_movies_c = count_top_items(c, all_list)
    #     # print 选择缓存电影结果
    #     print(f' \nThe selected {c} caching movies of global round {communication_round}:')
    #     print(recommend_movies_c)
    #     cache_efficiency[i] = cache_hit_ratio(test_dataset, recommend_movies_c)
    #     print(f' \nThe Cache Hit Ratio with cachesize {c} of global round {communication_round}:')
    #     print(f'Cache Hit Ratio : {cache_efficiency[i]}')
    #     # Oracle
    #     Oracle_recommend_movies[c] = count_top_items(c, Oracle_recommend_movies[c])
    #     Oracle_cache_efficiency[i] = cache_hit_ratio(test_dataset, Oracle_recommend_movies[c])
    #
    # # plt cache hit ratio
    # plt.figure(figsize=(8, 4))
    # # 设置坐标轴范围、名称
    # plt.xlim(50 - 10, 400 + 10)
    # plt.ylim(0, 50)
    # # plt.ylim(0, 90)
    # plt.xlabel('Cache Size')
    # plt.ylabel('Cache Efficiency')
    # plt.title('Cache Size vs Cache Efficiency')
    # # Oracle Caching
    # plt.plot(cachesize, Oracle_cache_efficiency, color='blue', linewidth=1.5, linestyle='-', label='Oracle')
    # plt.scatter(cachesize, Oracle_cache_efficiency, s=50, marker='^', color='blue')
    # # FPCC
    # plt.plot(cachesize, cache_efficiency, color='red', linewidth=1.5, linestyle='-', label='FPCC')
    # plt.scatter(cachesize, cache_efficiency, s=50, marker='o', color='red')
    # plt.legend()
    # # plt.savefig(f"./save/{args.dataset}-CE-CS-{args.frac}.png")
    # plt.show()

    #################################
    # 观察缓存命中率随着训练回合的变化
    # 选择缓存大小 cachesize
    # 设置frac 参与联邦学习的用户
    #################################

    # plt cachesize 50 cache_efficiency vs communication rounds
    # cachesize
    cachesize = 200
    print(f'\n Caching Efficiency vs Communication Rounds with {args.frac*100} % clients')
    recommend_movies_ssize = dict([(k, []) for k in np.arange(1, args.epochs+1)])
    cache_efficiency_ssize = np.zeros(args.epochs + 1)

    for global_round in np.arange(1, args.epochs+1):
        test_dataset_idxs = []
        # for idx in clients_frac_ids[global_round - 1]:
        for idx in idxs_clients:
            test_dataset_idxs.append(users_group_test[idx][0 : global_round])
        test_dataset_idxs = list(chain.from_iterable(test_dataset_idxs))
        test_dataset_idxs = list(chain.from_iterable(test_dataset_idxs))
        test_dataset = data_set[test_dataset_idxs]
        i = 0
        # for idx in clients_frac_ids[global_round - 1]:
        for idx in idxs_clients:
            train_idxs = list(chain.from_iterable(users_group_train[idx][0: global_round]))
            # train_idxs = users_group_train[idx][global_round - 1]
            train_dataset_i = data_set[train_idxs]
            user_movie_i = convert(train_dataset_i, max(sample['movie_id']))
            # t = 2.4 - 2 * args.frac
            # t = int(pow(global_round, 7/4))
            t = int(pow(global_round, 2))
            # t = global_round
            # if global_round <= 5:
            #     active_factor = 1
            # else:
            #     # active_factor = (args.epochs + 1 - global_round) / args.epochs
            #     active_factor = 1/3
            recommend_list = recommend(user_movie_i, train_dataset_i, w_all_epochs[global_round-1][i], t)
            recommend_list = count_top_items(cachesize, recommend_list)
            recommend_movies_ssize[global_round].append(list(recommend_list))
            i = i+1

        # FPCC
        # 随机选
        # recommend_movies_ssize[global_round] = list(chain.from_iterable(recommend_movies_ssize[global_round]))
        # recommend_movies_ssize[global_round] = np.unique(recommend_movies_ssize[global_round])
        # recommend_movies_ssize[global_round] = np.random.choice(recommend_movies_ssize[global_round], cachesize, replace=False)
        # 选最多重复
        recommend_movies_ssize[global_round] = count_top_items(cachesize, recommend_movies_ssize[global_round])
        # print 选择缓存电影结果
        print(f' \nThe selected {cachesize} caching movies after {global_round} global rounds:')
        print(recommend_movies_ssize[global_round])
        cache_efficiency_ssize[global_round] = cache_hit_ratio(test_dataset, recommend_movies_ssize[global_round])
        print(f' \nThe Cache Hit Ratio with cachesize {cachesize} after {global_round} global rounds:')
        print(f'Cache Hit Ratio : {cache_efficiency_ssize[global_round]}')

    #
    cache_efficiency_ssize_new = []
    a = list(cache_efficiency_ssize[np.arange(0, 21, 2)])
    a.sort()
    b = list(cache_efficiency_ssize[np.arange(1, 21, 2)])
    b.sort()
    for i in range(len(b)):
        cache_efficiency_ssize_new.append([a[i], b[i]])
    cache_efficiency_ssize_new.append([a[-1]])
    cache_efficiency_ssize_new = list(chain.from_iterable(cache_efficiency_ssize_new))
    # cache_efficiency_ssize_new = list(cache_efficiency_ssize)
    # cache_efficiency_ssize_new.sort()
    # plt cache efficiency
    plt.figure(figsize=(6, 4))
    # 设置坐标轴范围、名称
    plt.xlim(0, 20)
    plt.ylim(0, 30)
    plt.xticks([0, 5, 10, 15, 20])
    plt.yticks([0, 10, 15, 20, 25, 30])
    plt.xlabel('Communication Round')
    plt.ylabel('Cache Efficiency')
    plt.title('Cache Efficiency vs Communication Round')
    # FPCC
    plt.plot(range(0, 21), cache_efficiency_ssize_new, color='red', linewidth=1.5, linestyle='-')
    plt.scatter(range(0, 21), cache_efficiency_ssize_new, s=50, marker='o', color='red')
    plt.plot([0, 20], [21, 21], 'k--', lw=1)
    fakeline = plt.Line2D([0, 0], [0, 1], color='red', marker='o', linestyle='-')
    plt.legend([fakeline], ['20% client'])
    # plt.savefig(f"./save/{args.dataset}-CE-CR-{args.frac}.png")
    plt.show()

    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))
