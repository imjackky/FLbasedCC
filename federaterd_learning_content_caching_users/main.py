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
    # test_dataset & test_dataset_idx
    test_dataset_idxs = []
    for idx in range(args.clients_num):
        test_dataset_idxs.append(users_group_test[idx])
    test_dataset_idxs = list(chain.from_iterable(test_dataset_idxs))
    test_dataset = data_set[test_dataset_idxs]

    # build model
    global_model = AutoEncoder(int(max(data_set[:, 1])), 100)

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # all epoch weights
    w_all_epochs = dict([(k, []) for k in range(args.epochs)])

    # Training loss
    train_loss = []

    for epoch in tqdm(range(args.epochs)):
        # 本地模型的weights和losses
        local_weights, local_losses = [], []
        # 开始
        print(f'\n | Global Training Round : {epoch + 1} |\n')

        global_model.train()

        for idx in range(args.clients_num):
            local_model = LocalUpdate(args=args, dataset=data_set,
                                      idxs=users_group_train[idx])
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), client_idx=idx + 1, global_round=epoch + 1)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            w_all_epochs[epoch].append(w['linear1.weight'].tolist())

        # update global weights
        global_weights = average_weights(local_weights)
        # update global weights
        global_model.load_state_dict(global_weights)

        # train loss
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        print(f' \nAvg Training Stats after {epoch + 1} global rounds:')
        print(f'Training Loss : {np.mean(np.array(train_loss))}')

    # Caching size
    cachesize = args.cachesize
    # Recommend movies
    # FPCC / Oracle / m-e-greedy
    # dictionary index: client idx
    recommend_movies = dict([(k, []) for k in range(args.clients_num)])
    Oracle_recommend_movies = dict([(k, []) for k in cachesize])
    TS_recommend_movies = dict([(k, []) for k in cachesize])
    # cache efficiency
    # FPCC / random caching / Oracle caching / m-e-greedy / Thompson Sampling
    cache_efficiency = np.zeros(len(cachesize))
    random_cache_efficiency = np.zeros(len(cachesize))
    Oracle_cache_efficiency = np.zeros(len(cachesize))
    Greedy_cache_efficiency = np.zeros(len(cachesize))
    TS_cache_efficiency = np.zeros(len(cachesize))

    # algorithm  parameters
    # m-ε-greedy ε represents the probability to select files randomly from all the files.
    e = 0.3

    print('\n Caching Efficiency vs Cachesize')
    # recommend movies
    # 一个回合，已经训练好的FPCC
    for idx in range(args.clients_num):
        test_dataset_i = data_set[users_group_test[idx]]
        user_movie_i = convert(test_dataset_i, max(sample['movie_id']))
        recommend_movies[idx] = recommend(user_movie_i, test_dataset_i, w_all_epochs[args.epochs - 1][idx])

        for c in cachesize:
            Oracle_recommend_movies[c].append(list(Oracle_recommend(test_dataset_i, c)))

    # cache hit ratio
    for i in range(len(cachesize)):
        c = cachesize[i]
        # FPCC
        all_list = []
        for idx in range(args.clients_num):
            recommend_movies_c = count_top_items(c, recommend_movies[idx])
            all_list.append(list(recommend_movies_c))
        recommend_movies_c = count_top_items(c, all_list)
        # print 选择缓存电影结果
        print(f' \nThe selected {c} caching movies after {args.epochs} global rounds:')
        print(recommend_movies_c)
        cache_efficiency[i] = cache_hit_ratio(test_dataset, recommend_movies_c)
        print(f' \nThe Cache Hit Ratio with cachesize {c} after {args.epochs} global rounds:')
        print(f'Cache Hit Ratio : {cache_efficiency[i]}')
        # random caching
        random_caching_movies = list(np.random.choice(range(1, max(sample['movie_id']) + 1), c, replace=False))
        random_cache_efficiency[i] = cache_hit_ratio(test_dataset, random_caching_movies)
        # Oracle
        Oracle_recommend_movies[c] = count_top_items(c, Oracle_recommend_movies[c])
        Oracle_cache_efficiency[i] = cache_hit_ratio(test_dataset, Oracle_recommend_movies[c])
        # Thompson Sampling
        TS_recommend_movies[c] = thompson_sampling(args, data_set, test_dataset, c)
        TS_cache_efficiency[i] = cache_hit_ratio(test_dataset, TS_recommend_movies[c])

    # m-e-greedy
    Greedy_cache_efficiency = Oracle_cache_efficiency * (1 - e) + random_cache_efficiency * e

    # plt cache hit ratio
    plt.figure(figsize=(6, 6))
    # 设置坐标轴范围、名称
    plt.xlim(50 - 5, 400 + 5)
    plt.ylim(0, 50)
    # plt.ylim(0, 90)
    plt.xlabel('Cache Size')
    plt.ylabel('Cache Efficiency')
    plt.title('Cache Efficiency vs Cache Size')
    # Oracle Caching
    plt.plot(cachesize, Oracle_cache_efficiency, color='blue', linewidth=1.5, linestyle='-', label='Oracle')
    plt.scatter(cachesize, Oracle_cache_efficiency, s=50, marker='^', color='blue')
    # FPCC
    plt.plot(cachesize, cache_efficiency, color='red', linewidth=1.5, linestyle='-', label='FPCC')
    plt.scatter(cachesize, cache_efficiency, s=50, marker='o', color='red')
    # m-ε-greedy
    plt.plot(cachesize, Greedy_cache_efficiency, color='green', linewidth=1.5, linestyle='-', label='m-ε-greedy')
    plt.scatter(cachesize, Greedy_cache_efficiency, s=50, marker='*', color='green')
    # Thompson Sampling
    plt.plot(cachesize, TS_cache_efficiency, color='purple', linewidth=1.5, linestyle='-', label='Thompson Sampling')
    plt.scatter(cachesize, TS_cache_efficiency, s=50, marker='x', color='purple')
    # Random Caching
    plt.plot(cachesize, random_cache_efficiency, color='yellow', linewidth=1.5, linestyle='-', label='Random')
    plt.scatter(cachesize, random_cache_efficiency, s=50, marker='v', color='yellow')
    plt.legend()
    # plt.savefig(f"./save/{args.dataset}-CachingEfficiency.png")
    plt.show()

    # plt cachesize 50 cache_efficiency vs communication rounds
    print('\n Caching Efficiency vs Communication Rounds')
    recommend_movies_c50 = dict([(k, []) for k in np.arange(1, args.epochs+1)])
    cache_efficiency_c50 = np.zeros(args.epochs + 1)
    for global_round in np.arange(1, args.epochs+1):
        for idx in range(args.clients_num):
            test_dataset_i = data_set[users_group_test[idx]]
            user_movie_i = convert(test_dataset_i, max(sample['movie_id']))
            recommend_list = recommend(user_movie_i, test_dataset_i, w_all_epochs[global_round-1][idx])
            recommend_list = count_top_items(50, recommend_list)
            recommend_movies_c50[global_round].append(list(recommend_list))

        # FPCC
        recommend_movies_c50[global_round] = count_top_items(50, recommend_movies_c50[global_round])
        # print 选择缓存电影结果
        print(f' \nThe selected 50 caching movies after {global_round} global rounds:')
        print(recommend_movies_c50[global_round])
        cache_efficiency_c50[global_round] = cache_hit_ratio(test_dataset, recommend_movies_c50[global_round])
        print(f' \nThe Cache Hit Ratio with cachesize 50 after {global_round} global rounds:')
        print(f'Cache Hit Ratio : {cache_efficiency_c50[global_round]}')

    # plt cache efficiency
    plt.figure(figsize=(6, 6))
    # 设置坐标轴范围、名称
    plt.xlim(0, 10)
    plt.ylim(0, 20)
    plt.xlabel('Communication Round')
    plt.ylabel('Cache Efficiency')
    plt.title('Cache Efficiency vs Communication Round')
    # FPCC
    plt.plot(range(args.epochs+1), cache_efficiency_c50, color='red', linewidth=1.5, linestyle='-', label='FPCC')
    plt.scatter(range(args.epochs+1), cache_efficiency_c50, s=50, marker='o', color='red')
    plt.legend()
    # plt.savefig(f"./save/{args.dataset}-CacheEfficiency_CommunicationRound.png")
    plt.show()

    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))
