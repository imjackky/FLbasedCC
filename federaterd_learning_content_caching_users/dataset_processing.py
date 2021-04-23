# -*- coding: utf-8 -*- 
# Author: Jacky
# Creation Date: 2021/3/16

import numpy as np
import pandas as pd
import copy
import torch
from data_set import DataSet
from user_info import UserInfo
from options import args_parser
from data_set import convert
from itertools import chain
import utils


def get_dataset(args):
    """
    :param: args:
    :return: ratings: dataFrame ['user_id' 'movie_id' 'rating']
    :return: user_info:  dataFrame ['user_id' 'gender' 'age' 'occupation']
    """
    model_manager = utils.ModelManager('data_set')
    user_manager = utils.UserInfoManager(args.dataset)

    '''Do you want to clean workspace and retrain model/data_set user again?'''
    '''if you want to retrain model/data_set user, please set clean_workspace True'''
    model_manager.clean_workspace(args.clean_dataset)
    user_manager.clean_workspace(args.clean_user)

    # 导入模型信息
    try:
        ratings = model_manager.load_model(args.dataset + '-ratings')
        print("Load " + args.dataset + " data_set success.\n")
    except OSError:
        ratings = DataSet.LoadDataSet(name=args.dataset)
        model_manager.save_model(ratings, args.dataset + '-ratings')

    # 导入用户信息
    try:
        user_info = user_manager.load_user_info('user_info')
        print("Load " + args.dataset + " user_info success.\n")
    except OSError:
        user_info = UserInfo.load_user_info(name=args.dataset)
        user_manager.save_user_info(user_info, 'user_info')

    return ratings, user_info


def sampling(args):
    """
    :param args
    :return: sample: matrix user_id|movie_id|rating|gender|age|occupation|label
    :return: user_group_train, the idx of sample for each client for training
    :return: user_group_test, the idx of sample for each client for testing
    """
    # 存储每个client信息
    model_manager = utils.ModelManager('clients')
    '''Do you want to clean workspace and retrain model/clients again?'''
    '''if you want to change test_size or retrain model/clients, please set clean_workspace True'''
    model_manager.clean_workspace(args.clean_clients)
    # 导入模型信息
    try:
        users_group_train = model_manager.load_model(args.dataset + '-user_group_train')
        users_group_test = model_manager.load_model(args.dataset + '-user_group_test')
        sample = model_manager.load_model(args.dataset + '-sample')
        print("Load " + args.dataset + " clients info success.\n")
    except OSError:
        # 调用get_dataset函数，得到ratings,user_info
        ratings, user_info = get_dataset(args)
        # sample user_id|movie_id|rating|gender|age|occupation
        sample = pd.merge(ratings, user_info, on=['user_id'], how='inner')
        sample = sample.astype({'user_id': 'int64', 'movie_id': 'int64', 'rating': 'float64',
                                'gender': 'float64', 'age': 'float64', 'occupation': 'float64'})
        # 生成每个客户用来train和test的idx
        users_group_train = dict([(k, []) for k in range(args.clients_num)])
        users_group_test = dict([(k, []) for k in range(args.clients_num)])
        # 每个client包含的用户数
        # users_clients保存了每个client包含的用户数
        users_num_client = (user_info.index[-1] + 1) / args.clients_num
        # ml-1m & ml-100k
        if args.dataset == 'ml-1m':
            users_clients = np.arange(1, (user_info.index[-1] + 1)+1, dtype=int)[::-1]
        else:
            users_clients = []
            users_clients_tmp = list(sample.user_id)
            for i in users_clients_tmp:
                if i not in users_clients:
                    users_clients.append(i)
            users_clients = np.array(users_clients)
        # 判断总用户数是否为clients_num的倍数
        if users_num_client == round(users_num_client):
            users_clients = users_clients.reshape((-1, args.clients_num))
        elif users_num_client > round(users_num_client):
            # 去除多余的数据部分
            index = (user_info.index[-1] + 1) - round(users_num_client) * args.clients_num
            users_clients = users_clients[0: -index].reshape((-1, args.clients_num))
        else:
            # 增加数据0
            index = round(users_num_client) * args.clients_num - (user_info.index[-1] + 1)
            users_clients_new = np.zeros(round(users_num_client) * args.clients_num)
            users_clients_new[0:-index] = users_clients
            users_clients = users_clients_new.reshape((-1, args.clients_num))

        # 每个client每个回合包含的用户数
        # 前args.epochs - 1个回合包含的用户数
        users_num_client = round(users_num_client)
        users_num_client_round = round(users_num_client / args.epochs)

        # 生成用户数据集
        # 对用户数据集进行划分train/test
        for i in range(args.clients_num):
            print('loading client ' + str(i))
            # 对于前args.epochs-1个回合
            for j in range(args.epochs-1):
                # users_index_i：保存一个通信回合用户所有数据
                # users_index_i_train：保存一个通信回合用户用来训练的数据
                # users_index_i_test：保存一个通信回合用户用来测试的数据
                users_index_i_train = []
                users_index_i_test = []
                for k in range(users_num_client_round):
                    user_id_i = users_clients[k+j*users_num_client_round, i]
                    user_index_i = sample[sample.user_id == user_id_i].index.tolist()
                    # 选取数据的前80%作为当前通信回合的训练部分
                    # 选取数据的后20%作为当前通信回合的测试部分
                    users_index_i_train.append(user_index_i[0:int(0.8 * len(user_index_i))])
                    users_index_i_test.append(user_index_i[int(0.8 * len(user_index_i)):])

                users_index_i_train = list(chain.from_iterable(users_index_i_train))
                users_index_i_test = list(chain.from_iterable(users_index_i_test))

                users_group_train[i].append(users_index_i_train)
                users_group_test[i].append(users_index_i_test)
            # 对于最后一个回合
            users_index_i_train = []
            users_index_i_test = []
            for user_id_i in users_clients[(args.epochs-1)*users_num_client_round:, i]:
                user_index_i = sample[sample.user_id == user_id_i].index.tolist()
                # 选取数据的前80%作为当前通信回合的训练部分
                # 选取数据的后20%作为当前通信回合的测试部分
                users_index_i_train.append(user_index_i[0:int(0.8 * len(user_index_i))])
                users_index_i_test.append(user_index_i[int(0.8 * len(user_index_i)):])
            users_index_i_train = list(chain.from_iterable(users_index_i_train))
            users_index_i_test = list(chain.from_iterable(users_index_i_test))

            users_group_train[i].append(users_index_i_train)
            users_group_test[i].append(users_index_i_test)
            print('generate client ' + str(i) + ' info success\n')
        # 存储user_group_train user_group_test sample
        model_manager.save_model(sample, args.dataset + '-sample')
        model_manager.save_model(users_group_train, args.dataset + '-user_group_train')
        model_manager.save_model(users_group_test, args.dataset + '-user_group_test')

    return sample, users_group_train, users_group_test


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


if __name__ == '__main__':
    args = args_parser()
    # ratings, user_info = get_dataset(args)
    sample, users_group_train, users_group_test = sampling(args)
    # # 验证convert
    # client_6 = np.array(sample.iloc[users_group_test[6], :])
    # user_movie_6 = convert(client_6, max(sample['movie_id']))
