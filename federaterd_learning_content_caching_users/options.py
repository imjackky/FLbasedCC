# -*- coding: utf-8 -*- 
# Author: Jacky
# Creation Date: 2021/3/26

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=10,
                        help="number of rounds of training")
    parser.add_argument('--clients_num', type=int, default=10,
                        help="number of clients: K")
    parser.add_argument('--cachesize', type=list, default=[50, 100, 150, 200, 250, 300, 350, 400],
                        help="size of cache: CS")
    parser.add_argument('--local_ep', type=int, default=1,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=50,
                        help="local batch size: B")
    # parser.add_argument('--local_bs', type=int, default=5,
    #                     help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')

    # model arguments
    parser.add_argument('--model', type=str, default='AutoEncoder', help='model name')

    # workspace arguments
    parser.add_argument('--clean_dataset', type=bool, default=False, help="clean\
                        the model/data_set or not")
    parser.add_argument('--clean_user', type=bool, default=False, help="clean\
                        the user/ or not")
    parser.add_argument('--clean_clients', type=bool, default=False, help="clean\
                        the model/clients or not")

    # data set
    parser.add_argument('--dataset', type=str, default='ml-1m', help="name of dataset")
    # parser.add_argument('--dataset', type=str, default='ml-100k', help="name of dataset")

    # other arguments
    parser.add_argument('--gpu', default=None, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='adam', help="type \
                        of optimizer")
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    args = parser.parse_args()
    return args
