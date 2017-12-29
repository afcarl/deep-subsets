import sys
import os
import argparse
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath('..'))
from torch.autograd import Variable
import torch
import random
import numpy as np
from src.networks.mnist import Set2SubsetNet, Set2SubsetNetNull
from src.datatools import MNISTSubsets
from src.util_io import create_folder

def main(args):
    CUDA = False
    folder_name = args.name+'_'+args.task+'_'+args.architecture
    folder_path = os.path.join('./', folder_name)
    create_folder(folder_name)
    # create some different datasets for training:
    datasets = [
        torch.utils.data.DataLoader(
            MNISTSubsets(10000, set_sizes=[i], target=args.task),
            batch_size=64)
        for i in range(4,10)
        ]

    if args.architecture == 'set':
        net = Set2SubsetNet()
    elif args.architecture == 'null':
        net = Set2SubsetNetNull()
    elif args.architecture == 'seq':
        raise NotImplementedError('Sequence architecture is not yet implemented')
    else:
        raise ValueError('Unknown architecture. Must be set or seq!')
    
    optimizer = torch.optim.Adam(net.parameters())
    criterion = torch.nn.BCEWithLogitsLoss()

    if torch.cuda.is_available() and args.gpu != '':
        net.cuda()
        CUDA = True
        print('Using GPU')

    for n in range(args.epochs): # run for epochs
        # is this curriculum training?
        dataset = random.sample(datasets[:n+1], 1)[0]
        for i, (x, y) in enumerate(dataset): # batches in each epoch
            # zero the gradients
            optimizer.zero_grad()
            if CUDA:
                x = x.cuda()
                y = y.cuda()
            # prepare the data
            x, y = Variable(x), Variable(y)

            # run it through the network
            y_hat = net(x).squeeze()

            # calculate the loss
            loss = criterion(y_hat.float(), y.float())

            # update parameters
            loss.backward()
            optimizer.step()

        if n % 10 == 0:
            print('epoch: {}, loss: {}'.format(n, loss.cpu().data[0]))


    #TODO: fix the train=True to train=False
    datasets = [
        (i, torch.utils.data.DataLoader(
            MNISTSubsets(64, set_sizes=[i], target=args.task),
            batch_size=64))
        for i in range(4,100)
        ]

    set_sizes = []
    mse = []
    acc = []

    for set_size, dataset in datasets:
        for i, (x, y) in enumerate(dataset):
            # prepare the data
            if CUDA:
                x = x.cuda()
                y = y.cuda()
            x, y = Variable(x, volatile=True), Variable(y, volatile=True)

            # run it through the network
            y_hat = net(x).squeeze()

            # calculate the loss
            loss = criterion(y_hat.float(), y.float())
            if CUDA:
                loss = loss.cpu()
            set_sizes.append(set_size)
            mse.append(loss.data[0])
            acc.append(set_accuracy(y, y_hat).data[0])

    print(set_sizes)
    print(mse)
    print(acc)
    print(torch.FloatTensor(acc).mean())
    net.cpu()
    torch.save(net, os.path.join(folder_path, 'model.pyt'))



if __name__ == '__main__':

    parser = argparse.ArgumentParser('set2subset experiments')
    parser.add_argument('-e',
                        '--epochs',
                        help='Number of epochs to train',
                        type=int,
                        required=True)
    parser.add_argument('-a',
                        '--architecture',
                        help='Architecture to use (set, null, seq)',
                        type=str,
                        required=False,
                        default='set')
    parser.add_argument('-t',
                        '--task',
                        help='The task',
                        type=str,
                        required=False,
                        default='avg')
    parser.add_argument('-g',
                        '--gpu',
                        help='The gpu to use',
                        type=str,
                        required=False,
                        default='')
    parser.add_argument('-n',
                        '--name',
                        help='Name of the experiment',
                        type=str,
                        required=True,
                        default='experiment')
    args = parser.parse_args()

    # specify GPU ID on target machine
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    main(args)
