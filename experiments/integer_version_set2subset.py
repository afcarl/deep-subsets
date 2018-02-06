"""
An experiment to see if a supervised learning algorithm
can learn to select integer digits from a set of digits
This is to ensure that our network is correctly wired up
and that learning can actually happen from base 2 represented
digits.
"""
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
import math
import numpy as np
from src.networks.integer_subsets import IntegerSubsetNet
from src.datatools import IntegerSubsetsSupervised
from src.util_io import create_folder
from src.metrics import set_accuracy

increase_every = 1

def main(args):
    CUDA = False
    folder_name = args.name+'_'+args.task+'_'+args.architecture
    folder_path = os.path.join('./', folder_name)
    create_folder(folder_name)
    # create some different datasets for training:
    dataloaders = [
        torch.utils.data.DataLoader(
            IntegerSubsetsSupervised(100000, i, 10, target=args.task),
            batch_size=128)
        for i in range(4,10)
        ]

    if args.architecture == 'set':
        net = IntegerSubsetNet()
    elif args.architecture == 'null':
        net = IntegerSubsetNet(null_model=True)
    elif args.architecture == 'seq':
        raise NotImplementedError('Sequence architecture is not yet implemented')
    else:
        raise ValueError('Unknown architecture. Must be set or null!')
    
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
    criterion = torch.nn.BCEWithLogitsLoss()
    #lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1e-5)
    if torch.cuda.is_available() and args.gpu != '':
        net.cuda()
        CUDA = True
        print('Using GPU')

    for n in range(args.epochs): # run for epochs
        # is this curriculum training?

        batch_loss = 0
        dataset = random.sample(dataloaders[:math.ceil((n+1)/increase_every)], 1)[0]

        for i, (x, y) in enumerate(dataset): # batches in each epoch
            # zero the gradients
            optimizer.zero_grad()
            if CUDA:
                x = x.cuda()
                y = y.cuda()
            # prepare the data
            x, y = Variable(x).float(), Variable(y).float()

            # run it through the network
            y_hat = net(x)
            # calculate the loss
            loss = criterion(y_hat.float(), y.float())

            # update parameters
            loss.backward()
            optimizer.step()
            batch_loss += loss.cpu().data[0]
        lr_scheduler.step()

        if n % 10 == 0:
            set_acc, elem_acc = set_accuracy(y.squeeze(), y_hat.squeeze())
            print('epoch: {}, loss: {}, set acc: {}, elem_acc: {}'.format(n, batch_loss/(i+1), set_acc.data[0], elem_acc.data[0]))


    #TODO: fix the train=True to train=False
    datasets = [
        (i, torch.utils.data.DataLoader(
            IntegerSubsetsSupervised(256, i, 10, target='mean', seed=5),
            batch_size=256))
        for i in range(4,100)
        ]

    set_sizes = []
    mse = []
    set_accs = []
    elem_accs = []
    torch.save(net, os.path.join(folder_path, 'model-gpu.pyt'))

    for set_size, dataset in datasets:
        for i, (x, y) in enumerate(dataset):
            # prepare the data
            if CUDA:
                x = x.cuda()
                y = y.cuda()
            x, y = Variable(x, volatile=True), Variable(y, volatile=True).float()

            # run it through the network
            y_hat = net(x)
            # calculate the loss
            loss = criterion(y_hat, y)
            if CUDA:
                loss = loss.cpu()
            set_sizes.append(set_size)
            mse.append(loss.data[0])
            set_acc, elem_acc = set_accuracy(y.squeeze(), y_hat.squeeze())
            set_accs.append(set_acc.data[0])
            elem_accs.append(elem_acc.data[0])

    print(set_sizes)
    print(mse)
    print(set_accs)
    print(torch.FloatTensor(set_accs).mean())
    net.cpu()
    torch.save({'set_sizes': set_sizes,
                'mse':mse,
                'set_acc':set_accs,
                'elem_accs': elem_accs,
                'mean_acc':torch.FloatTensor(set_accs).mean()}, os.path.join(folder_path, 'results.json'))
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
                        help='Architecture to use (set, null)',
                        type=str,
                        required=False,
                        default='set')
    parser.add_argument('-t',
                        '--task',
                        help='The task',
                        type=str,
                        required=False,
                        default='mean')
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
    parser.add_argument('-s',
                        '--seed',
                        help='Seed',
                        type=str,
                        required=False,
                        default=0)
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)

    # specify GPU ID on target machine
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    main(args)
