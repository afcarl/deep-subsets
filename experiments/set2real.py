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
from src.networks.mnist import Set2RealNet, Seq2RealNet
from src.datatools import MNISTSets

create_folder = lambda f: [os.makedirs(os.path.join('./', f)) if not os.path.exists(os.path.join('./', f)) else False]

def plot_preds(x, y, net):
    image1, image2, image3, image4 = x.cpu()
    f = plt.figure()
    ax = f.add_subplot(221)
    ax.imshow(image1.numpy().reshape(28, 28))
    ax = f.add_subplot(222)
    ax.imshow(image2.numpy().reshape(28, 28))
    ax = f.add_subplot(223)
    ax.imshow(image3.numpy().reshape(28, 28))
    ax = f.add_subplot(224)
    ax.imshow(image4.numpy().reshape(28, 28))
    pred = np.around(net(Variable(x.unsqueeze(0), volatile=True)).data.cpu()[0][0], decimals=2)
    f.suptitle('predicted:' + str(pred) +' true:'+str(y))
    return f

def main(args):
    CUDA = False
    folder_name = args.name+'_'+args.task+'_'+args.architecture
    folder_path = os.path.join('./', folder_name)
    create_folder(folder_name)
    # create some different datasets for training:
    datasets = [
        torch.utils.data.DataLoader(
            MNISTSets(10000, set_sizes=[i], target=args.task),
            batch_size=64)
        for i in range(4,10)
        ]

    if args.architecture == 'set':
        net = Set2RealNet()
    elif args.architecture == 'seq':
        net = Seq2RealNet()
    else:
        raise ValueError('Unknown architecture. Must be set or seq!')
        
    optimizer = torch.optim.Adam(net.parameters())
    criterion = torch.nn.MSELoss()

    if torch.cuda.is_available() and args.gpu != '':
        net.cuda()
        CUDA = True

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
            y_hat = net(x)

            # calculate the loss
            loss = criterion(y_hat.float(), y.float())

            # update parameters
            loss.backward()
            optimizer.step()

        if n % 10 == 0:
            print('epoch: {}, loss: {}'.format(n, loss.cpu().data[0]))

    # save some of the predictions:
    x, y = next(iter(datasets[0]))
    if CUDA:
        x = x.cuda()
    plot_preds(x[0], y[0], net).savefig(os.path.join(folder_path, 'example1.pdf'))
    plot_preds(x[1], y[1], net).savefig(os.path.join(folder_path, 'example2.pdf'))
    plot_preds(x[2], y[2], net).savefig(os.path.join(folder_path, 'example3.pdf'))
    plot_preds(x[3], y[3], net).savefig(os.path.join(folder_path, 'example4.pdf'))

    #TODO: fix the train=True to train=False
    datasets = [
        (i, torch.utils.data.DataLoader(
            MNISTSets(64, set_sizes=[i], target=args.task, train=True),
            batch_size=64))
        for i in range(4,20)
        ]

    set_sizes = []
    mse = []

    for set_size, dataset in datasets:
        for i, (x, y) in enumerate(dataset):
            # prepare the data
            if CUDA:
                x = x.cuda()
                y = y.cuda()
            x, y = Variable(x, volatile=True), Variable(y, volatile=True)

            # run it through the network
            y_hat = net(x)

            # calculate the loss
            loss = criterion(y_hat.float(), y.float())
            if CUDA:
                loss = loss.cpu()
            set_sizes.append(set_size)
            mse.append(loss.data[0])

    print(set_sizes)
    print(mse)
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.plot(set_sizes, mse, '-')
    ax.set_xlabel('Set size')
    ax.set_ylabel('MSE')
    f.savefig(os.path.join(folder_path, 'generalization.pdf'))

    torch.save(net, os.path.join(folder_path, 'model.pyt'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser('set2real experiments')
    parser.add_argument('-e',
                        '--epochs',
                        help='Number of epochs',
                        type=int,
                        required=True)
    parser.add_argument('-a',
                        '--architecture',
                        help='Architecture to use (set, seq)',
                        type=str,
                        required=True)
    parser.add_argument('-t',
                        '--task',
                        help='The task (sum, mean, avg, max, 2max)',
                        type=str,
                        required=True)
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
    