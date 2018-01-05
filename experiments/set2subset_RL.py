import torch
import numpy as np
import sys
import random
import json
sys.path.append(os.path.abspath('..'))
import argparse
# sys.path.append('../subset-sum')
from src.datatools import SubsetSum
from src.util_io import create_folder
from src.networks.integer_subsets import IntegerSubsetNet

# from subsetsum import stackoverflow, wikipedia, hetland

from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn.functional import sigmoid
from torch.optim import Adam, lr_scheduler

# create some different datasets for training:
set_sizes = list(range(5, 15))
ss_datasets = [SubsetSum(10000, i, 5, target=10, empty_subset_reward=-100) for i in set_sizes]
dataloaders = [DataLoader(ss_dset, batch_size=128) for ss_dset in ss_datasets ]

def main(args):
    CUDA = False
    folder_name = args.name+'_'+args.task+'_'+args.architecture
    folder_path = os.path.join('./', folder_name)
    create_folder(folder_name)
    sums = []
    EPOCHS = 500
    beta = 0.9
    increase_every = 5
    net = IntegerSubsetNet(logprobs=True)


    if torch.cuda.is_available() and args.gpu != '':
        net.cuda()
        CUDA = True
        print('Using GPU')

    optimizer = Adam(net.parameters(), lr=0.0001)
    scheduler = lr_scheduler.StepLR(optimizer, 10)
    critic = 0
    rewards = []
    losses = []
    advantages = []
    for epoch in range(EPOCHS):
        ss = random.sample(dataloaders[:math.ceil((epoch+1)/increase_every)], 1)[0]
        ss.dataset.refresh_dataset()
        for i, x in enumerate(ss):
            if CUDA:
                x = x.cuda()
            x = Variable(x.float())
            log_probs = net(x)
    #         log_probs = log_probs[0].permute(1, 0, 2)
            policy = torch.distributions.Bernoulli(sigmoid(log_probs))
            actions = policy.sample()
            R = ss.dataset.reward_function(x.data.byte(), actions.data.byte())
            
            if CUDA:
                R = R.cuda()

            if i == 0:
                critic = R.mean()
            else:
                critic = critic*beta + (1-beta)*R.mean()
            advantage = Variable(R - critic).view(-1)
            loss = -(policy.log_prob(actions) * advantage).sum()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(net.parameters(), float(2), norm_type=2)
            optimizer.step()
            if i % 128 == 0:     
                print('epoch:',epoch,'advantage:', advantage.data[0], 'loss:', loss.data[0])
        scheduler.step()
        rewards.append(R.mean())
        advantages.append(advantage.data[0])
        losses.append(loss.data[0])
        
        if epoch % 50 == 0:
            subsets = ss.dataset.subset_elements(x.data.byte(), actions.data.byte(), bit_representation=True)
            sets = ss.dataset.bit_array_to_int_array(x.data.byte()).tolist()
            pairs = list(zip(sets, subsets))[:5]
            _sums = []
            for x, y in pairs:
                print('full set:', x)
                print('subset:', y)
                _sums.append(sum(y))
                print('sum:', _sums[-1])
            sums.append(np.mean(_sums))


    with open(os.path.join(folder_path, 'training_data.json'), 'w') as f:
        json.dump(
            dict(sums=sums,
                rewards=rewards,
                advantages=advantages,
                losses=losses), f)

    torch.save(net, os.path.join(folder_path, 'model-gpu.pyt'))
    

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
