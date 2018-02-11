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
import torch.nn.functional as F
import numpy as np
from src.networks.integer_subsets import IntegerSubsetNet
from src.datatools import IntegersLargerThanAverage, RLWrapper, IntegerSubsetsSupervised
from src.util_io import create_folder
from src.metrics import set_accuracy
from pg_methods.utils.baselines import MovingAverageBaseline
from pg_methods.utils.policies import BernoulliPolicy
from pg_methods.utils import gradients
from torch.nn.utils import clip_grad_norm

increase_every = 1
tau = 0.5 # activation probability
#lambda1 = 1
#lambda2 = 1

def main(args):
    CUDA = False
    folder_name = 'RL_'+args.name + '_' + args.task + '_' + args.architecture
    folder_path = os.path.join('./', folder_name)
    create_folder(folder_name)
    datasets = [IntegersLargerThanAverage(10000, i, 10) for i in range(4, 5)]
    critic = MovingAverageBaseline(0.9)
    if args.architecture == 'set':
        policy = BernoulliPolicy(IntegerSubsetNet())
    elif args.architecture == 'null':
        policy = BernoulliPolicy(IntegerSubsetNet(null_model=True))
    else:
        raise ValueError('Unknown architecture. Must be set or null!')

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4, eps=1e-1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5000, gamma=0.9) 
    if torch.cuda.is_available() and args.gpu != '':
        policy.cuda()
        CUDA = True
        print('Using GPU')

    environment = RLWrapper(datasets, 64, use_cuda=CUDA)
    data = environment.reset()
    for n in range(args.n_episodes):  # run for epochs

        actions, log_prob_actions = policy(data)
        #policy_p = F.sigmoid(policy.fn_approximator(data))
        log_prob_actions = log_prob_actions.sum(1)
        baseline = critic(data).view(-1, 1)

        # print(baseline.size())
        # print(actions.size())
        # print(log_prob_actions.size())
        # print(.size())

        data, reward, _, info = environment.step(actions)
        advantage = reward - baseline
        # print(advantage.size())
        # print(reward.size())
        critic.update_baseline(reward, advantage, baseline)
        loss = gradients.calculate_policy_gradient_terms(log_prob_actions, advantage)
        loss = loss.mean() # mean is fine since there is only really "one action"?
        # print(loss)
#        term1 = (policy_p.mean(0) - tau).sum() # regularization to activate each unit with probability tau
 #       term2 = (policy_p.mean(1) - tau).mean() # regularization to be a bit sparse
 #       term3 = ((policy_p - policy_p.mean(0))**2).sum()

 #       regularized_loss = loss + lambda1*(term1 + term2) * lambda2*term3
        optimizer.zero_grad()
#        regularized_loss.backward()
        loss.backward()
        clip_grad_norm(policy.fn_approximator.parameters(), 40)
        optimizer.step()
        scheduler.step()
        if n % 100 == 0:
            # print("")
            print('{}: loss {:3g}, episode_reward {:3g}'.format(n, loss.cpu().data[0], reward.mean()))
            y_target = torch.FloatTensor(environment.current_dataset.supervised_objective(data.data.int()))
            set_acc, elem_acc = set_accuracy(y_target, policy(data)[0].data.squeeze())
            print('  : set acc: {}, elem_acc: {}, set_size {}'.format(set_acc, elem_acc, environment.current_dataset.set_size))


    # now put this into "supervised" mode
    datasets = [
        (i, torch.utils.data.DataLoader(
            IntegerSubsetsSupervised(256, i, 10, target='mean', seed=5),
            batch_size=256))
        for i in range(4, 10)
    ]

    set_sizes = []
    mse = []
    set_accs = []
    elem_accs = []
    torch.save(policy, os.path.join(folder_path, 'model-gpu.pyt'))
    criterion = torch.nn.BCELoss()
    for set_size, dataset in datasets:
        for i, (x, y) in enumerate(dataset):
            # prepare the data
            if CUDA:
                x = x.cuda()
                y = y.cuda()
            x, y = Variable(x, volatile=True), Variable(y, volatile=True).float()

            # run it through the network
            y_hat, _ = policy(x)
            y_hat = y_hat.view_as(y)
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
    policy.cpu()
    torch.save({'set_sizes': set_sizes,
                'mse': mse,
                'set_acc': set_accs,
                'elem_accs': elem_accs,
                'mean_acc': torch.FloatTensor(set_accs).mean()}, os.path.join(folder_path, 'results.json'))
    torch.save(policy, os.path.join(folder_path, 'model.pyt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('set2subset experiments')
    parser.add_argument('-n_episodes',
                        '--n_episodes',
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
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    main(args)
