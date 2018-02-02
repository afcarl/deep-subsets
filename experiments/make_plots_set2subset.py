"""
Hacky script for now to draw a quick plot
"""
import matplotlib
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import torch
import argparse

parser = argparse.ArgumentParser('Quick Plot')
parser.add_argument('-gt5', '--gt5null', required=True)
parser.add_argument('-cf', '--contextfree', required=True)
parser.add_argument('-cb', '--contextbased', required=True)
parser.add_argument('-m', '--metric', required=True)
args = parser.parse_args()
sns.set_style('white')
sns.set_palette('colorblind')
sys.path.append('..')

NULL_RESULTS = torch.load(args.gt5null+'./results.json')
CF_RESULTS = torch.load(args.contextfree+'./results.json')
CE_RESULTS = torch.load(args.contextbased+'./results.json')
SET_SIZES = np.arange(4, 100)
TO_PLOT = args.metric
plt.figure(figsize=(10, 4))
plt.plot(SET_SIZES, CE_RESULTS[TO_PLOT], label='Context Based Model')
plt.plot(SET_SIZES, CF_RESULTS[TO_PLOT], label='Context Free Model')
plt.plot(SET_SIZES, NULL_RESULTS[TO_PLOT], label='Null Model')
plt.legend()
plt.ylim([0, 1])
plt.plot(np.ones_like(SET_SIZES)*10, np.linspace(0, 1, num=len(SET_SIZES)), '--', color='grey')
plt.text(0.1, 0.4, 'Training \nSets Sizes')
plt.text(11, 0.4, 'Test \nSets Sizes')
plt.title('Subset Selection Accuracy')
plt.savefig('{}.pdf'.format(args.metric))