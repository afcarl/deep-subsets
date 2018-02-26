# Deep Subsets
Learning deep functions on sets for set2subset transformations.

*note*: currently this project is on "hold" since I couldn't make it work that well. I will probably revisit this sometime in the future. However, most of the components to implement deep sets-style networks is available.

Most of the goodstuff is in `./src/`:

1. `datatools` contains `torch.utils.data.Dataset` objects that implement a variety of Datasets for training set2set/set2real/set2subset models. For example `numbers_data.NumbersDataset` will allow you to work with sets that contain integers. `set2real_data.py` contains datasets that produce sets of MNIST digits. 
2. `networks` contains a few neural networks that implement set2set architectures. `mnist.py` contains a convolutional context free encoder. `integer_subsets.py` implements encoders and decoders for integer sets.
3. `set_encoders` this is where we implement the layers from the paper [2]. See inline comments. 


`tests` contain unit tests for the code to ensure the layers have permutation invariance etc etc.

## Experiments

`experiments/` has most of my experiments:

1. `set2real.py` Basically trains a model that takes in subsets of MNIST digits and outputs the sum, mean, avg, max or 2max or if. The details can be found by running `python set2real.py -h`.
2. `set2subset.py` Trains a model where given a set of MNIST digits, will output a subset of them which are above the average. This is trained in a manner similar to [1]

We then move into `integer_version` tasks where the input is a set of bit representation of integers.

1. `integer_version_set2subset.py` implements the same task as 2 above (Integers --> subset above the average)
2. `integer_version_set2subset_RL.py` implements the task of selecting a subset of integers greater than the average but trained using policy gradient methods. This did not work. 

### Plots

The task here is to pick out elements of the set that are above the average. This clearly demonstrates the usefulness of the "set-level" aggregation functions proposed in [2]. However, it also demonstrates that there might be some weaknesses of the model for these kinds of subset selection tasks.

#### MNIST Subset performance

![image](https://user-images.githubusercontent.com/6295292/36649753-8276345c-1a6d-11e8-8fdd-928bb273b1b3.png)

This shows that the DeepSubsets based model (Context in the graph) has some generalization capability (when you go beyond the training regime). It also shows that we need layers like equation 11 from [2] to really allow any kind of "set-level" reasoning to occur (No Context). The null model is one that just predicts a selection for the elements that are above 5. This is why as the set size becomes larger, and the average of the set goes to 5, the accuracy goes higher. Since the Deep-subsets model shows a decreasing performance, it seems to suggest that it is not learning something useful about the task.

#### Integer Subset performance (Supervised Learning)

![image](https://user-images.githubusercontent.com/6295292/36649871-2334e33e-1a6e-11e8-805e-d053a961b3c6.png)




# References
[1] [Raffel, Colin, and Dieterich Lawson. "Training a subsampling mechanism in expectation." arXiv preprint arXiv:1702.06914 (2017).](http://colinraffel.com/publications/iclr2017training.pdf)

[2] [Zaheer, Manzil, et al. "Deep sets." Advances in Neural Information Processing Systems. 2017.](http://papers.nips.cc/paper/6931-deep-sets.pdf)
