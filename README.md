# RP-NN
## Random Projection for deep Neural Networks, Linear, CNN:
In this repository, we explain and implement a neural network models with Random Projection for Neural Networks models to
deal with data of high dimension.
This methods of random projection are mainly known to preserve distances for Linear ML algorithms such as Logistic regression and Linear Regression,
we will examine their capabilities as a layer of a Deep ANN with non-linear activations, we will also try and test which methods and which architectures yields best results,
and for what lower dimensions Random projections works.<br/>

CNNs are comprised of layers of convolutions and linear layers, we will test whether RP initialization for the CNN convolutional layers kernels would yield better results,
and whether a RP layer after the feature extraction Convolutional layers (and before the FC Linear layers) yields good results

## Databases
We would examine our work on different datasets as for the Linear NN models:
* MNIST dataset as a small examination of a popular dataset, it is a dataset of small dimension and thus we will use other high-dimensional datasets http://yann.lecun.com/exdb/mnist/.
* Condition monitoring of hydraulic systems Data Set, with 2205 examples each of 43680 attributes each http://archive.ics.uci.edu/ml/datasets/Condition+monitoring+of+hydraulic+systems.
* Gas sensor array exposed to turbulent gas mixtures Data Set, with 180 examples of 150000 attributes each http://archive.ics.uci.edu/ml/datasets/Gas+sensor+array+exposed+to+turbulent+gas+mixtures.
* CIFAR-10 for checking possible improvement on CNNs.
