"""
Unsupervised pre-training is a technique to train the network in case of scarcity of labeled data.
The DNN layers are trained one by one, keeping the previously trained layer intact.
The layers are trained using Autoencoders, before 2010 its used to be trained using RBMs.
Once, the hidden layers are trained, just train the last and output layer
We will implement this for MNIST data, just to understand the process and implementation complications.
"""


import tensorflow as tf
import numpy as np
import os