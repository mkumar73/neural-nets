"""
The exercise is to implement CIFAR object classification from scratch.
We will use the CIFAR10 dataset from tf.keras.datasets.cifar10.
Main idea is to build a CNN to classify correctly the images.
We will try to use all the applicable optimization techniques as mentioned below:
1. Gradient descent variants: Adam, RMSProp, SGD, SGD+Momentum etc.
2. Weight initialization: He initialization, Xaviers initialization
3. Batch Normalization (BN)
4. Early Stopping
5. Dropouts
6. and so on.
"""

import tensorflow as tf
import numpy as np
import  matplotlib.pyplot as plt
import os
