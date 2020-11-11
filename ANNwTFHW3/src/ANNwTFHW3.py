# TODO: import necessary libraries
import numpy as np
%tensorflow_version 2.x
import tensorflow as tf
import matplotlib.pyplot as plt

# TODO: Load genomic data set
# use https://www.tensorflow.org/datasets/api_docs/python/tfds/load. Just use string for loading
#(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# TODO: investigate data set

# TODO: data pipeline:
# TODO: only take 100k for training and 1k for testing
# TODO: Transform into one-hot encoding
# TODO: Batching
# TODO: Prefetching

# TODO: Define the model using keras, Arguments: Learning rate;
#  Hidden layers have sigmoid activation function,
#  Output neuron has softmax activation with 10 output units


# TODO: Training: epochs: 10, loss: categorical cross entropy, optimizer: SGD
#  Record loss and accuracy
# TOASK: What exactly is an epoch?

# TODO: Visualize accuracy and loss for
#  training AND test data using matplotlib