# TODO: reorder functions and class etc.
import numpy as np
# TODO: uncomment for Jupyter
# %tensorflow_version 2.x
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras import activations
import matplotlib.pyplot as plt


class MLP(Model):

    def __init__(self):
        super(MLP, self).__init__()
        self.hidden_layer1 = Dense(256, activation=activations.sigmoid)
        self.hidden_layer2 = Dense(256, activation=activations.sigmoid)
        self.out = Dense(10, activation=activations.softmax)

    def call(self, x):
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.out(x)
        return x


# TODO: Visualize accuracy and loss for
#  training AND test data using matplotlib

def onehotify(tensor, label):
    vocab = {'A': '1', 'C': '2', 'G': '3', 'T': '0'}
    for key in vocab.keys():
        tensor = tf.strings.regex_replace(tensor, key, vocab[key])
        split = tf.strings.bytes_split(tensor)
        labels = tf.cast(tf.strings.to_number(split), tf.uint8)
        onehot = tf.one_hot(labels, 4)
        onehot = tf.reshape(onehot, (-1,))
    label = tf.one_hot(label, 10)
    return onehot, label


def train_step(model, input, target, loss_function, optimizer):
    # loss_object and optimizer_object are instances of respective tensorflow classes
    with tf.GradientTape() as tape:
        prediction = model(input)
        loss = loss_function(target, prediction)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


def test(model, test_data, loss_function):
    # test over complete test data

    test_accuracy_aggregator = []
    test_loss_aggregator = []

    for (input, target) in test_data:
        prediction = model(input)
        sample_test_loss = loss_function(target, prediction)
        # TODO: change accuracy
        sample_test_accuracy = np.argmax(target, axis=1) == np.argmax(prediction, axis=1)
        sample_test_accuracy = np.mean(sample_test_accuracy)
        test_loss_aggregator.append(sample_test_loss.numpy())
        # TODO: is the np.mean here even necessary?
        test_accuracy_aggregator.append(np.mean(sample_test_accuracy))

    # TODO: is the np.mean here even necessary?
    test_loss = np.mean(test_loss_aggregator)
    # TODO: is the np.mean here even necessary?
    test_accuracy = np.mean(test_accuracy_aggregator)

    return test_loss, test_accuracy


"""
# Visualize accuracy and loss for training and test data. 
# One plot training and test loss.
# One plot training and test accuracy.
plt.figure()
line1, = plt.plot(train_losses)
line2, = plt.plot(test_losses)
plt.xlabel("Training steps")
plt.ylabel("Loss")
plt.legend((line1,line2),("training","test"))
plt.show()

plt.figure()
line1, = plt.plot(test_accuracies)
plt.xlabel("Training steps")
plt.ylabel("Accuracy")
plt.show()
"""


def data_pipeline(data, train, batch_size=128):
    # take only 1/10 of original data
    data = data.shard(10, 0)
    data = data.map(onehotify)
    # only apply following processes on training data
    # TOASK: Maybe, one needs to also do batch and shuffling to test data. Compare with Tensorflow_intro
    if train:
        data = data.batch(batch_size)
    return data

if __name__ == "__main__":
    # TODO: Change train_input into normal tf.data.Dataset object
    # use https://www.tensorflow.org/datasets/api_docs/python/tfds/load. Just use string for loading
    # TODO: Interestingly, setting the batch size to 128 destroys the whole thing.
    genomics_data = tfds.load('genomics_ood', split=['train', 'test'], as_supervised=True)
    train_data = data_pipeline(genomics_data[0], True)
    test_data = data_pipeline(genomics_data[1], False)
    # TODO: figure out if data is already prefetched or not
    # TODO: figure out if batch size is 128 or not

    # TODO: investigate data set

    # TODO: data pipeline:
    # TODO: only take 100k for training and 1k for testing
    # TODO: Transform into one-hot encoding
    # TODO: Batching
    # TODO: Prefetching

    # TODO: Make sure that I have 100k training
    model = MLP()
    num_epochs = 10
    learning_rate = 0.1
    loss = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.SGD(learning_rate)

    # Initialize lists for later visualization.
    train_losses = []
    # TODO: Also track train accuracy
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    for epoch in range(num_epochs):
        print('Epoch: ' + str(epoch))

        # TOASK: Maybe, one needs to also shuffling to test data. Compare with Tensorflow_intro
        train_data = train_data.shuffle(buffer_size=128)

        for (input, target) in train_data:
            train_loss = train_loss = train_step(model, input, target, loss, optimizer)
            # TODO: Maybe replace with running average like in Tensorflow_intro
            train_losses.append(train_loss)

        # testing
        test_loss, test_accuracy = test(model, test_data, loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

    # Visualize accuracy and loss for training and test data.
    # One plot training and test loss.
    # One plot training and test accuracy.
    plt.figure()
    line1, = plt.plot(train_losses)
    line2, = plt.plot(test_losses)
    plt.xlabel("Training steps")
    plt.ylabel("Loss")
    plt.legend((line1, line2), ("training", "test"))
    plt.show()

    plt.figure()
    line1, = plt.plot(test_accuracies)
    plt.xlabel("Training steps")
    plt.ylabel("Accuracy")
    plt.show()