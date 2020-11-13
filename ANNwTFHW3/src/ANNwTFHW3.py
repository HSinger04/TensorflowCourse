import matplotlib.pyplot as plt
import numpy as np
# TODO: uncomment for Jupyter
# %tensorflow_version 2.x
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import activations
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model


class MLP(Model):

    def __init__(self):
        super(MLP, self).__init__()
        self.hidden_layer1 = Dense(256, activation=activations.sigmoid)
        self.hidden_layer2 = Dense(256, activation=activations.sigmoid)
        self.out = Dense(10, activation=activations.softmax)

    @tf.function
    def call(self, x):
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.out(x)
        return x


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
    accuracy = None

    # loss_object and optimizer_object are instances of respective tensorflow classes
    with tf.GradientTape() as tape:
        prediction = model(input)
        accuracy = np.argmax(target, axis=1) == np.argmax(prediction, axis=1)
        loss = loss_function(target, prediction)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    accuracy = np.mean(accuracy)

    return loss, accuracy


def test(model, test_data, loss_function):
    # test over complete test data

    test_accuracy_aggregator = []
    test_loss_aggregator = []

    for (input, target) in test_data:
        prediction = model(input)
        sample_test_loss = loss_function(target, prediction)
        sample_test_accuracy = np.argmax(target, axis=1) == np.argmax(prediction, axis=1)
        sample_test_accuracy = np.mean(sample_test_accuracy)
        test_loss_aggregator.append(sample_test_loss.numpy())
        test_accuracy_aggregator.append(sample_test_accuracy)

    test_loss = np.mean(test_loss_aggregator)
    test_accuracy = np.mean(test_accuracy_aggregator)

    return test_loss, test_accuracy


def plot_stats(stat_list, stat_name):
    plt.figure()
    line, = plt.plot(stat_list)
    plt.xlabel("Training steps")
    plt.ylabel(stat_name)
    plt.show()


if __name__ == "__main__":

    # define some constants
    PREFETCH_SIZE = 5
    BATCH_SIZE = 100

    def data_pipeline(data):
        """ helper function for data pipeline - does all the things we need

        :param data:
        :return:
        """
        # take only 1/10 of original data
        # TODO: change shard to 10 again. 1000 only for faster debugging / testing.
        data = data.shard(10000, 0)
        data = data.map(onehotify)
        data = data.batch(BATCH_SIZE)
        # unsure if shuffle is needed here, but they did it in Tensorflow_Intro.ipynb so...
        # buffer_size and batch_size don't have to be the same btw., but we pick the same in this case
        data = data.shuffle(buffer_size=BATCH_SIZE)
        data = data.prefetch(PREFETCH_SIZE)
        return data


    # as_supervised results in only returning domain and labels and leaving out other unnecessary info
    genomics_data = tfds.load('genomics_ood', split=['train', 'test'], as_supervised=True)
    # each entry is a batch of size BATCH_SIZE
    train_data = data_pipeline(genomics_data[0])
    test_data = data_pipeline(genomics_data[1])

    tf.keras.backend.clear_session()

    model = MLP()
    # TODO: change num_epochs back to 10 for actual implementation
    num_epochs = 2
    learning_rate = 0.1
    running_average_factor = 0.95
    loss = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.SGD(learning_rate)

    # Initialize lists for later visualization.
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    # define calc_stat as a local function to access running_average_factor
    def calc_stat(run_avg, raw_stat):
        """Calculates appropriate statistic

        :param run_avg: current running average
        :param raw_stat: the raw statistic e.g. test_loss
        :return: if 0, return the raw statistic, else apply running average
        """

        if running_average_factor:
            return running_average_factor * run_avg + (1 - running_average_factor) * raw_stat
        return raw_stat


    for epoch in range(num_epochs):
        print('Epoch: ' + str(epoch))

        # shuffle train data. No need to shuffle test data
        train_data = train_data.shuffle(buffer_size=BATCH_SIZE)

        train_loss_stat = 0
        train_accuracy_stat = 0
        for (input, target) in train_data:
            train_loss, train_accuracy = train_step(model, input, target, loss, optimizer)
            train_loss_stat = calc_stat(train_loss_stat, train_loss)
            train_accuracy_stat = calc_stat(train_accuracy_stat, train_accuracy)
            train_losses.append(train_loss_stat)
            train_accuracies.append(train_accuracy_stat)

        # testing
        test_loss, test_accuracy = test(model, test_data, loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

    # plot train and test stats differently since they have different training sizes
    plot_stats(train_losses, "Training Loss")
    plot_stats(test_losses, "Testing Loss")
    plot_stats(train_accuracies, "Training Accuracy")
    plot_stats(test_accuracies, "Testing Accuracy")
