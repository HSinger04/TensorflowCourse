# TODO: reorder functions and class etc.
# TODO: Optimize with e.g. decorators. Check performance by running only for short epoch
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

    @tf.function
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
        # TODO: Why should we transform one hot vector back to such a weird form?
        onehot = tf.reshape(onehot, (-1,))
    label = tf.one_hot(label, 10)
    return onehot, label


def train_step(model, input, target, loss_function, optimizer):

    accuracy = None

    # loss_object and optimizer_object are instances of respective tensorflow classes
    with tf.GradientTape() as tape:
        prediction = model(input)
        # TODO: Confirm accuracy is of size batch
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
        # TODO: it seems that accuracy is fine this way, but not 100% sure. Is the whole batch.
        sample_test_accuracy = np.argmax(target, axis=1) == np.argmax(prediction, axis=1)
        sample_test_accuracy = np.mean(sample_test_accuracy)
        test_loss_aggregator.append(sample_test_loss.numpy())
        # FIXME: removed a np.mean around sample_test_accuracy. Should be fine
        test_accuracy_aggregator.append(sample_test_accuracy)

    # TODO: Note to myself: np.mean should stay here
    test_loss = np.mean(test_loss_aggregator)
    test_accuracy = np.mean(test_accuracy_aggregator)

    return test_loss, test_accuracy

# TODO
def plot_stats(train_stat, test_stat, stat_name):
    plt.figure()
    line1, = plt.plot(train_stat)
    line2, = plt.plot(test_stat)
    plt.xlabel("Training steps")
    plt.ylabel(stat_name)
    plt.legend((line1, line2), ("training", "test"))
    plt.show()


def data_pipeline(data, batch_size=100):
    # take only 1/10 of original data
    # TODO: change shard to 10 again. 1000 only for faster debugging / testing.
    data = data.shard(10000, 0)
    data = data.map(onehotify)
    data = data.batch(batch_size)
    # unsure if shuffle is needed here, but they did it in Tensorflow_Intro.ipynb so...
    # buffer_size and batch_size don't have to be the same btw., but we pick the same in this case
    # TODO: Prefetching
    data = data.shuffle(buffer_size=batch_size)
    return data

if __name__ == "__main__":
    BATCH_SIZE = 100
    # as_supervised results in only returning domain and labels and leaving out other unnecessary info
    genomics_data = tfds.load('genomics_ood', split=['train', 'test'], as_supervised=True)
    # each entry is a batch of size BATCH_SIZE
    train_data = data_pipeline(genomics_data[0], BATCH_SIZE)
    test_data = data_pipeline(genomics_data[1], BATCH_SIZE)
    # TODO: figure out if data is already prefetched or not

    # TODO: only take 100k for training and 1k for testing

    tf.keras.backend.clear_session()

    model = MLP()
    # TODO: change num_epochs back to 10 for actual implementation
    num_epochs = 2
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

        # shuffle train data. No need to shuffle test data
        train_data = train_data.shuffle(buffer_size=BATCH_SIZE)

        for (input, target) in train_data:
            train_loss, train_accuracy = train_step(model, input, target, loss, optimizer)
            # TODO: Maybe replace with running average like in Tensorflow_intro
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            # TODO: train_accuracies

        # testing
        test_loss, test_accuracy = test(model, test_data, loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

    plot_stats(train_losses, test_losses, "Loss")
    plot_stats(train_accuracies, test_accuracies, "Accuracy")
