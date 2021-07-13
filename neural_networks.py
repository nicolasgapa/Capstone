"""

Embry-Riddle Aeronautical University
Author: Nicolas Gachancipa
Name: Artificial neural networks.

"""
# Imports
import tensorflow.keras as k


# Functions.
# Out: Neurons in the last layer (i.e. number of outputs).
# input_length: Length of the input vector.
def network_5(out, input_length):
    return k.models.Sequential([k.layers.Dense(units=1491, activation='relu', input_shape=(None, input_length)),
                                # k.layers.Dense(768, activation="relu"),
                                # k.layers.LSTM(48, return_sequences=True),
                                # k.layers.LSTM(48),
                                # tf.keras.layers.Bidirectional(k.layers.LSTM(96)),
                                k.layers.Dense(384, activation="relu"),
                                # k.layers.Dense(192, activation="relu"),
                                k.layers.Dense(96, activation="relu"),
                                # k.layers.Dense(48, activation="relu"),
                                k.layers.Dense(24, activation="relu"),
                                # k.layers.Dense(12, activation="relu"),
                                k.layers.Dense(units=out, activation='softmax')])


def network_9(out, input_length):
    return k.models.Sequential([k.layers.Dense(units=768, activation='relu', input_shape=(None, input_length)),
                                k.layers.Dense(384, activation="relu"),
                                k.layers.Dense(192, activation="relu"),
                                k.layers.Dense(96, activation="relu"),
                                k.layers.Dense(48, activation="relu"),
                                k.layers.Dense(24, activation="relu"),
                                k.layers.Dense(12, activation="relu"),
                                k.layers.Dense(12, activation="relu"),
                                k.layers.Dense(units=out, activation='softmax')])


def network_7(out, input_length):
    return k.models.Sequential([k.layers.Dense(units=768, activation='relu', input_shape=(None, input_length)),
                                k.layers.Dense(384, activation="relu"),
                                k.layers.Dense(192, activation="relu"),
                                k.layers.Dense(96, activation="relu"),
                                k.layers.Dense(48, activation="relu"),
                                k.layers.Dense(24, activation="relu"),
                                k.layers.Dense(units=out, activation='softmax')])


def network_6b(out, input_length):
    return k.models.Sequential([k.layers.Dense(units=768, activation='relu', input_shape=(None, input_length)),
                                k.layers.Dense(384, activation="relu"),
                                k.layers.Dense(192, activation="relu"),
                                k.layers.Dense(96, activation="relu"),
                                k.layers.Dense(48, activation="relu"),
                                k.layers.Dense(units=out, activation='softmax')])


def network_4(out, input_length):
    return k.models.Sequential([k.layers.Dense(units=56, activation='relu', input_shape=(None, input_length)),
                                k.layers.Dense(28, activation="relu"),
                                k.layers.Dense(14, activation="relu"),
                                k.layers.Dense(units=out, activation='softmax')])
