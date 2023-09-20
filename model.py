import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.layers import Add, Activation, Dense, SeparableConv2D, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization


def cnn_model(input_shape):
    inputs = keras.Input(shape=input_shape)
    x = layers.Rescaling(1./255)(inputs)

    x = Conv2D(128, 3, strides=2, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    previous = x #residual to improve neuron independence

    for size in [256, 512, 728]:
        x = Activation("relu")(x)
        x = SeparableConv2D(size, 3, padding="same")(x)
        x = BatchNormalization()(x)

        x = Activation("relu")(x)
        x = SeparableConv2D(size, 3, padding="same")(x)
        x = BatchNormalization()(x)

        x = keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)

        residual = Conv2D(size, 1, strides=2, padding="same")(previous)
        x = keras.layers.add([x, residual])
        previous = x

    x = SeparableConv2D(1024, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = keras.layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = Dense(1, activation="sigmoid")(x)

    return keras.Model(inputs, outputs)
