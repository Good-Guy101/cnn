import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.layers import Add, Activation, Dense, SeparableConv2D, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
import cv2

from model import cnn_model
from data import get_data, preprocess

def main():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
        # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
            print(e)


    #Get xray images
    train_dataset = tf.keras.utils.image_dataset_from_directory("./chest_xray/train",
                                                            color_mode = 'grayscale',
                                                            batch_size = 8,
                                                            interpolation = 'bicubic'
                                                           )

    test_dataset = tf.keras.utils.image_dataset_from_directory("./chest_xray/test",
                                                            color_mode = 'grayscale',
                                                            batch_size = 8,
                                                            interpolation = 'bicubic'
                                                           )
    val_dataset = tf.keras.utils.image_dataset_from_directory("./chest_xray/val",
                                                            color_mode = 'grayscale',
                                                            batch_size = 8,
                                                            interpolation = 'bicubic'
                                                           )

    #Create model
    model = cnn_model((256,256,1))

    model.summary()
    keras.utils.plot_model(model, show_shapes=True)

    #Train model
    num_epochs = 13

    callbacks = [
        keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras"),
    ]

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        train_dataset,
        epochs =  num_epochs,
        validation_data = val_dataset,
        callbacks = callbacks
    )


    test_results = model.evaluate(test_dataset)
    print("Loss of the model is - " , test_results[0])
    print("Accuracy of the model is - " , test_results[1]*100 , "%")

if __name__ == "__main__":
    main()
