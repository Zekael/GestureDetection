import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
import tensorboard

class handGestureTrainer():

    def __init__(self,input_shape=(65,)):
        self.model = keras.Sequential()
        self.model.add(layers.Dense(128,activation="relu"),input_shape=input_shape)
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.Dense(128,activation="relu"))
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.Dense(128,activation="relu"))
        self.model.add(layers.Dense(4,activation="softmax"))


    def train(self,training_data,testing_data):
        self.model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy())
        self.model.fit(training_data,testing_data)





if __name__ == "__main__":
    trainer = handGestureTrainer()