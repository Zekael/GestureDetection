import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
import tensorboard

class ModelTrainer():

    def __init__(self,input_shape=(65,)):
        self.model = keras.Sequential()
        self.model.add(layers.Dense(128,activation="relu"),input_shape=input_shape)
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.Dense(128,activation="relu"))
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.Dense(128,activation="relu"))
        self.model.add(layers.Dense(4,activation="softmax"))


    def fit(self,training_data,testing_data,Epochs=30,batch_size=10):
        print("Training_data shape: "+str(training_data.shape))
        print("Testing_data shape: "+str(testing_data.shape))
        self.model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy())
        tensorboard_callbacks = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        self.model.fit(training_data,testing_data,callbacks=[tensorboard_callbacks])


    def evaluate(self,validation_x,validation_y):
        _,accuracy = self.model.evaluate(validation_x,validation_y)



if __name__ == "__main__":
    trainer = handGestureTrainer()

    data = pd.read_csv()