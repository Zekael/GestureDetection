import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
import tensorboard
from dataLoader import Loader
from sklearn.model_selection import train_test_split
import datetime
import time

class ModelHandler():

    def __init__(self,input_shape=(63,)):

        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard_callbacks = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        self.model = keras.Sequential()
        self.model.add(layers.Dense(64,activation="relu",input_shape=input_shape))
        self.model.add(layers.Dropout(0.3))
        self.model.add(layers.Dense(64,activation="relu"))
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.Dense(64,activation="relu"))
        self.model.add(layers.Dense(6,activation="softmax"))


    def fit(self,training_data,testing_data,Epochs=250,batch_size=5):
        print("Training_data shape: "+str(training_data.shape))
        print("Testing_data shape: "+str(testing_data.shape))
        self.model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        self.model.fit(training_data,testing_data,callbacks=[self.tensorboard_callbacks],epochs=Epochs)


    def evaluate(self,validation_x,validation_y):
        _,accuracy = self.model.evaluate(validation_x,validation_y)
        print("Accuray: ",accuracy)

    
    def predict(self,data):
        return self.model.predict(data)


    def saveModel(self,path):
        self.model.save(path)


    def loadModel(self,path):
        self.model = keras.models.load_model(path)


    def predictionToText(self,pred):

        argmaxPred = np.argmax(pred)

        seriesMapper = {
        0:"none",
        1:"one",
        2:"yo",
        3:"okay",
        4:"halt",
        5:"five",
        }

        return seriesMapper[argmaxPred]


if __name__ == "__main__":
    trainer = ModelHandler()
    
    trainer.loadModel("./Model/usableModelV1")

    csvData = Loader.loadFromCsv("./data/collective.csv",',')
    training, testing = Loader.splitTrainTarget(csvData)
    #print(testing.head(2))
    #print(training.head())

    x_train, x_test, y_train, y_test = train_test_split(training,testing,test_size=0.2,random_state=5678, shuffle=True)

    print(x_train.head(),x_test.head(),y_train.head(),y_test.head())

    trainer.fit(x_train,y_train,Epochs=200,batch_size=20)

    trainer.evaluate(x_test,y_test)

    trainer.saveModel("./Model/model"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))