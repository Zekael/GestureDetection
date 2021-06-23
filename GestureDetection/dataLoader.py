import pandas as pd
import csv
import numpy as np


class Loader:
    """Load csv data for hand points"""

    @staticmethod
    def splitTrainTarget(loadedCsv):
       
        seriesMapper = {
        0:[1,0,0,0,0,0],
        1:[0,1,0,0,0,0],
        2:[0,0,1,0,0,0],
        3:[0,0,0,1,0,0],
        4:[0,0,0,0,1,0],
        5:[0,0,0,0,0,1]
        }

        targetDF = loadedCsv[loadedCsv.columns[0]]
        trainDF = loadedCsv[loadedCsv.columns[range(2,65)]]

        print(trainDF.head(100))
        
        newTarget = pd.DataFrame(columns=['dummy','one','peace','okay','halt','five'])


        for val in targetDF:
            try:
                array = seriesMapper[int(val)]
                newSeries = pd.Series(data=array,index=['dummy','one','peace','okay','halt','five'])
                newTarget = newTarget.append(newSeries,ignore_index=True)

            except:
                print("Error Occured")

        return trainDF.astype(float), newTarget.astype(float)


    @staticmethod
    def loadFromCsv(path,separator):
        loadedCsv = pd.read_csv(path, sep=separator , engine='python')
        print("Loaded data with shape: ",loadedCsv.shape)
        return loadedCsv



if __name__ == "__main__":
    csvFrame=Loader.loadFromCsv("./data/handData1624449714.2942007.csv",' ')
    trainingData, validationData = Loader.splitTrainTarget(csvFrame)

    print(trainingData.head(),validationData.head())

