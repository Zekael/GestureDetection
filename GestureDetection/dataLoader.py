import pandas as pd
import csv
import numpy as np



class Loader:
    """Load csv data for hand points"""


    @staticmethod
    def splitTrainTarget(loadedCsv):
       
        cols = [0,1]
        targetDF = loadedCsv[loadedCsv.columns[cols]]
        trainDF = loadedCsv[loadedCsv.columns[range(2,65)]]
        
        newTarget = pd.DataFrame(columns=['dummy','fist','point','thumbs up','hand'])

        for index,row in targetDF.iterrows():
            
            handVal = row[1]

            try:
                if int(row[0])==1:
                    newRow = pd.Series([0,1,0,0,handVal],index=['dummy','fist','point','thumbs up','hand'])
                    #print(newRow)
                    newTarget = newTarget.append(newRow,ignore_index=True)

                elif int(row[0])==2:
                    newRow = pd.Series([0,0,1,0,handVal],index=['dummy','fist','point','thumbs up','hand'])
                    #print(newRow)
                    newTarget =newTarget.append(newRow,ignore_index=True)

                elif int(row[0])==3:
                    newRow = pd.Series([0,0,0,1,handVal],index=['dummy','fist','point','thumbs up','hand'])
                    #print(newRow)
                    newTarget =newTarget.append(newRow,ignore_index=True)

                else:
                    newRow = pd.Series([1,0,0,0,handVal],index=['dummy','fist','point','thumbs up','hand'])
                    #print(newRow)
                    newTarget =newTarget.append(newRow,ignore_index=True)

            except:
                print("Error occured at ",index,row)
                

        return trainDF.astype(float), newTarget.astype(float)


    @staticmethod
    def loadFromCsv(path,separator):
        loadedCsv = pd.read_csv(path, sep=separator , engine='python')
        print("Loaded data with shape: ",loadedCsv.shape)
        return loadedCsv



if __name__ == "__main__":
    csvFrame=Loader.loadFromCsv("./data/handData1624351939.2806554.csv",';')
    trainingData, validationData = Loader.splitTrainTarget(csvFrame)

    print(trainingData.head(),validationData.head())

