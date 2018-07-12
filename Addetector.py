import pandas as p
import numpy as n

dataSet='C:/Users/Mohan babu/Desktop/IIT_Fall_2017/Summer 2018/AI/Ad Dataset/ad.data'
data=p.read_csv(dataSet,sep=",",header=None,low_memory=False)
data.head(20)

#toNum can only be used for individual cells and not for groups of cells
def toNum(cell):
  try:
        return n.float(cell)
  except:
        return n.nan
    
#Apply this to every column in the Pandas series/column using "apply" function. apply function can be used for group of cells
def seriestoNum(series):
    return series.apply(toNum)

trainData=data.iloc[0:,0:-1].apply(seriestoNum)
trainData.head(30)

#dropping the NA rows
trainData=data.iloc[0:,0:-1].dropna()
trainData.head(30)

def toLabel(str):
  if str=="ad.":
        return 1
  else:
        return 0
    
trainLabel=data.iloc[trainData.index,-1].apply(toLabel)
trainLabel.head(30)

#Training phase and fit function is used to 
from sklearn.svm import LinearSVC

classi = LinearSVC()
classi.fit(trainData[100:2300],trainLabel[100:2300])

#predict function is used to identify the ad or not part
classi.predict(trainData.iloc[13].values.reshape(1,-1))