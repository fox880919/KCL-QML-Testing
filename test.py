# import datetime

# now = datetime.datetime.now()

from classes.time import MyTimeHelper

from data.my_dataframe import MyDataFrame

dateAndTime = MyTimeHelper().getTimeNow()

print('type of dateAndTime: ', type (dateAndTime))

formattedData= {
    'Feature_Map': ['Amplitude Embedding'], 
    'Data_Type': ['Wine Data'], 
    'PCA_Components': [8], 
    'Accuracy_Score': [1], 
    'Used_Metamorphic': [False], 
    'Date_And_Time': [dateAndTime]
}

print('before saving data')

myDataFrame = MyDataFrame()
myDataFrame.processToDataFrame(formattedData)

