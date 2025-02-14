import os

import pandas as pd

from classes.parameters import MyParameters

class MyDataFrame:

    def formatData(self, accuracyScore, usedMetamorphic, usedParameters, dateAndTime):

        myParameters = MyParameters()

        formattedData = {'Feature_Map': [myParameters.featureMaps[usedParameters['featureMapType']]], 
                'Data_Type':[myParameters.allDataTypes[usedParameters['dataType']]], 
                'PCA_Components': [usedParameters['components']],
                'Accuracy_Score': [accuracyScore],
                'Used_Metamorphic': [usedMetamorphic],
                'Apply_Scalar_value': [myParameters.applyScalarValue],
                'Scalar_Value': [myParameters.scaleValue if myParameters.applyScalarValue else 1],
                'Apply_Rotation_Angle': [myParameters.applyAngleRotation],
                'Angle_Rotation': [myParameters.angle if myParameters.applyAngleRotation else 0],
                'Apply_Permutation': [myParameters.applyPermutation],
                'Invert_Labels': [myParameters.invertAllLabels],
                'Apply_Perturb_Noise': [myParameters.applyPerturbNoise],
                'Perturb_Noise': [myParameters.perturbNoise if myParameters.applyPerturbNoise else 0],
                'Modify_Circuit_Depth': [myParameters.modifyCircuitDepth],
                'adding_Additional_Feature': [myParameters.addAdditionalFeature],
                'Adding_Additional_Data_Point': [myParameters.addAdditionalInputsAndOutputs],

                'Date_And_Time': [dateAndTime],
                }
        
        return formattedData



    def processToDataFrame(self, data):
        myDataFrame = pd.DataFrame(data)

        MyDataFrame.saveDataFrame(myDataFrame)


    def saveDataFrame(myDataFrame):

        # myDataFrame.index = myDataFrame.index + 1

        lastIndex = MyDataFrame.getDataIndex()

        filePath = MyDataFrame.getFilePath()

        # print('lastIndex: ', lastIndex)

        myDataFrame.index = range(lastIndex + 1, lastIndex + 1 + len(myDataFrame))

        # print('lastIndex: ', lastIndex)

        if lastIndex < 0: 
            myDataFrame.to_csv(filePath, mode='a', index=True, header = True)
        else:
            myDataFrame.to_csv(filePath, mode='a', index=True, header = False)
  

    def getDataIndex():

            filePath = MyDataFrame.getFilePath()
            try:
                # Read the existing CSV file
                existingDataFrame = pd.read_csv(filePath)
                last_index = existingDataFrame.index[-1]  # Get the last index
            except FileNotFoundError:

                print
                # If the file doesn't exist, start from index 0
                last_index = -1  # Start from 0 for new data

            return last_index
            # return range(last_index + 1, last_index + 1 + len(new_df))


    
    def getFilePath():

        current_directory = os.getcwd()  # Get the current working directory
        # print('current_directory: ', current_directory)

        # parent_directory = os.path.dirname(current_directory)  # Get the parent directory
        # print('parent_directory: ', parent_directory)

        # target_folder = os.path.join(parent_directory, 'saved_data') 
        
        target_folder = os.path.join(current_directory, 'saved_data')

         # Construct the target folder path
        # print('target_folder: ', target_folder)

        file_path = os.path.join(target_folder, 'my_dataframe.csv')

        # print('file_path is:', file_path)

        return file_path
    
        


