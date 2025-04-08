import os

import pandas as pd

from classes.parameters import MyParameters

class MyDataFrame:

    def formatData(self, accuracyScore, usedMetamorphic, usedParameters, dateAndTime, foldIndex):

        # myParameters = MyParameters()

        formattedData = {
                # 'Feature_Map': [MyParameters.featureMaps[usedParameters['featureMapType']]], 
                # 'Data_Type':[MyParameters.allDataTypes[usedParameters['dataType']]], 
                'Feature_Map': [MyParameters.featureMaps[MyParameters.featureMapType]], 
                'Data_Type':[MyParameters.allDataTypes[MyParameters.dataType]],                 
                'PCA_Components': [MyParameters.pca_components],
                'Accuracy_Score': [accuracyScore],
                'n_fold': MyParameters.n_folds,
                'fold_index': foldIndex,
                'Used_Metamorphic': [usedMetamorphic],
                'Apply_Scalar_value': [MyParameters.applyScalarValue],
                'Scalar_Value': [MyParameters.scaleValue if MyParameters.applyScalarValue else 1],
                'Apply_Rotation_Angle': [MyParameters.applyAngleRotation],
                'Angle_Rotation': [MyParameters.angle if MyParameters.applyAngleRotation else 0],
                'Apply_Permutation': [MyParameters.applyPermutation],
                'Invert_Labels': [MyParameters.invertAllLabels],
                'Apply_Perturb_Noise': [MyParameters.applyPerturbNoise],
                'Perturb_Noise': [MyParameters.perturbNoise if MyParameters.applyPerturbNoise else 0],
                'Modify_Circuit_Depth': [MyParameters.modifyCircuitDepth],
                'adding_Additional_Feature': [MyParameters.addAdditionalFeature],
                'Adding_Additional_Data_Point': [MyParameters.addAdditionalInputsAndOutputs],

                'Date_And_Time': [dateAndTime],
                }
        
        return formattedData



    def processToDataFrame(self, data):
        myDataFrame = pd.DataFrame(data)

        # don't use in testing
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

        file_path = os.path.join(target_folder, MyParameters.savingFileName)

        # print('file_path is:', file_path)

        return file_path
    
        
    #not used
    def getDataFrameByName(self, modelName):


        df = pd.read_csv('saved_data/my_dataframe.csv')

        filtered_row = df[df['Name'] == modelName]

        if not filtered_row.empty:
            
            return filtered_row
        
        else:

            print(f"No data found for the name: {modelName}")

    def getModelScoreValue(self, mr, value, kfoldIndex, nfold):

        extraDigit = ''

        if mr< 10:
            extraDigit = '0'

        dfFilteredRows = MyDataFrame.getDataFrameByParameters(mr, value, kfoldIndex, nfold)

        # modelName = 'saved_models/SVM' + extraDigit + str(mr) + '-' + str(value) + '-' + str(kfoldIndex) + '-of-' + str(nfold)

        # print('modelName: ', modelName)
        
        # print('len(dfFilteredRow): ', len(dfFilteredRows))

        # print('dfFilteredRow: ', dfFilteredRow)

        # print('Accuracy_Score: ', dfFilteredRows.iloc[0]['Accuracy_Score'])

        return dfFilteredRows.iloc[0]['Accuracy_Score']


    def getDataFrameByParameters(mr, value, kFoldIndex, nfold):

        mrColumnName = MyDataFrame.getColumnNameFromMr(mr)

        # print('mrColumnName: ', mrColumnName)

        # df = pd.read_csv('saved_data/my_dataframe.csv')
        df = pd.read_csv(f'saved_data/{MyParameters.savingFileName}')

        mr_condition = df[mrColumnName] == value

        nfold_condition = df['fold_index'] == kFoldIndex
        kfold_condition = df['n_fold'] == nfold

        combined_condition = mr_condition & kfold_condition & nfold_condition

        # combined_condition = mr_condition & kfold_condition & nfold_condition

        filtered_rows = df[combined_condition]
        
        # filtered_row = df[df['Name'] == modelName]

        if not filtered_rows.empty:
            
            return filtered_rows
        
        else:

            print(f"No data frame found for the parameters")


    
    def getColumnNameFromMr(mr):

        if mr == 0:
            return 'Used_Metamorphic'
        
        elif mr == 1:
            return 'Scalar_Value'    

        elif mr == 2:

            return 'Angle_Rotation'
        
        elif mr == 3:

            return 'Apply_Permutation'

        elif mr == 4:

            return 'Invert_Labels'

        elif mr == 5:

            return 'Perturb_Noise'

        


