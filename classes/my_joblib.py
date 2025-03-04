import joblib

import os


class MyJoblib:

    def saveModelToFile(self, model, fileName):
        
        totalFileName=  f'{fileName}.pkl'

        joblib.dump(model,totalFileName)


    def loadModelFromFile(self, fileName):

        totalFileName=  f'{fileName}.pkl'

        if os.path.exists(totalFileName):
            # print("File exists.")

            file_size = os.path.getsize(totalFileName)
            if file_size == 0:
                doNothing=True

                # print(f"The file '{totalFileName}' is empty.")
            else:
                doNothing=True

                # print(f"The file '{totalFileName}' is not empty. Size: {file_size} bytes.")

        else:
            print("File does not exist.")

        print('fileName: ', totalFileName)
        loaded_model = joblib.load(totalFileName)

        return loaded_model
