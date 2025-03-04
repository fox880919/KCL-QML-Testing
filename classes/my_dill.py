import dill

import os

class MyDill:


    def saveModelToFile(self, model, fileName):

        fullFileName = f'{fileName}.pkl'

        with open(fullFileName, 'wb') as file:
            dill.dump(model, file)        


    def loadModelFromFile(self, fileName):
        
        fullFileName = f'{fileName}.pkl'

        # print('fullFileName: ', fullFileName)

        if os.path.exists(fullFileName):
            # print("File exists.")

            file_size = os.path.getsize(fullFileName)
            if file_size == 0:
                doNothing=True
                # print(f"The file '{fullFileName}' is empty.")
            else:
                doNothing=True
                # print(f"The file '{fullFileName}' is not empty. Size: {file_size} bytes.")

        else:
            print("File does not exist.")

        with open(fullFileName, 'rb') as file:

            loaded_model = dill.load(file)

            # if loaded_model is None:
            #     print(f"The file '{fullFileName}' contains None.")
            # else:
            #     print(f"Loaded object: {loaded_model}")
            #     print(f"Type of loaded object: {type(loaded_model)}")

            # print(f"Loaded object: {loaded_model}")
            return loaded_model
