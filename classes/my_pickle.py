import pickle

class MyPickle:

    def saveModelToFile(model, fileName):
        with open('{fileName}.pkl', 'wb') as file:
            pickle.dump(model, file)        


    def loadModelFromFile(fileName):
        
        with open('{fileName}.pkl', 'rb') as file:

            loaded_model = pickle.load(file)

            return loaded_model
