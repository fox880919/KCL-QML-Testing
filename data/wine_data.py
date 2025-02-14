import numpy as np 
from sklearn.datasets import load_wine 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler 

class WineData:

    def prepareData(self):

        seed = 1234

        np.random.seed(seed)

        x,y = load_wine(return_X_y = True)

        x_tr, x_test, y_tr, y_test = train_test_split(x, y, train_size = 0.9)

        scaler = MaxAbsScaler()
        x_tr = scaler.fit_transform(x_tr)

        x_test = scaler.transform(x_test)
        x_test = np.clip(x_test,0,1)

        
        # print('in prepare data')

        # print('np is: ', np)

        return np, x_tr, x_test, y_tr, y_test
