import numpy as np 
from sklearn.datasets import load_wine 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler 
from sklearn.model_selection import KFold

from classes.parameters import MyParameters

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
    
    def prepareNFoldData(self):

        seed = 1234

        np.random.seed(seed)

        x,y = load_wine(return_X_y = True)

        scaler = MaxAbsScaler()
        x = scaler.fit_transform(x)
        x = np.clip(x, 0, 1)  # Clip values to [0, 1]


        n_folds = MyParameters.n_folds

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        
        train_data_list = []
        test_data_list = []        
        for train_index, test_index in kf.split(x):
            
            # Split the data into training and testing sets for this fold
            x_tr, x_test = x[train_index], x[test_index]
            y_tr, y_test = y[train_index], y[test_index]

            train_data_list.append((x_tr, y_tr))
            test_data_list.append((x_test, y_test))

        return np, train_data_list, test_data_list


                

