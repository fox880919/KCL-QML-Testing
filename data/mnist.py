import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler 
from sklearn.model_selection import KFold

from sklearn.datasets import fetch_openml

from classes.parameters import MyParameters

class MNIST:

    def prepareData(self):

        seed = 1234

        np.random.seed(seed)

        fashion_mnist = fetch_openml("Fashion-MNIST", version=1, as_frame=False)
        x = fashion_mnist.data.astype('float32') / 255.0
        y = fashion_mnist.target.astype('int')

        if MyParameters.usePercentageOfData == True:
        
        # Get only 1% of the data (700 samples)
            x_small, _, y_small, _ = train_test_split(x, y, train_size= MyParameters.PercentageOfData, stratify=y, random_state=42)

            # Split that small dataset into train/test
            x_tr, x_test, y_tr, y_test = train_test_split(x_small, y_small, train_size=0.9, stratify=y_small, random_state=42)

        else:

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

        fashion_mnist = fetch_openml("Fashion-MNIST", version=1, as_frame=False)
        x = fashion_mnist.data.astype('float32') / 255.0
        y = fashion_mnist.target.astype('int')

        if MyParameters.usePercentageOfData == True:

        # ⬇️ Take only 1% of the data (700 samples)
            x, _, y, _ = train_test_split(x, y, train_size= MyParameters.PercentageOfData, stratify=y, random_state=seed)


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


                

