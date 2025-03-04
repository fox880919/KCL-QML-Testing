from sklearn.decomposition import PCA
from wine_data import WineData

class DataManager:
      
      options = {
          0 : WineData,
          # 4 : sqr,
          # 9 : sqr,
          # 2 : even,
          # 3 : prime,
          # 5 : prime,
          # 7 : prime,
      }
      
      def getDatabyNumber(self, type = 0):
          
        if not type < len(DataManager.options):
           
          #  print('type is not available and will call type 0')
           type = 0

        myData = DataManager.options[type]()

        np, x_tr, x_test, y_tr, y_test = myData.prepareData()

        
        return np, x_tr, x_test, y_tr, y_test
      
      def getListOfFoldDatabyNumber(self, type = 0):

        if not type < len(DataManager.options):
           
          #  print('type is not available and will call type 0')
           type = 0

        myData = DataManager.options[type]()

        np, train_data_list, test_data_list = myData.prepareNFoldData()

        
        return np, train_data_list, test_data_list