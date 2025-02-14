from sklearn.decomposition import PCA
from wine_data import WineData

class DataManager:
      
      def getDatabyNumber(self, type = 0):
          
        options = {
          0 : WineData,
          # 4 : sqr,
          # 9 : sqr,
          # 2 : even,
          # 3 : prime,
          # 5 : prime,
          # 7 : prime,
        }

        if not type < len(options):
           
           print('type is not available and will call type 0')
           type = 0

        myData = options[type]()

        np, x_tr, x_test, y_tr, y_test = myData.prepareData()

        
        return np, x_tr, x_test, y_tr, y_test
                    