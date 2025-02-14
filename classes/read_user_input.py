
class ReadUserInput:

    def checkIfUserWantsToUseDefaultParameters(self):

        userParameters =  input("Should we use default Parameters? (0 for Yes, and 1 for No)") or 0

        if not userParameters.isdigit():
            userParameters = 0
        else:
            userParameters = int(userParameters)

        if userParameters == 0:
            return True
        else:
            return False
        
    def readGenericBoleanInput(self, message):

        userParameters =  input(message) or False

        if not userParameters.isdigit():
            userParameters = 0
        else:
            userParameters = int(userParameters)

        if userParameters == 0:
            return True
        else:
            return False
        
    def startReadingUserInput(self, step = 0):
        
        options = {
          0 : ReadUserInput.readDataTypeInput,
        }

        dataType = options[step](self) 

        return dataType
        # print("dataType is: ", dataType)

    def readDataTypeInput(self):

        dataType =  input("Enter data type (default: 0):") or 0

        if not dataType.isdigit():
            dataType = 0
        else:
            dataType = int(dataType)

        return dataType
    
    def readFeatureMapTypeInput(self):

        featureMapType =  input("Enter feature map type (default: 0):") or 0

        if not featureMapType.isdigit():
            featureMapType = 0
        else:
            featureMapType = int(featureMapType)

        return featureMapType
    
    def readGeneralNumericInput(self, message, defaultNumber):

        generalNumericInput =  input(message) or 0

        if not generalNumericInput.isdigit():
            generalNumericInput = defaultNumber
        else:
            generalNumericInput = int(generalNumericInput)

        return generalNumericInput
        