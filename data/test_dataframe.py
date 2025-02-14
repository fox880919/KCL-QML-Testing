from my_dataframe import MyDataFrame

# import pandas as pd

data = {
  "calories": [420, 380, 390],
  "duration": [50, 40, 45]
}

myDataFrame = MyDataFrame()

myDataFrame.processToDataFrame(data)

#load data into a DataFrame object:
# df = pd.DataFrame(data)

# print(df) 

#refer to the row index:
# print(df.loc[0])

#use a list of indexes:
# print(df.loc[[0, 1]])

# data = {
#   "calories": [420, 380, 390],
#   "duration": [50, 40, 45]
# }

# df = pd.DataFrame(data, index = ["day1", "day2", "day3"])

# print(df) 


#refer to the named index:
# print(df.loc["day2"])


# import os

# current_directory = os.getcwd()  # Get the current working directory
# print('current_directory: ', current_directory)

# parent_directory = os.path.dirname(current_directory)  # Get the parent directory
# print('parent_directory: ', parent_directory)

# target_folder = os.path.join(parent_directory, 'saved_data')  # Construct the target folder path
# print('target_folder: ', target_folder)

# file_path = os.path.join(target_folder, 'my_dataframe.csv')

# df.index = df.index + 1

# df.to_csv(file_path, index=True)

# # df.to_csv('csv/my_saved_data.csv', encoding='utf-8', index=False)
