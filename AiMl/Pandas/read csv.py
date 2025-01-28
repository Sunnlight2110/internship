import pandas as pd
# ! Read CSV files
# ? Load csv data into data frame

df = pd.read_csv('data.csv')
# print(type(df))
# print(df.to_string())  #To string is used to load entire data into dataframe
# print(df.columns)
# print(list(df.Pulse))
# print(df.head(5).T)   #Create transpose of dataset

"""If rows in csv is more than max rows, print(df) will return first and last 5 rows"""
print(pd.options.display.max_rows)

# ? Increate max numbers of rows
pd.options.display.max_rows = 99

