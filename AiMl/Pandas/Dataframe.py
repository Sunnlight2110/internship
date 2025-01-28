import pandas as pd

#! Data frames
"""Data frame are multi-dimensional tables
Series is a column, where dataframe is table"""

data = {
    "Cars":["audi",'jaguar','omni'],
    'price' : [10,20,1000000]
}
df = pd.DataFrame(data)
print(df)

# ? Locate rows
print(df.loc[1])
print(df.loc[[0,2]])  #Accepts list

# ? Index
df = pd.DataFrame(data,index=['luxary1','laxary2','omnipotant'])
print(df)
print(df.loc['omnipotant'])  #Get data by index

