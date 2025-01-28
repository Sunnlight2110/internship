import pandas as pd

# ! Viewing data
# ? head returns headers and spesified numbers of rows
df = pd.read_csv('data.csv')
print(df.head(10))

# ? Tail returns headers and last numbers of rows
print(df.tail(5))

# ! Info about dataset
# ? Info gives information about data set
print(df.info())

