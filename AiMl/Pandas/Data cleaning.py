import pandas as pd

#/ Cleaning means Fixing bed data into data set
#/ Bed data could be: Empty cells, wrong formate, wrong data, duplicates

# ! Remove null values rows
# "Date" in row 22, and "Calories" in row 18 and 28

df = pd.read_csv('data.csv')
print(df.to_string())
print("*"*100)
new_df = df.dropna()  #/ Returns data frame with null value rows removed
print(new_df.to_string())

# ? Default dropna returns new dataframe and not change original, to change use inplace = True argument

# ! Replace null values
print(df.fillna(-1).to_string())  #/ Returns dataframe with null value replaced
print(df['Calories'].fillna(-1)) #/ Returns dataframe with null value replaced in just a specified column

# ! Replace using Mean Median and mode
# / Mean = average, median = middle value, mode = most repeated value

calories_mean = df['Calories'].mean()
print(calories_mean)
# print(df['Calories'].fillna(calories_mean).to_string())
# print(df['Calories'].fillna(df['Calories'].median()).to_string())
# print(df['Calories'].fillna(df['Calories'].mode()).to_string())



# ! Cleaning data with wrong formate
#  row 22 and 26, the 'Date'

# ! Convert all records into correct formate
# df = pd.read_csv('data.csv')
# df['Date'] = pd.to_datetime(df['Date']) 
# print('*'*100)
# print(df.to_string())



# ! Wrong data
# / Does not have to be empty cell or wrong formate
# row 7, the duration is 450, but for all the other rows the duration is between 30 and 60.
# / Replacing values

df.loc[7,'Duration'] = 45
# With loop
for i in df.index:
    if df.loc[i,'Duration'] > 120:
        df.loc[i,'Duration'] = 120

# ! Removing rows:
df = pd.read_csv('data.csv')

for i in df.index:
    if df.loc[i,'Duration'] > 120:
        df.drop(i,inplace=True)


# ! Duplicates
df = pd.read_csv('data.csv')
print(df.duplicated().to_string())  #returns True for duplicated rows otherwise false

new_df = df.drop_duplicates()    #Removes duplicates
print(new_df.to_string())

