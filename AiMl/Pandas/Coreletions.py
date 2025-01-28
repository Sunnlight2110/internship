import pandas as pd

"""
coreletion refres from 0 to 1
reletion mean if a colmun changes data, how much data other column changes?
1 means perfect coreletion
0 mean no coreletion

atleast 0.6 is good coreletion
"""
df = pd.read_csv('data.csv')

print(df.corr())    #Shows relation between columns
# / corr ingnores non numeric columns

