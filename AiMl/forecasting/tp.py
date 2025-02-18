import pandas as pd
string = '2007 0101 01'
string = string[0:4]+'/'+string[5:7]+'/'+string[7:9]+'/'+string[10:]
print(string)

string = pd.to_datetime(string)
print(string)