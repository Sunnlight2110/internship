import pandas as pd

# ! Read JSON
# ? Load Json files into data frame

df = pd.read_json('data.json')
print(df)