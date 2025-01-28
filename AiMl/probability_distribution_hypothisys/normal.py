import pandas as pd
import numpy as np
import warnings

beml = pd.read_csv('BEML_2010_2016.csv')
glaxo = pd.read_csv('GLAXO_2010_2016.csv')

print(beml[:5])
print(glaxo[:5])