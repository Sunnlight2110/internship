import pandas as pd

# ! >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Series <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

"""Series
Pandas series are like columns(one dimensional array)"""

data = [1,2,3,4]

print(pd.Series(data))
print(pd.Series(data)[0])  #Returns first value of series

"""Labels
Labels can be used to access specified value
If nothing is specified, values are labeled with index numbers"""
#? =========================Create labels ===============================================================
labeled = pd.Series(data,index=['a','b','c','d'])
print(labeled)
print(labeled['b'])  #Return value of b label

#? =========================Key/value object as series ================================================== 
cars = {
    "audi":1000,
    "bmw":2000,
    "lambo":3000,
    "omni":100000000000000000000000000000000
}
print(pd.Series(cars))  #The keys becomes labels
print(pd.Series(cars,index=['omni','lmabo']))  #Partial
