import numpy as np

# 1. Identifying missing data(represented as np.nan)

np1 = np.array([1,2,np.nan,4,5,np.nan,np.nan,8,np.nan],dtype=np.float16)

missing = np.isnan(np1)
print("Missing data ", missing)

# 2. Handling missing data

# remove missing data
np2 = np1[~np.isnan(np1)]
print("No missing data array ",np2)

# replace missing data with specefic value(mean or median)
mean = np.nanmean(np1) #compute mean
np2 = np.where(np.isnan(np1),mean,np1)
print("replacing missing with mean ",np2)

"""
3. Operations ignoring missing data
nan-aware function to ignore nan
"""

total = np.nansum(np1)
print("total avoiding missing data, ",total)

"""
4. Working with integer
Integer does not support nan, instad use placeholder like -1 or 999
"""
np1 = np.array([1,2,-1,4,5,-1,-1,8,-1],dtype=np.int16)
missing = np1 == -1
mean = np1[~missing].mean()
print("mean in integer avoiding missing data ", mean)

"""
5. Advanced technique
use np.ma to mark and handle missing data
"""

np1 = np.ma.array([1,2,np.nan,4,5,np.nan,np.nan,8,np.nan],mask = np.isnan([1,2,np.nan,4,5,np.nan,np.nan,8,np.nan]))
print("masked array ", np1)
print("array without missing data ", np1.compressed())

"""
Interpolation for Missing Data
You can interpolate missing values in a linear or polynomial manner. NumPy doesn’t have built-in interpolation, but it can be done using SciPy
"""

from scipy.interpolate import interp1d
np1 = np.array([1,2,np.nan,4,5,np.nan,np.nan,8,np.nan],dtype=np.float16)
x = np.arange(len(np1))
mask = ~np.isnan(np1)

interpolader = interp1d(x[mask],np1[mask],bounds_error= False, fill_value="-1")
np1_interpolade = interpolader(x)
print("interpolader data ",np1_interpolade)


