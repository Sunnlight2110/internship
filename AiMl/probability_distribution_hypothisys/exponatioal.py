import spicy
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

# probability system will fail before 100 hours
print(spicy.stats.expon.cdf(x=1000,loc = 1/1000,scale = 1000))  #x = hours Loc only graphical, scale = rate of decay

# system will not fail up to 2000 hours
print(1-spicy.stats.expon.cdf(2000,1/1000,1000))

# 10% of system will fail
print(spicy.stats.expon.ppf(.1,1/1000,1000))

# sn.barplot(x = df['success'], y = df['probability'])
# plt.show()