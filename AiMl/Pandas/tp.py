import seaborn as sn
import matplotlib.pyplot as plt

age = [12,13,154,213,123,1000]

# sn.barplot(age)
# plt.hist(age,bins=1)
sn.pairplot(age)

# plt.xlim(-10,220)
plt.show()