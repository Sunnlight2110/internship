import spicy
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

"""E commerce company
20 customers purchased
"""
# probability of exactly 5 customers returns items
print(spicy.stats.binom.pmf(5,20,0.1))

# probability of maximum than 5 customers returns
print(spicy.stats.binom.cdf(5,20,0.1))

# Probability of more than 5 customers returns
print(1-spicy.stats.binom.cdf(5,20,0.1))

# probability of average customers returns

df = pd.DataFrame(
    {'success': range(0,21),
     'probability':spicy.stats.binom.pmf(range(0,21),20,0.1)}
)

sn.barplot(
    x = df['success'], y = df['probability'],
    color = 'blue'
)

plt.show()