import spicy
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

"""call center has 10 calls/hour"""

# number of calls= max 5
print(spicy.stats.poisson.cdf(5,10))    #(no. of events, avg. no. of events)

# no of calls over 3 hour period = 30
print(spicy.stats.poisson.cdf(30,30))

df = pd.DataFrame({
    'success': range(0,30),
    'probability': list(spicy.stats.poisson.pmf(range(0,30),10))
})

sn.barplot(x = df['success'], y = df['probability'])
plt.show()