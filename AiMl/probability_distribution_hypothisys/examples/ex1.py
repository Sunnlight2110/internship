import numpy as np
import scipy 

"""
Customer returns follows poisson distribution
    25 returns/day

a: probability of return exceeds 30 in a day
b: chance of return = 0.05, probability of at least 2 returns in a day
"""

#  A:
proba = 1- scipy.stats.poisson.cdf(30,25)
print(proba)

# B:
prob = scipy.stats.binom.cdf(2,25,0.02)
print(prob)