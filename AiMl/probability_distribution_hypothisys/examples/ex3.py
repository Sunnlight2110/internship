import scipy

"""
TIme of failure in machine follows exponential distribution
between failures estimated 85 hours

A: prob that system will fail before 85 hours
B: prob system will not fail up to 85 hours
"""

# A: 
prob_a =1 -  scipy.stats.expon.cdf(x = 85, scale = 85)
print(prob_a)

# B:
prob_b = 1-prob_a
print(prob_b)