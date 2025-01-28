import scipy
"""
Student appling for mastes in 8 universities
constant and independent probability of getting select in each = 0.42

A: probability she will get call from at least 3
B: probability she will get call from exactly 4
"""

# A:
prob_a = 1 - scipy.stats.binom.cdf(2,8,0.42)

print(prob_a)

# B:
prob_b = scipy.stats.binom.pmf(4,8,0.42)

print(prob_b)