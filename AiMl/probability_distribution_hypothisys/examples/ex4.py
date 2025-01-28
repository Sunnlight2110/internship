import scipy

"""
pestisized to 1000 famers, 10 acers of farmland STD = 5 lits
pestisize spray/weak is normal distribution

A: proportion farmer spraying more than 50 lits/weak
B: proportion farmer spraying less than 10 lits/weak
C: proportion farmer spraying between 30lits and 60 lits/weak
"""

prob = scipy.stats.norm(50,5) # mean for 1000 farmers, 5 std

# A:
prob_a = 1 - prob.cdf(50)

# B:
prob_b = prob.cdf(10)

# C:
prob_c = prob.cdf(60) - prob.cdf(30)

print(prob_a,prob_b,prob_c,sep="\n")