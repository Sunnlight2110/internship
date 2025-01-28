import numpy as np
# ? Uniform
# Each has same probability
print('uniform',np.random.uniform(0,10,5))  #(lowest value, highest value, size)

# ?Normal
# data clusterd around mean
print('normal',np.random.normal(0,1,5)) #(loc = mean, scale = std, size)

# ?Binomial
# number of success in fixed number of trials
print('binomial',np.random.binomial(10,0.5,5))  #(n = no. of trials, p = probability of success, size)

# ?poisson
# No. of events in fixed no. of interval, time, space
print('poisson',np.random.poisson(3,5)) #(lam = rate of occurance, size)
# ?Exponential
# model time between events in poisson
print('exponential', np.random.exponential(1,5))    #(scale = mean time between events,size)

# ?Logistic
# model growth and logistic regression (e.g., population growth)
print('logistic', np.random.logistic(0, 1, 5))    #(loc = mean, scale = std deviation, size)

# ?Multinomial
# model multiple categorical outcomes (e.g., dice rolls)
print('multinomial', np.random.multinomial(10, [0.2, 0.3, 0.5], 5))    #(n = trials, pvals = probability of each outcome, size)

# ?Chi-Square
# used for statistical hypothesis testing (e.g., goodness-of-fit tests)
print('chi-square', np.random.chisquare(2, 5))    #(df = degrees of freedom, size)

# ?Rayleigh
# model skewed data (e.g., signal processing, radar)
print('rayleigh', np.random.rayleigh(1, 5))    #(scale = scale parameter, size)

# ?Pareto
# model power-law distributions (e.g., wealth distribution)
print('pareto', np.random.pareto(2, 5))    #(a = shape parameter, size)

# ?Zipf
# model ranked data (e.g., word frequency)
print('zipf', np.random.zipf(2, 5))    #(a = shape parameter, size)