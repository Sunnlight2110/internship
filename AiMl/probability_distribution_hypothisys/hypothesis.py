import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
"""Scenario: A company is conducting a survey to improve employee satisfaction. They have data on employee satisfaction scores from two departments (Department A and Department B) and they want to analyze this data using hypothesis tests.
Data:
Department A: 50 employees, satisfaction scores are assumed to follow a normal distribution with a mean of 6 and standard deviation of 1.
Department B: 50 employees, satisfaction scores are assumed to follow a normal distribution with a mean of 6.5 and standard deviation of 1.2.
"""

groupA = np.random.normal(6,1,50)
groupB = np.random.normal(6,1.2,50)

# ? Z test
"""Testing if A's satisfaction scores differs from mean?"""
z_test, p_value = stats.ttest_1samp(groupA,6)
print(f"ztest: z statistics:{z_test},p_value:{p_value}")


# ? one sample t_test
"""Testing if B's satisfaction scores differs from mean?"""
t_test, p_value = stats.ttest_1samp(groupB,6.5)
print(f"one sample: t statistics:{t_test},p_value:{p_value}")

# ? two sample t_test
t_test, p_value = stats.ttest_ind(groupA,groupB)
print(f"two sample: t statistics:{t_test},p_value:{p_value}")

# ? paired t_test
"""Assuming the same employees before and after a training program, we simulate correlated data"""
after_training_A = groupA + np.random.normal(0.5,0.5,50)    #+ for improvements
t_test, p_value = stats.ttest_rel(groupA,after_training_A)
print(f"paired test: t statistics:{t_test},p_value:{p_value}")

# ? Chi-squared test
"""Testing if there's a relationship between employee gender and satisfaction using synthetic categorical data"""
observed = np.array([   #[male high,male low],[female high,female low]
    [30,20],
    [15,25]
])

# ? Chi-squared
chi2_stat, p_value, dof, expected = stats.chi2_contingency(observed)    #default = stats.chisquar
print(f"Chi-Squared Test: Chi-squared statistic: {chi2_stat}, p-value: {p_value}")

# ? Anova(Analysis of variance)
"""
sales or revenue across groups that received different levels of discounts
 (e.g., 10%, 20%, 30%) to analyze the effect of discounts on sales.
"""

discount_10 = [200, 220, 210, 190, 205]
discount_20 = [250, 260, 240, 230, 245]
discount_30 = [280, 300, 290, 270, 295]

# find one way Anova
statistic, value = stats.f_oneway(discount_10,discount_20,discount_30)

labels = ['discount_10','discount_20','discount_30']
data = [discount_10,discount_20,discount_30]
plt.boxplot(data , label=labels,patch_artist=True, boxprops=dict(facecolor='lightblue'))

plt.ylabel('sales')
plt.show()

