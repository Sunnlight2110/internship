import numpy as np

# print("Basic mathematical functions","-"*80)
np1 = np.array([1,2,3])
# np2 = np.array([4,5,6])

# print("add",np.add(np1,np2))
# print("subtract",np.subtract(np1,np2))
# print("multiply",np.multiply(np1,np2))
# print("divide",np.divide(np1,np2))
# print("power",np.power(np1,np2))
# print('square root',np.sqrt(np1))

# print("floor and ceiling","-"*80)
# np1 = np.array([1.2,2.6,3.5])
# print("floor(round down)",np.floor(np1))
# print("ceiling(round down)",np.ceil(np1))

# print("Trigonometric functions","-"*80)
# np1 = np.array([0,np.pi/2,np.pi])
# print("sin",np.sin(np1))
# print("cos",np.cos(np1))
# print("tan",np.tan(np1))
# print("arcsin",np.arcsin(np1))
# print("arccos",np.arccos(np1))
# print("arctan",np.arctan(np1))

# np2 = np.degrees(np1) #Converts radient into degree
# print("degree",np2)
# print("radians",np.radians(np2))


# print("Expositional and Logarithmic functions","-"*80)
# np1 = np.array([1,np.e,np.e**2])
# print("exp ",np.exp(np1)) #compares e^x for each elements
# print("log",np.log(np1)) #Natural logarithm
# print("log10",np.log10(np1)) #logarithm for base 10

# print("Aggregation functions functions","-"*80)
# np1 = np.array([1,2,3,4,5])
# print("sum",np.sum(np1)) #sum of all elements
# print("prod",np.prod(np1)) #multiplication of all elements
# print("cumsum",np.cumsum(np1)) 
# print("cumprod",np.cumprod(np1))

# """Statistic functions"""
# print("Basic descriptive statistic functions","-"*80)
# np1 = np.array([1,2,3,4,5])
# print("mean",np.mean(np1)) #Gives average of all elements
# print("median",np.median(np1)) #Gives middle value of array
# print("std",np.std(np1)) #Gives standerd deviation
# print("var",np.var(np1)) #Gives computer variation(diffrence or change in data)

# print("Min and Max","-"*80)
# np1 = np.array([1,7,8,7,6,54,457,8,8,3])
# print("min",np.min(np1))
# print("argmin",np.argmin(np1)) #Gives Index
# print("max",np.max(np1))
# print("argmax",np.argmax(np1)) #Gives Index

# print("Percentiles and Quantiles","-"*80)
# np1 = np.array([1,2,3,4,5])
# print("percentile",np.percentile(np1,75)) #Gives percentile(array,percentage)
# print("quantile",np.quantile(np1,0.50)) #Divides array into 4 equal sized intervals and acts as percentile


# print("Correlation and Covariance","-"*80)
# np1 = np.array([1,2,3,4,5])
# np2 = np.array([5,4,3,2,1])
# print("corrcoef",np.corrcoef(np1,np2)) #Measures the degree of joint variability of two variables.
# print("cov",np.cov(np1,np2)) #Measures the strength and direction of the linear relationship.

# print("Random sampling","-"*80)
# np1 = np.array([1,2,3,5,4,6,7,8])
# print("random.choice",np.random.choice(np1,size=5,replace=False)) #replace means if same item can be chosen or not again
# print("rand",np.random.rand(5)) #Generates between 0s and 1s
# print("randn",np.random.randn(6)) 
# print("rendit",np.random.randint(1,10,5)) #Generates between integers specified(Min, Max, size of array)

""" NumPy GCD (Greatest Common Divisor)
When to Use: To find the greatest divisor that two numbers share """

a = np.array([12, 24, 36])
b = np.array([18, 36, 54])
print('gcd', np.gcd(a, b)) 

"""NumPy LCM (Least Common Multiple)
When to Use: To find the smallest number that both numbers can divide into"""

a = np.array([12, 15, 20])
b = np.array([18, 25, 30])
print('lcm', np.lcm(a, b))  

""". NumPy Hyperbolic Functions (for shape of hyperbola)
What Are They?: Hyperbolic functions are mathematical functions that are analogs of the trigonometric functions, but based on hyperbolas instead of circles. These are particularly useful in fields like physics and engineering, and help us solve many types of equations.

When to Use: Use these for operations involving hyperbolic angles or functions. Examples include calculations in relativistic mechanics, signal processing, and more."""

X = np.array([0,1,2])
print('sinh', np.sinh(X))
print('cosh', np.cosh(X))
print('tanh', np.tanh(X))

a = np.array([1, 2, 3, 4, 5])
b = np.array([4, 5, 6, 7, 8])

# Unique elements in an array
unique_elements = np.unique(a)

# Intersection: common elements between a and b
intersection = np.intersect1d(a, b)

# Set difference: elements in a but not in b
set_diff = np.setdiff1d(a, b)

# Union: all unique elements from both arrays
union = np.union1d(a, b)

print('Unique elements in a:', unique_elements)
print('Intersection of a and b:', intersection)
print('Difference between a and b:', set_diff)
print('Union of a and b:', union)
