import numpy as np

# sampling without replacement
np1 = np.arange(1,11)

print("WO replacement",np.random.choice(np1,10,replace=False))
print("With replacement",np.random.choice(np1,10,replace=True))

# sampling in range of numbers
print("randint",np.random.randint(1,10,size=12))    #(start,end,size)
print('rand',np.random.rand(5)) #Array of floats between 0 and 1 size = 5(in bracket)
print('uniform',np.random.uniform(1,10,size = 5))   #(start,end,size)   random floats in range
print("shuffle",np.random.shuffle(np1)) #shuffle array randomly
# Set the random seed for reproducibility
np.random.seed(42)
# Generate random numbers after setting the seed
random_with_seed = np.random.rand(5)
print("Random Numbers with Seed:", random_with_seed)

print("Permutation",np.random.permutation(50))  #random permutation of numbers from 0 to x-1, size = x

"""Advanced sampling techniques"""
np1 =np.arange(1,11)
np2 = np.random.randint(0,3) #Random starting point
print("random sample ",np1[np2::2])

np2 = np.random.choice(np1,2)
np3 = np.random.choice(np1,2)
print("Combine samples",np.concatenate([np2,np3]))

np1 = np.random.normal(loc=2,scale=5,size=10)  #loc = mean of distribution, scale = standard version of distribution
print("random.normal",np1)