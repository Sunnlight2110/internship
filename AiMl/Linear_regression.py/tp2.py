import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generate data
np.random.seed(42)
data = {
    'Square_Feet': np.random.randint(500, 3000, 100),
    'Num_Rooms': np.random.randint(2, 7, 100),
    'House_Age': np.random.randint(0, 30, 100),
    'Price': np.random.randint(50000, 500000, 100)
}

df = pd.DataFrame(data)
