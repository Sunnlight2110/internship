import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import random
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

sales_df = pd.read_csv('C:\\Users\\My PC\\Desktop\\Internship\\Machine Learning (Codes and Data Files)\\Data\\Advertising.csv')
# print(sales_df.head(5))

Y = sales_df.Sales
X = sales_df[['TV','Radio','Newspaper']]

# standardize
Y = np.array((Y-Y.mean())/Y.std())
X = X.apply(lambda X : (X-X.mean())/X.std(),axis = 0)

# ? random initialization for bias and weights
def initialize(dim):
    """
    Initialize the weights and bias for a linear regression model.
    Parameters:
    dim (int): Number of elements to be initialized besides the bias.
    Returns:
    
    """
    np.random.seed(seed=42)
    random.seed(42)
    b = random.random()
    w = np.random.rand(dim)
    return b,w      #b = intersection w = slope

b,w = initialize(3)
# print(f'bias:{b},weights:{w}')

# ? predict Y for bias and values

def predict_Y(b,w,X):
    """
    b : bias
    w : weights
    X : the input matrix
    """
    return b + np.matmul(X,w)     #matmul = multiplication for higher dimensions

b,w = initialize(3)
Y_hat = predict_Y(b,w,X)    #Predicted values
# print(Y_hat[0:10])

# ? Cost function (MSE)
def get_cost(Y, Y_hat):
    """
    Calculate the cost function for linear regression.
    Parameters:
    Y (array): Actual values.
    Y_hat (array): Predicted values.
    Returns:
    float: The cost value.
    """
    Y_resid = Y - Y_hat
    return np.sum(np.matmul(Y_resid.T, Y_resid))/len(Y_resid)
    # return np.mean(Y_resid**2)

# print('MSE',get_cost(Y,Y_hat))

# ? Update bias and weights

def update_beta(x,y,y_hat,b_0,w_0,learning_rate):
    # gradient of bias
    db = (np.sum(y_hat-y)*2)/len(y)
    # gradients of weights
    dw = (np.dot((y_hat - y),x)*2)/len(y)
    # update bias
    b_1 = b_0 - learning_rate * db
    # update beta
    w_1 = w_0 -learning_rate * dw

    return b_1, w_1

# print(f'before:- bias:{b},weights:{w}')
b,w = update_beta(X,Y,Y_hat,b,w,0.01)
# print(f'after:- bias:{b},weights:{w}')

# ? Find the optimal bias and weights
def run_gradient_descent(X,Y,alpha=0.01,num_iterations = 100):
    b,w = initialize(X.shape[1])
    iter_num = 0

    # gd_iteration_df keeps track of cost at every 10 iterations
    gd_iterations_df = pd.DataFrame(columns=['iterations','cost'])
    result_idx = 0

    for each_iter in range(num_iterations):
        Y_hat = predict_Y(b,w,X)

        this_cost = get_cost(Y,Y_hat)

        prev_b = b  #save previous bias
        prev_w = w  #save previous weight

        b,w = update_beta(X,Y,Y_hat,prev_b,prev_w,alpha)
        
        # for every 10 iterations store cost
        if (iter_num % 10 ==0):
            gd_iterations_df.loc[result_idx] = [iter_num,this_cost]
            result_idx = result_idx + 1
        iter_num+=1

    # print('final estimation of b and w:',b,w)

    return gd_iterations_df,b,w

gd_iterations_df,b,w = run_gradient_descent(X,Y,alpha=0.001,num_iterations=200)
# print(gd_iterations_df[0:10])

# plt.plot(
#     gd_iterations_df['iterations'],
#     gd_iterations_df['cost']
# )
# plt.xlabel('Number of iterations')
# plt.ylabel('Cost/MSE')
# plt.show()

# print('final estimation of b and w:',b,w)

alpha_df_1, b, w = run_gradient_descent(
    X,Y,
    alpha=0.01,
    num_iterations=2000
)
# print('final estimation of b and w:',b,w)
alpha_df_2, b, w = run_gradient_descent(
    X,Y,
    alpha=0.001,
    num_iterations=2000
)
print('final estimation of b and w:',b,w)
# plt.plot(alpha_df_1['iterations'],alpha_df_1['cost'],label = 'alpha = 0.01')
# plt.plot(alpha_df_2['iterations'],alpha_df_2['cost'],label = 'alpha = 0.001')
# plt.legend()
# plt.xlabel('Number of iterations')
# plt.ylabel('Cost/MSE')
# plt.show()

"""Look for flat line that forms about 90 degree"""

train_x,test_x,train_y,test_y = train_test_split(
    sales_df[['TV','Radio','Newspaper']],
    sales_df.Sales,
    test_size=0.3,
    random_state=42
)

# print(len(train_x))
# print(len(test_x))

linreg = LinearRegression()
linreg.fit(train_x,train_y)

# print(linreg.intercept_)
# print(linreg.coef_)
# print(list(zip(['TV','Radio','Newspaper'],list(linreg.coef_))))

# ? Making predictions
pred_y = linreg.predict(test_x)

test_pred_df = pd.DataFrame({
    'actual' : test_y,
    'predicted' : np.round(pred_y,2),
    'residuals' : test_y-pred_y
})
# print(test_pred_df.sample(10))

r2 = metrics.r2_score(train_y,linreg.predict(train_x))
# print('r squared:',r2)
mse = metrics.mean_squared_error(test_y,pred_y)
rsme = round(np.sqrt(mse),2)
print('RSME:',rsme)