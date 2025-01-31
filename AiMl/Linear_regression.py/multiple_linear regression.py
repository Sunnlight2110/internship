import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

"""
IPL Overview: The Indian Premier League (IPL) is a professional T20 cricket league, launched in 2008 by the BCCI.
Franchise System: IPL features franchises that consist of players from around the world. Players are acquired through an annual auction.
Auction Process: The auction follows an English auction system. Teams bid for players, with prices starting at USD 50 million for ownership.
Influence of Other Formats: Although the IPL focuses on T20 cricket, the performance of players in Test and One-Day formats can influence their pricing in the IPL auction.
Player Data: The dataset used in this program contains the performance details of 130 players who participated in at least one IPL season from 2008-2011.
"""

df = pd.read_csv('C:\\Users\\My PC\\Desktop\\Internship\\Machine Learning (Codes and Data Files)\\Data\\IPL IMB381IPL2013.csv')
print(df.columns)   # Returns columns

"""
AGE             Age of the player at the time of auction classified into three categories. Category 1 (125) means the players less than 25 years old, category 2 means that the age is between 25 and 35 years (825-35) and category 3 means that the age is more than 35 (A35).
RUNS-S          Number of runs scored by a player.
RUNS-C          Number of runs conceded by a player.
HS              Highest score by a batsman in IPL.
AVE-B           Average runs scored by a batsman in IPL.
AVE-BL          Bowling average (number of runs conceded/number of wickets taken) in IPL..
SR-B            Batting strike rate (ratio of the number of runs scored to the number of balls faced) in IPL
SR-BL           Bowling strike rate (ratio of the number of balls bowled to the number of wickets taken) in IPL
SIXERS          Number of six runs scored by a player in IPL.
WKTS            Number of wickets taken by a player in IPL.
 ECON           Economy rate of a bowler (number of runs conceded by the bowler per over) in IPL.
CAPTAINCY EXP   Captained either a T20 team or a national team.
ODI-SR-B        Batting strike rate in One-Day Internationals.
ODI-SR-BL       Bowling strike rate in One-Day Internationals.
ODI-RUNS-S      Runs scored in One-Day Internationals.
ODI-WKTS        Wickets taken in One-Day Internationals.
T-RUNS-S        Runs scored in Test matches.
T-WKTS          Wickets taken in Test matches.
PLAYER-SKILL    Player's primary skill (batsman, bowler, or allrounder).
COUNTRY         Country of origin of the player (AUS: Australia; IND: India; PAK: Pakistan; SA: South Africa; SL: Sri Lanka: NZ: Nev Zealand; WI: West Indies; OTH: Other countries).
YEAR-A          Year of Auction in IPL.
IPL TEAM        Team(s) for which the player had played in the IPL (CSK: Chennai Super Kings; DC: Deccan Chargers, DD: Delhi Dare devils; KXI: Kings XI Punjab; KKR: Kolkata Knight Riders; MI: Mumbai Indians; PWI: Pune Warriors India; RR: Rajasthan Royals; RCB: Royal Challengers Bangalore). A + sign is used to indicate that the player has played for more than ne team. For example,
                     CSK+ would mean that the player has played for CSK as well as for one or more other
"""
X_features = df.columns
X_features = ['AGE', 'COUNTRY', 'PLAYING ROLE',
    'T-RUNS', 'T-WKTS', 'ODI-RUNS-S', 'ODI-SR-B',
    'ODI-WKTS', 'ODI-SR-BL', 'CAPTAINCY EXP', 'RUNS-S',
    'HS', 'AVE', 'SR-B', 'SIXERS', 'RUNS-C', 'WKTS',
    'AVE-BL', 'ECON', 'SR-BL'] 

# Encoding Categorical Features
df['PLAYING ROLE'].unique() 
categorical_features = ['AGE','COUNTRY','PLAYING ROLE', 'CAPTAINCY EXP']

encoded_df = pd.get_dummies( df[X_features],
columns = categorical_features,
drop_first = True ) 

X_features = encoded_df.columns

# ==========================================data cleaning
# print(encoded_df.dtypes)
X = sm.add_constant(encoded_df)
Y = df['SOLD PRICE'] 

# Ensure all data is numeric
X = X.apply(pd.to_numeric, errors='coerce')
Y = pd.to_numeric(Y, errors='coerce')

# Convert boolean columns to integers
X = X.astype({col: 'int' for col in X.select_dtypes(include='bool').columns})

# Drop rows with NaN values
X = X.dropna()
Y = Y[X.index]

train_X, test_X, train_y, test_y = train_test_split(X, Y, train_size=0.8, random_state=42)

#  ===============================================================Building model
model = sm.OLS(train_y, train_X).fit()

# ================================================================Calculate vif factor
def get_vif_factors(X):
    X_matrix = X.values #Converts dataframe into matrix
    vif = [variance_inflation_factor(X_matrix, i) for i in range(X.shape[1])]  # stores vif values of each feature
    # Create dataframe to store result
    vif_factors = pd.DataFrame()
    vif_factors['column'] = X.columns
    vif_factors['VIF'] = vif
    return vif_factors

vif_factors = get_vif_factors(X[X_features])

# Check correlation map for large VIF
# columns_with_large_vifs = vif_factors[vif_factors.VIF>4].column
# plt.figure(figsize=(12,10))
# seaborn.heatmap(X[columns_with_large_vifs].corr(),annot=True)
# plt.title('Heatmap depicting')
# plt.show()

# ============================================================Get rid of multi-collinearity
columns_to_be_removed = ['T-Runs','T-WKTS','RUNS-S','HS','AVE','RUNS-C','SR-B','AVE-BL','ECON','ODI-SR-B','ODI-RUNS-S','AGE_2','SR-BL']
X_new_features = list(set(X_features) - set(columns_to_be_removed))

# ===============================================================Build new model after removing collinearity
vif = get_vif_factors(X[X_new_features])
# build new model after removing multi-collinear
X_new = X[X_new_features]
X_new = sm.add_constant(X_new)
model_new = sm.OLS(train_y, X_new.loc[train_X.index]).fit()
print(model_new.summary2())

# Identify significant variables
significant_vars = ['COUNTRY_IND','COUNTRY_ENG','SIXERS','CAPTAINCY EXP_1']

# Build new model with significant variables
train_X = train_X[significant_vars]
model_3 = sm.OLS(train_y, train_X).fit()
print(model_3.summary2())

# ===============================================================Test for normality of residuals (P-P plot)
# Test for normality of residuals using P-P plot
def draw_pp_plot(model, title='P-P Plot of Model Residuals'):
    residuals = model.resid
    sm.ProbPlot(residuals).ppplot(line='45')
    plt.title(title)
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Sample Quantiles')
    plt.show()

# Draw P-P plot for the model
draw_pp_plot(model_3)