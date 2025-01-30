import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

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

model = sm.OLS(train_y, train_X).fit()


