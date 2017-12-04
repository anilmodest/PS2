import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import seaborn as sns
#%matplotlib inline

from sklearn.feature_selection import RFE, f_regression
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, RandomizedLasso)
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import datacleaner

print('hi')

stock = pd.read_csv("testData.csv", nrows=10)
'''
stock = stock.drop(['sr_No', 'Ticker', 'Company', 'sector', 'industry','country','P_S_Ratio','P_B_Ratio','P_C_Ratio'
         ,'P_C_Ratio', 'Insider_Trans', 'Insider_Own', 'Inst_Own','Inst_Trans','Float_Short','Perf_Quart','Perf_Half', 'Recom', 'Float'
                       , 'ATR', 'SMA20', 'SMA50', 'SMA200', '52W_High', '52W_Low', 'RSI', 'Price.1', 'Volatility_Month'
                    ],axis=1)
                    
'''

stock = stock[['market_Cap','P_E_Ratio','Price','ROE','Dividend','Beta','Perf_YTD']]
print(stock)
stock.head()
datacleaner.dataCleaner(stock)


# Looking for nulls
print(stock.isnull().any())
print(stock.dtypes)

#gt = sns.load_dataset(stock)
gt = sns.pairplot(stock, hue='Beta')
#gt.pairplot(gt, hue)

'''
#'Price','ROE','Perf_YTD', 'Beta'
with sns.plotting_context("notebook",font_scale=2.5):
    g = sns.pairplot(stock[['market_Cap', 'Beta']],
                 hue='Beta', palette='afmhot',size=6)
g.set(xticklabels=[]);
'''



str_list = [] # empty list to contain columns with strings (words)
for colname, colvalue in stock.iteritems():
    if type(colvalue[1]) == str:
         str_list.append(colname)
# Get to the numeric columns by inversion
num_list = stock.columns.difference(str_list)
# Create Dataframe containing only numerical features
house_num = stock[num_list]
f, ax = plt.subplots(figsize=(16, 12))
plt.title('Pearson Correlation of features')
# Draw the heatmap using seaborn
#sns.heatmap(house_num.astype(float).corr(),linewidths=0.25,vmax=1.0, square=True, cmap="PuBuGn", linecolor='k', annot=True)
sns.heatmap(house_num.astype(float).corr(),linewidths=0.25,vmax=1.0, square=True, cmap="cubehelix", linecolor='k', annot=True)



# First extract the target variable which is our Beta
Y = stock.Beta.values
# Drop Beta from the stock dataframe and create a matrix out of the stock data
stock = stock.drop(['Beta'], axis=1)
X = stock.as_matrix()
# Store the column/feature names into a list "colnames"
colnames = stock.columns


# Define dictionary to store our rankings
ranks = {}
# Create our function which stores the feature rankings to the ranks dictionary
def ranking(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x,2), ranks)
    return dict(zip(names, ranks))

# Finally let's run our Selection Stability method with Randomized Lasso
rlasso = RandomizedLasso(alpha=0.04)
rlasso.fit(X, Y)
ranks["rlasso/Stability"] = ranking(np.abs(rlasso.scores_), colnames)
print('finished')

# Construct our Linear Regression model
lr = LinearRegression(normalize=True)
lr.fit(X,Y)
#stop the search when only the last feature is left
rfe = RFE(lr, n_features_to_select=1, verbose =3 )
rfe.fit(X,Y)
ranks["RFE"] = ranking(list(map(float, rfe.ranking_)), colnames, order=-1)

# Using Linear Regression
lr = LinearRegression(normalize=True)
lr.fit(X,Y)
ranks["LinReg"] = ranking(np.abs(lr.coef_), colnames)

# Using Ridge
ridge = Ridge(alpha = 7)
ridge.fit(X,Y)
ranks['Ridge'] = ranking(np.abs(ridge.coef_), colnames)

# Using Lasso
lasso = Lasso(alpha=.05)
lasso.fit(X, Y)
ranks["Lasso"] = ranking(np.abs(lasso.coef_), colnames)

rf = RandomForestRegressor(n_jobs=-1, n_estimators=50, verbose=3)
rf.fit(X,Y)
ranks["RF"] = ranking(rf.feature_importances_, colnames);

# Create empty dictionary to store the mean value calculated from all the scores
r = {}
for name in colnames:
    r[name] = round(np.mean([ranks[method][name]
                             for method in ranks.keys()]), 2)

methods = sorted(ranks.keys())
ranks["Mean"] = r
methods.append("Mean")

print("\t%s" % "\t".join(methods))
for name in colnames:
    print("%s\t%s" % (name, "\t".join(map(str,
                                          [ranks[method][name] for method in methods]))))


# Put the mean scores into a Pandas dataframe
meanplot = pd.DataFrame(list(r.items()), columns= ['Feature','Mean Ranking'])

# Sort the dataframe
meanplot = meanplot.sort_values('Mean Ranking', ascending=False)

# Let's plot the ranking of the features
sns.factorplot(x="Mean Ranking", y="Feature", data = meanplot, kind="bar",
               size=14, aspect=1.9, palette='coolwarm')

plt.show()

print('Test')

