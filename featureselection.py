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

stock = pd.read_csv("testData.csv")
stock = stock.drop(['sr_No', 'Ticker', 'Company', 'sector', 'industry','country','P_S_Ratio','P_B_Ratio','P_C_Ratio'
         ,'P_C_Ratio', 'Insider_Trans', 'Insider_Own', 'Inst_Own','Inst_Trans','Float_Short','Perf_Quart','Perf_Half', 'Recom', 'Float'],axis=1)

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
plt.show()

print('Test')