import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import seaborn as sns
import helpers


from sklearn.feature_selection import RFE, f_regression
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, RandomizedLasso)
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor


class FeatureRanker:
    def __init__(self, feature_lst, data_frame, plot_by_feature):
        self.feature_lst = feature_lst
        self.data_frame = data_frame
        self.plot_by_feature = plot_by_feature

    def cleanandfilterdata(self):
        stock = self.data_frame[self.feature_lst]
        stock.head()
        return helpers.datacleaner(self.feature_lst, stock)

    def generateheatmap(self, filtered_data):
        str_list = []  # column names
        for colname, colvalue in filtered_data.iteritems():
            if type(colvalue[1]) == str:
                str_list.append(colname)
        # select numric columns
        num_list = filtered_data.columns.difference(str_list)
        # select only numerical features
        stock_num = filtered_data[num_list]
        f, ax = plt.subplots(figsize=(16, 12))
        plt.title('Pearson Correlation of features')
        # Draw the heatmap
        sns.heatmap(stock_num.astype(float).corr(), linewidths=0.25, vmax=1.0, square=True, cmap="cubehelix",
                    linecolor='k', annot=True)

    def extracttargetvariables(self, filtered_data):
        Y = filtered_data[self.plot_by_feature].values
        # Drop plot by variable from selection and create a matrix out of the stock data
        filtered_data = filtered_data.drop([self.plot_by_feature], axis=1)
        X = filtered_data.as_matrix()
        # extract the column/feature names into a list "colnames"
        colnames = filtered_data.columns
        return Y, X, colnames


        # function to stores the feature rankings to the ranks dictionary
    def ranking(self, ranks, names, order=1):
        minmax = MinMaxScaler()
        ranks = minmax.fit_transform(order * np.array([ranks]).T).T[0]
        ranks = map(lambda x: round(x, 2), ranks)
        return dict(zip(names, ranks))

    #function determines ranking using different techniques e.g. randomised lasso, lasso, RF
    def determinmultiplerankings(self, ranks, X, Y, colnames):
        # randomised lasso ranking
        rlasso = RandomizedLasso(alpha=0.04)
        rlasso.fit(X, Y)
        ranks["rlasso/Stability"] = self.ranking(np.abs(rlasso.scores_), colnames)
        print('lasso ranking determined')

        # Construct Linear Regression model
        lr = LinearRegression(normalize=True)
        lr.fit(X, Y)
        # stop the search when only the last feature is left
        rfe = RFE(lr, n_features_to_select=1, verbose=3)
        rfe.fit(X, Y)
        ranks["RFE"] = self.ranking(list(map(float, rfe.ranking_)), colnames, order=-1)

        # Using Linear Regression
        lr = LinearRegression(normalize=True)
        lr.fit(X, Y)
        ranks["LinReg"] = self.ranking(np.abs(lr.coef_), colnames)

        # Using Ridge
        ridge = Ridge(alpha=7)
        ridge.fit(X, Y)
        ranks['Ridge'] = self.ranking(np.abs(ridge.coef_), colnames)

        # Using Lasso
        lasso = Lasso(alpha=.05)
        lasso.fit(X, Y)
        ranks["Lasso"] = self.ranking(np.abs(lasso.coef_), colnames)

        rf = RandomForestRegressor(n_jobs=-1, n_estimators=50, verbose=3)
        rf.fit(X, Y)
        ranks["RF"] = self.ranking(rf.feature_importances_, colnames)
        return ranks

    def calculatefeatureranks(self):
        filtered_data = self.cleanandfilterdata()
        print(filtered_data.isnull().any())
        print(filtered_data.dtypes)
        print('plot features w.r.t. selected feature')
        gt = sns.pairplot(filtered_data, hue=self.plot_by_feature, palette='afmhot')
        '''
        with sns.plotting_context("notebook", font_scale=2.5):
            g = sns.pairplot(filtered_data, hue=self.plot_by_feature, palette='afmhot', size=6)
        g.set(xticklabels=[]);
        '''

        self.generateheatmap(filtered_data)
        (Y, X, colnames) = self.extracttargetvariables(filtered_data)
        ranks = {}
        ranks = self.determinmultiplerankings(ranks, X, Y, colnames)

        # Create empty dictionary to store the mean value calculated from all the scores
        m = {}
        for name in colnames:
            m[name] = round(np.mean([ranks[method][name]
                                     for method in ranks.keys()]), 2)

        methods = sorted(ranks.keys())
        ranks["Mean"] = m
        methods.append("Mean")

        print("\t%s" % "\t".join(methods))
        for name in colnames:
            print("%s\t%s" % (name, "\t".join(map(str,
                                                  [ranks[method][name] for method in methods]))))

        # Put the mean scores into dataframe
        meanplot = pd.DataFrame(list(m.items()), columns=['Feature', 'Mean Ranking'])

        # Sort dataframe
        meanplot = meanplot.sort_values('Mean Ranking', ascending=False)

        # plot the ranking of the features
        sns.factorplot(x="Mean Ranking", y="Feature", data=meanplot, kind="bar",
                       size=14, aspect=1.9, palette='coolwarm')

        plt.show()
        return meanplot





