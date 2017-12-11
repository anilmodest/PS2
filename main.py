import featureRanker
import pandas as pd

def main():
    #stock = pd.read_csv("testData.csv", nrows=20)
    #fRanker = featureRanker.FeatureRanker(['market_Cap', 'P_E_Ratio', 'Price', 'ROE', 'Dividend', 'Beta', 'Perf_YTD'],
                                          #stock, 'Price')
    stock = pd.read_csv("AAPL_quarterly_financial_data.csv")
    fRanker = featureRanker.FeatureRanker(['Dividend per share', 'Price', 'P/E ratio', 'ROE', 'Revenue', 'Shareholders equity', 'Liabilities', 'Assets'],
                                          stock, 'P/E ratio')
    ranks = fRanker.calculatefeatureranks()

if __name__ == "__main__":
    print("starting evaluations")
    main()
    print("finished evaluation")


