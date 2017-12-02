import helpers
import numpy as np
import pandas as pd
import numbers

def dataCleaner(stock):
    stock.P_E_Ratio = stock.P_E_Ratio.apply(lambda x: helpers.tryParseFloat(x))
    stock.P_E_Ratio = stock.P_E_Ratio.astype(float)

    stock.Beta = stock.Beta.apply(lambda x: helpers.tryParseFloat(x))
    stock.Beta = stock.Beta.astype(float)

    stock.Perf_Year = stock.Perf_Year.apply(lambda x: helpers.tryParseFloat(x))
    stock.Perf_Year = stock.Perf_Year.astype(float)

    stock.market_Cap = stock.market_Cap.apply(lambda x: np.where('-' in str(x), 0, x))

    finalVal = []
    for cap in stock.market_Cap:
        if ('B' in str(cap)):
            finalVal.append(float(str(cap).replace('B', '')) * 100000000)
        elif ('M' in str(cap)):
            finalVal.append(float(str(cap).replace('M', '')) * 1000000)
        else:
            finalVal.append(cap)
            print('not recognized' + cap)

    print(finalVal)

    stock.market_Cap = finalVal
    stock.market_Cap = stock.market_Cap.apply(lambda x: np.where(isinstance(x, float), x, '0'))

    # stock.market_Cap = stock.market_Cap.apply(lambda x:np.where('B' in str(x),  print(float(str(x).replace('B', ''))*100000000), print(x + ':Other')))
    # stock.market_Cap = stock.market_Cap.apply(lambda x:np.where('B' in str(x), float(str(x).replace('B', ''))*100000000, str(x)))
    # stock.market_Cap = stock.market_Cap.apply(lambda x:np.where('M' in str(x), float(str(x).replace('M', ''))*1000000, str(x)))

    # stock.market_Cap = stock.market_Cap.apply(lambda x: np.where(x.isdigit(), x, '0'))
    stock.market_Cap = stock.market_Cap.astype(float)


    stock.Volume = stock.Volume.astype(float)

    stock.Dividend = stock.Dividend.apply(lambda x: helpers.tryParseFloat(x))
    stock.Dividend = stock.Dividend.astype(float)

    stock.ROA = stock.ROA.apply(lambda x: helpers.tryParseFloat(x.replace('%', '')))
    stock.ROA = stock.ROA.astype(float)

    stock.ROE = stock.ROE.apply(lambda x: helpers.tryParseFloat(x.replace('%', '')))
    stock.ROE = stock.ROE.astype(float)

    stock.ROI = stock.ROI.apply(lambda x: helpers.tryParseFloat(x.replace('%', '')))
    stock.ROI = stock.ROI.astype(float)

    stock.Debt_Eq = stock.Debt_Eq.apply(lambda x: helpers.tryParseFloat(x))
    stock.Debt_Eq = stock.Debt_Eq.astype(float)

    stock.Outstanding = stock.Outstanding.apply(lambda x: helpers.tryParseFloat(x))
    stock.Outstanding = stock.Outstanding.astype(float)

    stock.Short_Ratio = stock.Short_Ratio.apply(lambda x: helpers.tryParseFloat(x))
    stock.Short_Ratio = stock.Short_Ratio.astype(float)