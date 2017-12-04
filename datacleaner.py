import helpers
import numpy as np
import pandas as pd
import numbers


def datacleanhelper(col_names, data):
    for col_name in col_names:
        if(col_name is 'market_Cap'):
            values = []
            for val in data[col_name]:
                if ('B' in str(val)):
                    values.append(float(str(val).replace('B', '')) * 100000000)
                elif ('M' in str(val)):
                    values.append(float(str(val).replace('M', '')) * 1000000)
                else:
                    values.append(val)
            data[col_name] = values

        else:
            data[col_name] = data[col_name].apply(lambda x: helpers.tryParseFloat(x))
            data[col_name] = data[col_name].astype(float)
    return data


def dataCleaner(stock):
    stock.P_E_Ratio = stock.P_E_Ratio.apply(lambda x: helpers.tryParseFloat(x))
    stock.P_E_Ratio = stock.P_E_Ratio.astype(float)

    stock.Beta = stock.Beta.apply(lambda x: helpers.tryParseFloat(x))
    stock.Beta = stock.Beta.astype(float)

    #stock.Perf_Year = stock.Perf_Year.apply(lambda x: helpers.tryParseFloat(x))
    #stock.Perf_Year = stock.Perf_Year.astype(float)

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


    stock.market_Cap = stock.market_Cap.astype(float)


    #stock.Volume = stock.Volume.astype(float)

    stock.Dividend = stock.Dividend.apply(lambda x: helpers.tryParseFloat(x))
    stock.Dividend = stock.Dividend.astype(float)



    stock.ROE = stock.ROE.apply(lambda x: helpers.tryParseFloat(x.replace('%', '')))
    stock.ROE = stock.ROE.astype(float)


    #stock.Debt_Eq = stock.Debt_Eq.apply(lambda x: helpers.tryParseFloat(x))
    #stock.Debt_Eq = stock.Debt_Eq.astype(float)


    #stock.Short_Ratio = stock.Short_Ratio.apply(lambda x: helpers.tryParseFloat(x))
    #stock.Short_Ratio = stock.Short_Ratio.astype(float)