def tryParseFloat(val):
    ret = 0.0
    try:
        print('value before: ' + str(val)+'$')
        if '%' in val:
            val = val.strip().replace('%', '')
        if '-' in val:
            val = val.strip().replace('-', '')
        print('value after : ' + str(val)+'$')
        ret = float(val.strip())
        print('converted to float:' + str(ret) +'$')
        return ret
    except Exception as err:
        print('unable to convert: ' + str(ret)+'$')
        print(err)
        return ret


def datacleaner(col_names, data):
    for col_name in col_names:
        if col_name is 'market_Cap':
            values = []
            for val in data[col_name]:
                if 'B' in str(val):
                    values.append(float(str(val).replace('B', '')) * 100000000)
                elif 'M' in str(val):
                    values.append(float(str(val).replace('M', '')) * 1000000)
                else:
                    values.append(val)
            data[col_name] = values

        else:
            print(col_name)
            #data[col_name].apply(lambda x: print(str(x) + ': {}'.format(str(tryParseFloat(x)))))
            data[col_name] = data[col_name].apply(lambda x: tryParseFloat(str(x)))
            data[col_name] = data[col_name].astype(float)
    return data

