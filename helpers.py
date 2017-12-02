def tryParseFloat(val):
    ret = 0.0
    try:
        ret = float(val)
        return ret
    except Exception as err:
        return ret

