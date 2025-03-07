def OperatorPower(O, list):
    out = 0
    for l in list:
        out = O(out, l)
    return out