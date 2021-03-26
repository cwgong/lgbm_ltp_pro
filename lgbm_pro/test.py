import pandas as pd

df = pd.read_csv('./kkk.txt',header=None,sep=' ')
print(df[:])

df1 = df.drop(1,axis=1,inplace=False)
print(df1)

s = '0:0.8976644277572632'
def pre_deal_data(x):
    if type(x) != int:
        feature_list = x.split(":")
    else:
        t = type(x)
        print(t)
        feature_list = [x]
    if len(feature_list) > 1:
        return float(feature_list[1])
    else:
        return float(x)

bool_array = df1.applymap(pre_deal_data)
print(bool_array)
print(bool_array.info())