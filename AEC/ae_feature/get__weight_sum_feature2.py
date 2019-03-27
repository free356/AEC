import pandas as pd
import numpy as np
import math
from numpy import array

def cal_weight(data):
    df=(data-data.min())/(data.max()-data.min())
    #print(df)
    #12
    k = 1.0 / math.log(rows)
    lnf = [[None] * cols for i in range(rows)]
    lnf = array(lnf)
    for i in range(0, rows):
        for j in range(0, cols):
            if df.iloc[i,j] == 0:
                lnfij = 0.0
                #print("you o")
            else:
                p = df.iloc[i,j] / df.sum()[j]
                lnfij = math.log(p) * p 
            lnf[i][j] = lnfij
    lnf = pd.DataFrame(lnf)
    E=lnf.sum()*(-k)
    #E是每个指标的信息熵
    w=(1-E)/(cols-(np.sum(E)))#W是每一个指标的权重
    return w
def cal_score(data,w):
    f = [[None] for i in range(rows)]#shape是（12，9）
    for i in range(0,rows):
        sum=0
        for j in range(0, cols):
            sum+=w[j]*data.iloc[i,j]
        f[i]=sum
    return f
feature1=[]
for i in range(0,381):
    data = pd.read_csv('./site_{}_feature1.csv'.format(i))
    cols=data.columns.size
    rows=data.iloc[:,0].size
    w=cal_weight(data)
    score=cal_score(data,w)
    feature1.append(score)
print(feature1)
print(np.array(feature1).shape)

