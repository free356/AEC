import numpy as np 
import pandas as pd 
from sklearn import preprocessing
from sklearn.decomposition import PCA
site=pd.read_csv("./wangxiao.csv",usecols=["T3","T8","T10","T13"])



def pca_weight(data):
    data_scale= preprocessing.scale(data)
    #print(type(data_scale))
    pca=PCA(n_components=4)
    m=pca.fit(data_scale)
    w=pca.explained_variance_ratio_
    return w
def get_weight_sum(data):
    w=pca_weight(data)
    weight_sum=[]
    for i in range(216):
        l=0
        for j in range(4):
            l+=w[j]*data.iloc[i,j]
        weight_sum.append(l)
    return weight_sum

def get_average_weight_sum(data):
    w=pca_weight(data)
    weight_sum=[]
    for s in range(12):
        average=data.iloc[18*s:18*s+18,:].mean()
        average_weight_sum=0
        for s2 in range(4):
            average_weight_sum+=average[s2]*w[s2]
        weight_sum.append(average_weight_sum)
    return weight_sum

def get_min_indexs(data):
    weight_sum=np.array(get_weight_sum(data))
    average_weight_sum=np.array(get_average_weight_sum(data))
    min_indexs=[]
    for j2 in range(12):
        sub=weight_sum[18*j2:18*j2+18]-average_weight_sum[j2]
        sub=sub.tolist()
        abs_sub=np.abs(sub).tolist()
        min_index=abs_sub.index(min(abs_sub))
        min_indexs.append(min_index)
    return min_indexs
def get_data(data):
    da=[]
    for j3 in range(12):
        index=get_min_indexs(data)
        s=data.iloc[18*j3:18*j3+18,:].iloc[index[j3]]
        da.append(s)
    return da 
n=381
for i in range(n):
    data=site.iloc[216*i:216*i+216,:]
    tmy=get_data(data)
    tmy=pd.DataFrame(tmy)
    tmy.to_csv("./typeyear/site_{}_tmy_feature.csv".format(i),index=False)




     

