# encoding : utf-8

import xlrd
import xlwt
import math
import numpy as np
import pandas as pd
from sklearn import preprocessing

data=pd.read_excel("./qx_input.xlsx",usecols=[2,3,4,5,6,7,8,9,10,11,12,13],header=None,names=['arr1','arr2','arr3','arr4','arr5','arr6','arr7','arr8','arr9','arr10','arr11','arr12'])

out_file="./out_pca.csv"
# 数据预处理
df_scaled = preprocessing.scale(data)


from sklearn.decomposition import PCA
pca = PCA()
pca.fit(df_scaled)
#print(pca.explained_variance_ratio_)
sum=0
for i in range(len(pca.explained_variance_ratio_)):
    sum+=pca.explained_variance_ratio_[i]
    
    print(i)
    print(sum)

print(pca.explained_variance_)
pca = PCA(n_components=2)
pca.fit(df_scaled)
low_d=pca.transform(df_scaled)
pd.DataFrame(low_d).to_csv(out_file,index=False)

