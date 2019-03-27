import numpy as np 
import pandas as pd 
site=pd.read_csv("./ae_12_7_2_7_12_out.csv")
n=381
feature=[]
for i in range(n):
    data=site.iloc[217*i:217*i+216,:]
    if i==2:
        print(data.head())
    m1=data.iloc[217*i:217*i+216,0]].values
    m2=data.iloc[217*i:217*i+216,1]].values   
    feature1=pd.DataFrame({"m1":m1,"m2":m2})
    feature.append(deature1)
print(feature)


