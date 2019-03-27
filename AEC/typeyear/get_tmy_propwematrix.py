import numpy as np 
import pandas as pd 
for i in range(0,381):
    data = pd.read_csv('./site_{}_tmy_feature.csv'.format(i))
    m1=data.iloc[0,0]
    m2=data.iloc[0,1]
    m3=data.iloc[0,2]
    m4=data.iloc[0,3]
    feature=pd.DataFrame({"m1":m1,"m2":m2,"m3":m3,"m4":m4},index=[1])
    for j in range(1,12):
        m1=data.iloc[j,0]
        m2=data.iloc[j,1]
        m3=data.iloc[j,2]
        m4=data.iloc[j,3]
        feature1=pd.DataFrame({"m1":m1,"m2":m2,"m3":m3,"m4":m4},index=[1])
        feature=pd.concat([feature,feature1],axis=1)
    feature.to_csv("./tmy_site/site_{}_tmy_concanate_feature.csv".format(i),index=False)
        



