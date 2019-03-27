import pandas as pd
import numpy as np
import math
from numpy import array
feature1 = pd.read_csv('./feature1.csv')
feature2 = pd.read_csv('./feature2.csv')
feature=pd.concat([feature1,feature2],axis=1)
#print(feature)
feature.to_csv("./feature.csv",index=False)
print(feature.columns.size)
print(feature.iloc[:,0].size)

