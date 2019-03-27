import pandas as pd
import numpy as np
import math
from numpy import array
feature = pd.read_csv("site_0_tmy_concanate_feature.csv")
for i in range(1,381):
    data = pd.read_csv("site_{}_tmy_concanate_feature.csv".format(i))
    feature=pd.concat([feature,data])
print(np.array(feature).shape)
feature.to_csv("./tmy_feature.csv",index=False)

