import numpy as np 
import pandas as pd 
site=pd.read_csv("./ae_12_7_2_7_12_out.csv")
#data=data.iloc[217*2:217*2+216,:]
#print(data)
n=381
for i in range(n):
    data=site.iloc[217*i:217*i+216,:]
    if i==2:
        print(data.head())
    m2000=data.iloc[0:0+12,1].values
    m2001=data.iloc[12:12+12,1].values
    m2002=data.iloc[12*2:12*2+12,1].values
    m2003=data.iloc[12*3:12*3+12,1].values
    m2004=data.iloc[12*4:12*4+12,1].values
    m2005=data.iloc[12*5:12*5+12,1].values
    m2006=data.iloc[12*6:12*6+12,1].values
    m2007=data.iloc[12*7:12*7+12,1].values
    m2008=data.iloc[12*8:12*8+12,1].values
    m2009=data.iloc[12*9:12*9+12,1].values
    m2010=data.iloc[12*10:12*10+12,1].values
    m2011=data.iloc[12*11:12*11+12,1].values
    m2012=data.iloc[12*12:12*12+12,1].values
    m2013=data.iloc[12*13:12*13+12,1].values
    m2014=data.iloc[12*14:12*14+12,1].values
    m2015=data.iloc[12*15:12*15+12,1].values
    m2016=data.iloc[12*16:12*16+12,1].values
    m2017=data.iloc[12*17:12*17+12,1].values
    feature1=pd.DataFrame({"2000":m2000,"2001":m2001,"2002":m2002,"2003":m2003,"2004":m2004,"2005":m2005,"2006":m2006,"2007":m2007,"2008":m2008,"2009":m2009,"2010":m2010,"2011":m2011,"2012":m2012,"2013":m2013,"2014":m2014,"2015":m2015,"2016":m2016,"2017":m2017})
    feature1.to_csv("./site2/site_{}_feature2.csv".format(i),index=False)
    

