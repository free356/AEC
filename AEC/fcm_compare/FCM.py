# -*- coding:utf-8 -*-
from pylab import *
from numpy import *
import pandas as pd
import numpy as np
import operator
import math
import matplotlib.pyplot as plt
import random
from sklearn import datasets
from mpl_toolkits.basemap import Basemap
#数据保存在.csv文件中
#df_full = pd.read_csv("./Test2.csv")
df_full = pd.read_csv("./ae_feature.csv")
df=df_full
#columns = list(df_full.columns)
#features = columns[4:len(columns)]
#class_labels = list(df_full[columns[-1]])
#df = df_full[features]

yy = []
xx = [tt for tt in range(100)]

dd = df_full.values
# print(dd)
dd = np.array(dd)

# 维度
num_attr = len(df.columns) - 1
# 分类数
kk = 5
# 最大迭代数
MAX_ITER = 99
# 样本数
n = len(df)    #the number of row
# 模糊参数
m = 2.00

centerr = [[] for k in range(kk)]
avgg = [k for k in range(kk)]
class_s = [[] for k in range(kk)]

def get_error():
    for t in range(kk):
        temp = []
        cnt = 0
        for y in range(4, len(dd[0])):
            temp.append(0)
        for y in range(len(labels)):
            if labels[y] == t:
                for u in range(4, len(dd[0])):
                    temp[u - 4] += float(dd[y][u])
                cnt += 1
        for y in range(4, len(dd[0])):
            temp[y - 4] /= float(cnt)
        for y in range(len(temp)):
            centerr[t].append(temp[y])

    for t in range(len(labels)):
        class_s[labels[t]].append(t)

    for t in range(kk):
        sum = 0
        for y in range(len(class_s[t])):
            for u in range(y + 1, len(class_s[t])):
                temp = 0
                for l in range(4, len(dd[0])):
                    temp += (float(dd[y][l]) - float(dd[u][l])) ** 2
                temp = math.sqrt(temp)
                sum += temp
        sum = sum * 2 / float(len(class_s[t]) * (len(class_s[t]) - 1))
        avgg[t] = sum

    sum = 0
    for t in range(kk):
        maxx = -20000
        for y in range(kk):
            if t == y:
                continue
            dcen = 0
            for u in range(4, len(dd[0])):
                dcen += (float(dd[t][u]) - float(dd[y][u])) ** 2
            dcen = math.sqrt(dcen)
            if (avgg[t] + avgg[y]) / dcen > maxx:
                maxx = (avgg[t] + avgg[y]) / dcen
        sum += maxx
    sum /= float(kk)
    return sum

#初始化模糊矩阵
def initializeMembershipMatrix():
    membership_mat = list()
    for i in range(n):
        random_num_list = [random.random() for i in range(kk)]
        summation = sum(random_num_list)
        temp_list = [x/summation for x in random_num_list]#首先归一化
        membership_mat.append(temp_list)
    return membership_mat

#计算类中心点
def calculateClusterCenter(membership_mat):
    cluster_mem_val = zip(*membership_mat)
    cluster_centers = list()
    cluster_mem_val_list = list(cluster_mem_val)
    for j in range(kk):
        x=cluster_mem_val_list[j]
        xraised = [e ** m for e in x]
        denominator = sum(xraised)
        temp_num = list()
        for i in range(n):
            data_point = list(df.iloc[i])
            prod = [xraised[i] * val for val in data_point]
            temp_num.append(prod)
        numerator = map(sum, zip(*temp_num))
        center = [z/denominator for z in numerator]#每一维都要计算。
        cluster_centers.append(center)
    return cluster_centers

#更新隶属度
def updateMembershipValue(membership_mat, cluster_centers):
#    p = float(2/(m-1))
    data=[]
    for i in range(n):
        x = list(df.iloc[i])#取出文件中的每一行数据
        data.append(x)
        distances = [np.linalg.norm(list(map(operator.sub, x, cluster_centers[j]))) for j in range(kk)]
        for j in range(kk):
            den = sum([math.pow(float(distances[j]/distances[c]), 2) for c in range(kk)])
            membership_mat[i][j] = float(1/den)       
    return membership_mat,data

#得到聚类结果
def getClusters(membership_mat):
    cluster_labels = list()
    for i in range(n):
        max_val, idx = max((val, idx) for (idx, val) in enumerate(membership_mat[i]))
        cluster_labels.append(idx)
    return cluster_labels

def fuzzyCMeansClustering():
    # 主程序
    membership_mat = initializeMembershipMatrix()
    curr = 0
    while curr <= MAX_ITER:#最大迭代次数
        cluster_centers = calculateClusterCenter(membership_mat)
        membership_mat,data = updateMembershipValue(membership_mat, cluster_centers)
        tmp = 0
        for ii in range(5):
            for jj in range(381):
                dis = 0
                for kk in range(len(cluster_centers[0])):
                    dis += (float(dd[jj][kk]) - cluster_centers[ii][kk]) ** 2
                    #dis += (float(dd[jj][kk+4]) - cluster_centers[ii][kk]) ** 2
                tmp += membership_mat[jj][ii] ** 2 * dis
        yy.append(tmp)
        cluster_labels = getClusters(membership_mat)
        curr += 1
    #print(membership_mat)
    return cluster_labels, cluster_centers,data,membership_mat

def xie_beni(membership_mat,center,data):
    sum_cluster_distance=0
    min_cluster_center_distance=inf
    for i in range(kk):
        for j in range(n):
            sum_cluster_distance=sum_cluster_distance + membership_mat[j][i]** 2 * sum(power(data[j,:]- center[i,:],2))#计算类一致性
    for i in range(kk-1):
        for j in range(i+1,kk):
            cluster_center_distance=sum(power(center[i,:]-center[j,:],2))#计算类间距离
            if cluster_center_distance<min_cluster_center_distance:
                min_cluster_center_distance=cluster_center_distance
    return sum_cluster_distance/(n*min_cluster_center_distance)
labels,centers,data,membership= fuzzyCMeansClustering()
#print(labels)
#print(centers)
center_array=array(centers)
label=array(labels)
datas=array(data)
#print(yy)
#plt.plot(xx, yy)
#plt.show()


print("DBI: ",get_error())
#Xie-Beni聚类有效性
print("Xie-Beni聚类有效性：",xie_beni(membership,center_array,datas))
"""
# from this, the code is to draw the map
plt.figure(figsize=(12, 6))
m = Basemap(llcrnrlon=77, llcrnrlat=10, urcrnrlon=140, urcrnrlat=51, projection='lcc', lat_1=33, lat_2=45, lon_0=100)
m.drawcoastlines()
m.drawcountries(linewidth=1)

m.etopo()


colors = ['r', 'g', 'y', 'b', 'c', 'm', 'k', 'pink']
parallels = np.arange(0.,81,10.)
# labels = [left,right,top,bottom]
m.drawparallels(parallels,labels=[False,True,True,False])
meridians = np.arange(10.,351.,20.)
m.drawmeridians(meridians,labels=[True,False,False,True])
for i in range(len(labels)):
    lon, lat = float(dd[i][3]), float(dd[i][2])
    xpt,ypt = m(lon,lat)
    # print(xpt, ypt)
    lonpt, latpt = m(xpt,ypt,inverse=True)
    # m.plot(xpt, ypt, 'bo', markersize=3)
    # m.plot(xpt, ypt, color = labels[i], markersize = 3)
    m.scatter(xpt, ypt, 7, color = colors[labels[i]], marker = 'o')

# plt.scatter(center_array[:,0],center_array[:,1],marker = 'x', color = 'm', s = 50)

plt.show()
"""


