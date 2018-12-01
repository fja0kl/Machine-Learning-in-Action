#!/usr/bin/python3
#coding:utf-8
import numpy as np

# 数据预处理: 0均值. 1方差----数据保持相同刻度;
# 计算数据集的协方差矩阵,计算协方差矩阵的特征值,特征向量; 
# 取topN个特征值对应的特征向量作为降维后的向量坐标基; 
# 将数据映射到低维坐标系上.
def pca(dataMat, topN=999999):
    # 0 均值; 一行代表一条数据
    meanVals = np.mean(dataMat, axis=0)
    dataMat -= meanVals
    # 1 方差
    varVals = np.std(dataMat, axis=1)
    dataMat /= varVals
    # 计算协方差矩阵
    covMat = np.cov(dataMat)
    # 计算协方差矩阵的特征值, 特征向量: 返回的特征向量矩阵中特征向量按照列排列
    eigVals, eigVectors = np.linalg.eig(np.mat(covMat))
    # 对特征值排序
    eigValsIdx = np.argsort(eigVals)# 升序排序
    # 对特征值筛选出topN个特征指,从而确定降维后的坐标基
    eigValsIdx = eigValsIdx[:-(topN + 1):-1]#从后往前选
    redEigVectors = eigVectors[:, eigValsIdx]
    lowDimsMat = dataMat * redEigVectors
    reconMat = varVals * (lowDimsMat * redEigVectors.T) + meanVals
    return lowDimsMat, reconMat

if __name__ == '__main__':
    dataMat = np.random.random(size=1000000)*100
    dataMat = dataMat.reshape(1000, 1000)
    lowDDataMat, reconMat = pca(dataMat, 1)
    print(lowDDataMat.shape)
    print(np.sum(np.power(dataMat-reconMat, 2))/np.shape(dataMat)[0])