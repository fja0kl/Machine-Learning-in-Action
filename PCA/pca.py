#coding:utf8
from numpy import *

def loadDataSet(filename, delim='\t'):
    fr = open(filename)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [map(float, line) for line in stringArr]
    return mat(datArr)

def pca(dataMat, topNfeat=9999999):
    meanVals = mean(dataMat, axis=0) # 计算平均值
    meanRemoved = dataMat - meanVals # 1. 去除平均值
    covMat = cov(meanRemoved, rowvar=0) # 2. 计算协方差矩阵
    eigVals, eigVects = linalg.eig(mat(covMat)) # 计算协方差矩阵的特征值和特征向量
    eigValInd = argsort(eigVals) # 3. 将特征值从小到大排序
    eigValInd = eigValInd[:-(topNfeat+1):-1] # 保留最上面的N个特征值；
    redEigVects = eigVects[:, eigValInd]
    lowDDataMat = meanRemoved * redEigVects # 将数据转换到上述N个特征向量构建的新空间中；
    reconMat = (lowDDataMat * redEigVects.T) + meanVals # 重构后的数据用于调试
    return lowDDataMat, reconMat

if __name__ == '__main__':
    dataMat = random.random(size=1000000)*100
    dataMat = dataMat.reshape(1000, 1000)
    lowDDataMat, reconMat = pca(dataMat, 1)
    print(lowDDataMat.shape)
    print(sum(power(dataMat-reconMat, 2))/shape(dataMat)[0])

