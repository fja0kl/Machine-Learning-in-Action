#coding:utf8
from numpy import *

def loadDataSet(filename):
    dataMat = []; labelMat = []
    fr = open(filename)
    n = len(fr.readline().strip().split('\t'))
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        data = []
        for i in range(n-1):
            data.append(float(lineArr[i]))
        dataMat.append(data)
        labelMat.append(float(lineArr[-1]))
    return dataMat, labelMat

def buildStump(dataArr,classLables, D):
    """
    构建弱分类器：决策树类型；树桩
    :param dataArr:
    :param classLables:
    :param D:
    :return: 决策树；误差；预测结果
    """
    dataMat = mat(dataArr); classMat = mat(classLables).transpose()
    n = shape(dataMat)[1]
    bestIndex = 0; bestStump={}; minError = inf;