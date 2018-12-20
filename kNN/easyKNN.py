#coding:utf8
from numpy import *
import operator

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    lables = ['A','A','B','B']
    return group, lables

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX,(dataSetSize,1)) - dataSet#tile：将inx扩展，扩展后的形状为（dataSetSize，1）；然后进行运算；；；；目的是通过矩阵来，加快运算速度
    sqDiffMat = diffMat ** 2
    sqDiffMat = sqDiffMat.sum(axis=1)#按行求和
    distances = sqDiffMat ** 0.5
    sortedDistIndicies = distances.argsort()#numpy argsort函数返回的是数组值从小到大的索引值
    print sortedDistIndicies
    print distances
    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDistIndicies[i]]
        classCount[voteLabel] = classCount.get(voteLabel,0) + 1#在字典classCount中通过 键 查找相应的值，如果字典中没有键，值为0；有获取其值；；最后加1
        print classCount

    sortedClassCount = sorted(classCount.items(),key=lambda a:a[1],reverse=True)#根据出现次数排序，降序排序
    return sortedClassCount[0][0]#返回概率最大（出现次数最多的键，即类别）

if __name__ == '__main__':
    dataSet, labels = createDataSet()
    inX = [0,0.2]
    cate = classify0(inX,dataSet,labels,3)
    print (cate)