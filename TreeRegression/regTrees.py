#coding:utf8
from numpy import *

class treeNode():
    def __init__(self, feat, val, right, left):
        featureToSplitOn = feat
        valueOfSplit = val
        rightBranch = right
        leftBranch = left

def loadDataSet(filename):
    dataMat = []
    with open(filename) as fr:
        for line in fr.readlines():
            curLine = line.strip().split('\t')
            fltLine = map(float, curLine) # 将curLine列表中的元素，变成float类型；返回list
            # map(function, sequence[, sequence, ...]) -> list 将function应用到可迭代对象的组成元素上，返回list
            dataMat.append(fltLine)
    return dataMat

def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0], :]
    print nonzero(dataSet[:,feature] > value)
    mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0], :]
    # 解释一下：
    # dataSet[nonzero(dataSet[:,feature] > value)[0], :][0]
    # nonzero(dataSet[:,feature] > value)[0] 返回两个维度上的结果，现在只取行维度上，满足条件的结果；得到满足条件的行坐标
    # 然后依据得到的结果对数据集进行裁剪，划分；得到两个矩阵 a，b
    return mat0, mat1

def createTree(dataSet, leafType=regLeaf, errorType=regErr, ops=(1,4)):
    """
    回归树，模型树；叶子类型不同，误差计算函数不同，需要其他参数的元组
    :param dataSet:
    :param leafType: 叶子类型；
    :param errorType: 误差计算函数
    :param ops: 树构建过程中需要的其他参数；以元组形式给出；
    :return: 树
    """
    feat, val = chooseBestSplit(dataSet, leafType, errorType, ops)
    if feat == None:
        return val
    retTree = {}
    retTree['spInd'] = feat #划分特征的下标
    retTree['spVal'] = val #划分特征的分类值
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errorType, ops)
    retTree['right'] = createTree(rSet, leafType, errorType, ops)

    return retTree

def regLeaf(dataSet):
    return mean(dataSet[:, -1])

def regErr(dataSet):
    return var(dataSet[:, -1]) * shape(dataSet)[0]

def chooseBestSplit(dataSet, leafType=regLeaf, errorType=regErr, ops=(1,4)):
    tolS = ops[0]; tolN = ops[1]
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    m, n = shape(dataSet)
    S = errorType
    bestS = inf; bestIndex = 0; bestValue = 0
    for featIndex in range(n-1):
        for splitVal in set(dataSet[:, featIndex]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN): continue
            newS = errorType(mat0) + errorType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if (S - bestS) < tolS:
        return None, leafType(dataSet)
    mat0 , mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    return bestIndex, bestValue