#coding:utf8
from math import log

def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        labelCounts[currentLabel] = labelCounts.get(currentLabel,0) + 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = labelCounts[key]/float(numEntries)
        shannonEnt -= prob*log(prob, 2)
    return shannonEnt

def createDataSet():
    dataSet = [[1,1,'yes'],
               [1, 1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0,1,'no']]
    labels = ['no surfacing','flippers']#特征
    return dataSet, labels

def splitDataSet(dataSet, axis, value):
    """
    对数据集进行划分
    :param dataSet:数据集
    :param axis: 选择的特征，在该特征上对数据集进行划分;;;最好的划分方式
    :param value: 特征的取值；划分依据
    :return: 划分后的子集；；方便进行下一次划分；；递归执行
    """
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    """
    选择最好的特征对数据集进行划分
    :param dataSet: 训练数据集
    :return: 选择的数据特征下标
    """
    numFeatures = len(dataSet[0]) - 1#dataSet 数据格式：每一行最后为类标签
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob*calcShannonEnt(subDataSet)#加权
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    """
    生成决策树的停止条件：1.遍历完所有的属性；2.每个分支下所有实例都属于同一分类。
    该函数用来解决第一种情况：遍历完所有属性后，如何确定该节点的类别（投票原则-多数表决）
    :param classList: 叶子结点中的实例，类别标签列表
    :return: 该叶子结点的类别
    """
    classCnt = {}
    for vote in classList:
        classCnt[vote] = classCnt.get(vote, 0) + 1
    sortedClassCount = sorted(classCnt.items(),key=lambda a:a[1],reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    """
    创建决策树
    :param dataSet:数据集
    :param labels: 特征名称列表
    :return: 决策树
    """
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):#结束条件1：结点中所有实例都是一个类别；
        return classList[0]
    if len(dataSet[0]) == 1:#结束条件2：结点中，分类特征只剩下一个---多数表决
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]#特征的名称，eg：工资、性别等等----中间结点，进行判断
    myTree = {bestFeatLabel:{}}#第一次划分已经完成；对子节点进行划分

    del labels[bestFeat]
    featValues = [example[bestFeat] for example in dataSet]
    uniqueValues = set(featValues)
    for value in uniqueValues:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)

    return myTree

def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key],featLabels,testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

def storeTree(inputTree, filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)

if __name__ == '__main__':
    tree = grabTree('classifierStorage.txt')
    print tree
