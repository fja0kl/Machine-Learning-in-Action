#coding:utf8
from numpy import *

#数据库中数据集为：feature1  feature1    feature1    label
def file2Matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()#直接将数据全部读取进来，形成列表，每一行为一个元素
    numberOfLines = len(arrayOLines)
    returnMatrix = zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMatrix[index,:] = listFromLine[0:3]#赋值方式很厉害！
        classLabelVector.append(listFromLine[-1])
        index += 1
    return returnMatrix, classLabelVector

def autoNorm(dataSet):
    """
    归一化：消除取值范围较大的 特征对分类的主导影响作用
    :param dataSet: 数据集
    :return:
    normDataSet：归一化的数据集{0~1 之间}
    ranges：数据中每一特征对应的数值范围；shape:(1,len(features))
    minVals：数据中每一特征对应的最小值; shape:(1,len(features))
    ----方便对新数据进行归一化操作
    """
    minVals = dataSet.min(0)#a.min() :全部的最小值；；a.min(axis=0):每列中的最小值；a.min(axis=1):每行中的最小值
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    # normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]#行数
    normDataSet = dataSet - tile(minVals, (m,1))#(m,n)重复次数：行方向上重复m次，列方向上重复1次
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet, ranges, minVals

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX,(dataSetSize,1)) - dataSet#tile：将inx扩展，扩展后的形状为（dataSetSize，1）；然后进行运算；；；；目的是通过矩阵来，加快运算速度
    sqDiffMat = diffMat ** 2
    sqDiffMat = sqDiffMat.sum(axis=1)#按列求和
    distances = sqDiffMat ** 0.5
    sortedDistIndicies = distances.argsort()#numpy argsort函数返回的是数组值从小到大的索引值
    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDistIndicies[i]]
        classCount[voteLabel] = classCount.get(voteLabel,0) + 1#在字典classCount中通过 键 查找相应的值，如果字典中没有键，值为0；有获取其值；；最后加1

    sortedClassCount = sorted(classCount.items(),key=lambda a:a[1],reverse=True)#根据出现次数排序，降序排序
    return sortedClassCount[0][0]#返回概率最大（出现次数最多的键，即类别）

def datingClassTest():
    """
    使用测试数据测试分类器效果
    :return: 误分率
    """
    hoRatio = 0.10#测试数据所占的比例---将数据集按照1:9的比例划分，1是测试集；9是训练集
    datingDataMat,datingLabels = file2Matrix('datingTestSet2.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]#数据集总数
    numTestVecs = int(m*hoRatio)#测试集数目
    errorCount = 0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],\
                                     datingLabels[numTestVecs:m],3)
        print ("the classifier came back with: %s,the real answer is: %s" \
               %(classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1
    print ("the total error rate is: %f" %(errorCount/float(numTestVecs)))

def classifyPerson():
    """
    应用分类器进行分类
    输入测试样例数据；进行分类
    :return: 分类结果
    """
    resultList = ["not at all","in small doses","in large doses"]
    percentTats = float(raw_input(\
        "percentage of time spent playing video games?"))#raw_input:读取控制台输入数据，赋值给变量，等效于C++里的cin>>a;
    ffMiles = float(raw_input("frequent flier miles earned per year?"))
    iceCream = float(raw_input("liters of ice cream consumed per year?"))

    datingDataMat,datingLabels = file2Matrix("datingTestSet2.txt")
    normMat,ranges,minVals = autoNorm(datingDataMat)

    inArr = array([ffMiles,percentTats,iceCream])
    classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels,k=3)#对输入数据进行归一化，然后再分类
    print ("You will probably like this person: %s" % resultList[int(classifierResult)-1])

def classifyPerson1():
    """
    应用分类器进行分类
    输入测试样例数据；进行分类
    :return: 分类结果
    """
    resultList = ["不喜欢","魅力一般","极具魅力（韩国欧巴）"]
    percentTats = float(raw_input(\
        "花费在游戏、视频上的时间百分比?"))#raw_input:读取控制台输入数据，赋值给变量，等效于C++里的cin>>a;
    ffMiles = float(raw_input("每月工资有多少?"))
    iceCream = float(raw_input("每年消耗的冰淇淋有多少升?"))

    datingDataMat,datingLabels = file2Matrix("datingTestSet2.txt")
    normMat,ranges,minVals = autoNorm(datingDataMat)

    inArr = array([ffMiles,percentTats,iceCream])
    classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels,k=3)#对输入数据进行归一化，然后再分类
    print ("你将约会的人很可能是: %s" % resultList[int(classifierResult)-1])

if __name__ == '__main__':
    classifyPerson1()