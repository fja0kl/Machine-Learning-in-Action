#coding:utf8
from numpy import *

def sigmod(x):
    return 1/(1+exp(-x))

def gradAscent(dataMatIn, classLabels):
    dataMat = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m,n = shape(dataMat)
    alpha = 0.01
    weights = ones((n,1))
    maxIters = 500
    for i in range(maxIters):
        h = sigmod(dataMat*weights)#m*1
        error = (labelMat - h)#m*1
        weights = weights + alpha*dataMat.transpose()*error
    return weights

def stocGradAscent0(dataMat,classLabels):
    """
    随机梯度算法：每次更新使用每个样例进行计算；而不是梯度算法中的所有数据进行过一次更新
    每次更新选择的随机样本按照矩阵顺序依次选择，计算
    :param dataMat: 数据矩阵 m*n
    :param classLabels: 标签矩阵 1*m
    :return:权重矩阵
    """
    m, n = shape(dataMat)
    alpha = 0.01
    weights = ones(n)#行向量,初始化为 1;0 也行
    for i in range(m):
        h = sigmod(sum(dataMat[i]*weights))
        error = classLabels[i] - h#一个数 just a number
        weights = weights + alpha*error*dataMat[i]
    return weights

def stocGradAscent1(dataMat, classLabels, numIters=150):
    """
    改进的随机梯度算法
    设置迭代次数；每次更新选择的训练样本是随机的
    学习率alpha随着迭代次数的增加，逐渐减小
    :param dataMat:
    :param classLabels:
    :param numIters:
    :return:
    """
    m,n = shape(dataMat)
    weights = ones(n)
    for j in range(numIters):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.1
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmod(sum(dataMat[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMat[randIndex]
            del dataIndex[randIndex]
    return weights

def classifyVector(inX, weights):
    prob = sigmod(sum(inX*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainSet = []; trainLabels = []
    for line in frTrain.readlines():
        line = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(line[i]))
        trainSet.append(lineArr)
        trainLabels.append(float(line[21]))
    trainWeights = stocGradAscent1(array(trainSet),trainLabels)
    errorCnt = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr),trainWeights)) != int(currLine[21]):
            errorCnt += 1
    errorRate = float(errorCnt)/numTestVec
    print ("the error rate of this test is: %f" % errorRate)
    return errorRate

def multiTest():
    numTests = 100; errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print ("after %d iterations the average error rate is: %f" %(numTests, float(errorSum)/numTests))


if __name__ == '__main__':
    multiTest()