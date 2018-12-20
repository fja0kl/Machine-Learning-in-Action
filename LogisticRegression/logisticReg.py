#coding:utf8
from numpy import *
import matplotlib.pyplot as plt

def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('./data/testSet.txt','r')
    for line in fr.readlines():
        line = line.strip().split()
        dataMat.append([1.0, float(line[0]), float(line[1])])
        labelMat.append(int(line[2]))
    return dataMat, labelMat

def sigmod(x):
    return 1/(1+exp(-x))

def gradAscent(dataMatIn, classLabels):
    dataMat = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m, n = shape(dataMat)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n,1))
    for k in range(maxCycles):
        h = sigmod(dataMat*weights)
        error = (labelMat - h)
        weights = weights + alpha*dataMat.transpose()*error#省去了数学推导式，但是确实是这么计算的，只是简化了中间推导过程
    return weights

def stocGradAscent0(dataMat,classLabels):
    m, n =shape(dataMat)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmod(sum(dataMat[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha*error*dataMat[i]
    return weights

def stocGradAscent1(dataMat, classLabels, numIter=150):
    m,n = shape(dataMat)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0+j+i) + 0.1
            randIndex = int(random.uniform(0,len(dataIndex)))
            h = sigmod(sum(dataMat[randIndex]*weights)) - classLabels
            error = classLabels[randIndex] - h
            weights = weights + alpha*error*dataMat[randIndex]
            del dataIndex[randIndex]
    return weights

def classifyVector(inX, weights):
    prob = sigmod(sum(inX*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def plotBestFit(weights):
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    m = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = [];ycord2 = []

    for i in range(m):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1, s=30,c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1');plt.ylabel('X2')
    plt.show()

if __name__ == '__main__':
    dataMat, labelMat = loadDataSet()
    weights = stocGradAscent1(array(dataMat),labelMat)
    print weights
    plotBestFit(weights)