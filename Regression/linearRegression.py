#coding:utf8
from numpy import mat,linalg,exp,shape,eye,zeros,matmul
import matplotlib.pyplot as plt


def loadDataSet(filename):
    dataMat = []; labelMat = []
    numFeat = len(open(filename).readline().strip().split('\t')) -1
    fr = open(filename)
    for line in fr.readlines():
        line = line.strip().split('\t')
        lineArr = []
        for i in range(numFeat):
            lineArr.append(float(line[i]))
        dataMat.append(lineArr)
        labelMat.append(float(lineArr[-1]))
    return dataMat, labelMat

def standRegres(xArr,yArr):
    xMat = mat(xArr); yMat = mat(yArr).T
    xTx = xMat.T * xMat
    if linalg.det(xTx) == 0.0:
        print ("矩阵不可逆")
        return
    ws = xTx.I * (xMat.T*yMat)
    return ws

def lwlr(testPoint, xArr, yArr,k=1.0):
    xMat = mat(xArr); yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye(m))
    for j in range(m):
        diffMat = testPoint - xMat[j,:]
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T * (weights*xMat)
    if linalg.det(xTx) == 0.0:
        print ("矩阵不可逆")
        return
    ws = xTx.I * (xMat.T *(weights*yMat))
    return testPoint*ws

def lwlrTest(testArr, xArr, yArr, k=1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat


if __name__ == '__main__':
    xArr, yArr = loadDataSet('ex0.txt')
    yHat = lwlrTest(xArr,xArr,yArr,0.3)
    print yHat
    xMat = mat(xArr)
    srtInd = xMat[:, 1].argsort()
    xSort = xMat[srtInd][:,0,:]
    fig = plt.figure()
    fig.clf()
    ax = fig.add_subplot(111)
    ax.plot(xSort[:,1],yHat[srtInd])
    ax.scatter(xMat[:,1].flatten().A[0], mat(yArr).T.flatten().A[0], s=2,c='red')
    plt.show()
