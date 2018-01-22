#coding:utf8
from numpy import mat,linalg

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

if __name__ == '__main__':
    xArr, yArr = loadDataSet('ex0.txt')
    print xArr[:2]
    ws = standRegres(xArr,yArr)
    print ws
