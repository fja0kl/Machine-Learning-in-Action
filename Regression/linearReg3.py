import numpy as np

def loadData(filename):
    data = []; labels = []
    with open(filename) as f:
        numFeats = len(f.readline().split('\t')) - 1
        for line in f.readlines():
            row = line.strip().split('\t')
            rowArr = []
            for i in range(numFeats):
                rowArr.append(float(row[i]))
            data.append(rowArr)
            labels.append(float(row[-1]))

    return data, labels

"""
线性回归:通过正规方程,直接计算解析解.
w = (x.T * x).I * x.T * y

x: 第一列是1, 把偏置包括在w系数里.
"""
def linearReg(data, labels):
    dataMat = np.mat(data); labelsMat = np.mat(labels).T
    xTx = dataMat.T * dataMat
    # 判断xTx是否是奇异矩阵
    if np.linalg.det(xTx) == 0:
        print("矩阵xTx不可逆,无法求解")
        return
    # 通过正规方程计算解析解.
    w = xTx.I * dataMat.T * labelsMat
    return w

# 计算误差:误差公式为残差平方和.residual sum of square
def rssError(y, yHat):
    return np.sum(np.power(y-yHat, 2))

# 回归模型预测
def predict(ws, X):
    return X * ws

def lwlr(data, labels):
    