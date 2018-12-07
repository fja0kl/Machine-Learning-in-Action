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
# 计算的是当前点的lr模型:训练集中每个点一个模型;当测试点变化时,需要重新计算.
# 当数据集非常大时,效率不高.
def lwlr(point, data, labels, k=1.0):
    # ws = (X.T*W*X).I * X.T*W*y
    xMat = np.mat(data); yMat = np.mat(labels).T
    m, _ = np.shape(xMat)
    # 样本点权重矩阵初始化,一个对角矩阵
    weights = np.mat(np.eye(m))
    # 根据其他点与point之间的距离计算样本点权重
    for i in range(m):
        diffMat = point - xMat[i, :]
        weights[i, i] = np.exp(diffMat * diffMat.T/(-2.0*k**2))
    xTx = xMat.T * (weights * xMat)
    if np.linalg.det(xTx) == 0.0:
        print('奇异矩阵,不可逆')
        return
    ws = xTx.I * xMat.T * (weights * yMat)
    # 返回当前点的预测结果
    return point*ws

def lwlrTest(testArr, data, labels, k=1.0):
    # 批量测试
    m, _ = np.shape(testArr)
    yHat = np.zeros((m,1))
    for i in range(m):
        yHat[i] =lwlr(testArr[i,:],data, labels, k)
    
    return yHat

def ridgeReg(data, labels, lmbda=0.2):
    # ws = （X.T * X + lam*I）.I * (X.T * y)
    xMat = np.mat(data); yMat = np.mat(labels).T
    m, _ = np.shape(xMat)
    xTx = xMat.T * xMat
    demon = xTx + lmbda * np.eye(m)
    ws = demon.I * (xMat.T * yMat)

    return ws