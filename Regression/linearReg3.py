import numpy as np
np.seterr(divide='ignore', invalid='ignore')

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

def ridgeReg(xMat, yMat, lmbda=0.2):
    # ws = （X.T * X + lam*I）.I * (X.T * y)
    # xMat = np.mat(data); yMat = np.mat(labels).T
    m, _ = np.shape(xMat)
    xTx = xMat.T * xMat
    # 保证矩阵可逆
    demon = xTx + lmbda * np.eye(m)
    if np.linalg.det(demon) == 0.0:#有可能lambda=0,所以还需要进行判断
        print ("矩阵不可逆")
        return
    ws = demon.I * (xMat.T * yMat)

    return ws

#判断不同lambda下ridge回归情况,不同lambda对回归系数的影响如何?
def ridgeTestLamda(data, labels, numIters=30):
    #先对数据进行标准化处理
    xMat = np.mat(data); yMat = np.mat(labels).T
    yMean = np.mean(yMat, axis=0)#列向量
    yMat = yMat - yMean
    xMean = np.mean(xMat,axis=0)
    xStd = np.std(xMat,axis=0)
    xMat = (xMat - xMean)/xStd

    wMat = np.zeros((numIters, np.shape(xMat)[1]))
    for i in range(numIters):
        #lambda = exp(i-10): 由小到大,看系数如何变化----逐渐趋于0
        ws = ridgeReg(xMat, yMat, np.exp(i-10))
        wMat[i, :] = ws.T #变成行向量
    
    return wMat


# 权重系数的优化:通过设置最大迭代次数来终止
# stagewise一种贪心算法:每次迭代,尝试对每个特征进行优化,最终选择优化效果最好的特征进行优化;
# 也就是说每次迭代仅仅优化了一个特征(不同次迭代优化的特征可能相同)---因为对当前特征进行优化,提升效果最明显
def stagewise(data, labels, eps=0.005, numIters=20):
    # 0均值,1方差
    xMat = np.mat(data); yMat = np.mat(labels).T
    yMean = np.mean(yMat, axis=0)
    yMat -= yMean
    xMean = np.mean(xMat, axis=0)
    xVar = np.var(xMat, axis=0)
    xMat = (xMat-xMean)/xVar
    #保存迭代结果
    _, n = np.shape(xMat)
    wsMat = np.zeros((numIters, n))
    ws = np.zeros((n, 1))
    wsTmp = ws.copy()
    wsBest = ws.copy()

    #迭代
    for i in range(numIters):
        lowestErr = np.inf
        for j in range(n):
            #特征的每次变化:增加 or 减少 delta
            for sign in [-1,1]:
                wsTmp = ws.copy()
                wsTmp[j] += sign*eps
                #计算更改后对应结果
                yTest = xMat*wsTmp
                rss = rssError(yMat.A, yTest.A)#numpy ndarray形式,not matrix
                #比较误差大小,保存最小误差以及对应权重参数
                if rss < lowestErr:
                    lowestErr = rss
                    wsBest = wsTmp
        # 保存上次优化结果,方便继续进行优化;感觉和梯度下降算法类似,当前选择,梯度最大方向更新,之后在更新效果之后,接着进行更新.
        ws = wsBest.copy()#不会浪费之前的努力,站在巨人的肩膀上
        wsMat[i, :] = ws.T
    
    return wsMat


