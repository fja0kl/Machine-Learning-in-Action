#coding:utf8
import matplotlib.pyplot as plt
import numpy as np
import random

def loadDataSet(filename):
    dataMat = []
    labelMat = []
    numFeat = len(open(filename).readline().split('\t')) - 1
    fr = open(filename)
    for line in fr.readlines():
        line = line.strip().split('\t')
        lineArr = []
        for i in range(numFeat):
            lineArr.append(float(line[i]))
        dataMat.append(lineArr)
        labelMat.append(float(line[-1]))
    return dataMat, labelMat


# 普通的线性回归 ---> 欠拟合
# 线性回归：一条直线 y = ax + b
# 优化函数：平方误差  ols：普通最小二乘
# 回归系数：a，b
def standRegres(xArr,yArr):
    """
    ws = (X.T * X).I * X.T*y
    :param xArr: 输入的数据数组
    :param yArr: 标签数组，list
    :return: ws回归系数
    """
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) == 0.0:
        print ("矩阵不可逆")
        return
    ws = xTx.I * (xMat.T*yMat)
    return ws

# 局部加权线性回归 ，给每个数据点赋予权重--->在求权重回归系数时添加
# 每个数据点 都赋予权重系数，---数据点的权重；与预测点越远，权重系数越小；
# 然后，计算回归系数ws；
# 属于非参数学习方法：当样本点改变，参数需要重新计算，不是固定不变的；
# 因此，当数据集很大时，计算量非常大
def lwlr(testPoint, xArr, yArr,k=1.0):
    """
    ws = (X.T*W*X).I * X.T*W*y
    :param testPoint: 输入的数据点
    :param xArr: 输入数据数组
    :param yArr: 标签数组
    :param k: 高斯核的参数k
    :return: 输入数据的分类结果
    """
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    m = np.shape(xMat)[0]
    weights = np.mat(np.eye(m))
    for j in range(m): # m个样本点
        diffMat = testPoint - xMat[j,:] # 计算周围点与当前计算点，之间的差距
        weights[j, j] = np.exp(diffMat*diffMat.T/(-2.0*k**2)) # 权重系数计算公式；对角，
    xTx = xMat.T * (weights*xMat)
    if np.linalg.det(xTx) == 0.0:
        print ("矩阵不可逆")
        return
    ws = xTx.I * (xMat.T *(weights*yMat))
    return testPoint*ws #加权后的分类结果

# 使用lwlr局部加权线性回归对测试集进行分类
def lwlrTest(testArr, xArr, yArr, k=1.0):
    """
    使用局部线性加权回归对测试集进行预测，返回分类结果。
    :param testArr: 测试集
    :param xArr: 训练集
    :param yArr: 训练集标签
    :param k: 波长；类高斯核函数的k值大小；
    :return: 测试集的回归值，预测值。
    """
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat

# 误差估计
def rssError(yArr, yHatArr):
    """
    计算回归系数的优劣：参数指标，误差和。
    :param yArr: 实际值
    :param yHatArr: 预测值
    :return: 误差平方和
    """
    return ((yArr-yHatArr)**2).sum()

# 数据的特征数目 比样本点还多：处理方法
# 岭回归
# lasso
# 前向逐步回归
# LAR，PCA回归

# 岭回归 ：数据的特征 > 样本点 ；岭：单位矩阵I
def ridgeRegres(xMat,yMat,lam=0.2):
    """
    依据岭回归，计算回归系数ws
    ws = （X.T * X + lam*I）.I * (X.T * y)
    :param xMat: 数据矩阵
    :param yMat: 标签矩阵，列向量
    :param lam: lam参数
    :return: 回归系数ws #列向量
    """
    xTx = xMat.T*xMat
    demon = xTx + np.eye(np.shape(xMat)[1])*lam
    if np.linalg.det(demon) == 0.0:
        print ("矩阵不可逆")
        return
    ws = demon.I * (xMat.T*yMat)
    return ws

def regularize(xMat):
    """
    对数据矩阵进行标准化：0均值，单位方差
    实际上标准化过程，应该减去均值，然后除以标准差，而不是方差！
    :param xMat:
    :return:训练集标准化后的结果；
    """
    inMat = xMat.copy()
    inMeans = np.mean(inMat,axis=0)
    # inVar = np.var(inMat, axis=0)
    # inMat = (inMat - inMeans)/inVar
    inStd = np.std(inMat,axis=0)
    inMat = (inMat - inMeans)/inStd
    return inMat

def ridgeTest(xArr,yArr,numTestPts = 30):
    """
    在一组lam上测试结果，find the best lambda
    :param xArr: 输入数据数组
    :param yArr: 标签数组，行方向
    :param numTestPts: 迭代次数,用来控制不同的步长；lambda
    :return: 不同lam上求得回归系数ws矩阵
    """
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    yMean = np.mean(yMat,0)#求列方向上（也只有列）的均值，即：每个特征的均值，然后进行 0均值处理
    yMat = yMat - yMean
    # xMat = regularize(xMat)
    xMeans = np.mean(xMat,0)
    xStd = np.std(xMat,0)#计算xMat矩阵列方向上的标准差；
    xMat = (xMat - xMeans)/xStd#标准化，在 0~1 之间

    wMat = np.zeros((numTestPts,np.shape(xMat)[1]))#设置ws数组；lam个数 * 特征数
    for i in range(numTestPts):
        ws = ridgeRegres(xMat,yMat,np.exp(i-10)) #lambda 逐渐衰减
        wMat[i,:] = ws.T
    return wMat

# 前向逐步回归，与lasso算法得到的效果差不多，但是更加简单
# 最终,只是针对一个特征的回归系数进行了修改，并不是全部的；
# 即：是通过对全部特征对应的权重系数，进行增加或减少，寻找最好的权重系数---只有一个特征对应的权重系数不为0；其余都是0
def stageWise(xArr,yArr,eps=0.005,numIt=100):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat,axis=0)
    yMat = yMat - yMean
    xMat = regularize(xMat)
    m, n = np.shape(xMat)
    returnMat = np.zeros((numIt, n))
    ws = np.zeros((n, 1))
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):# 每次迭代
        lowestError = np.inf #当前最小误差为正无穷
        for j in range(n): # 每个特征
            for sign in [-1,1]: #增大 or 缩小
                wsTest = ws.copy() #防止改变ws矩阵
                wsTest[j] += eps*sign
                yTest = xMat * wsTest
                rssE = rssError(yMat.A, yTest.A)#新W下的误差
                if rssE < lowestError: #误差判断？ 对当前特征，找到最好的权重系数
                    lowestError = rssE
                    wsMax = wsTest #当前特征下，增大or缩小改变情况；最优解
        ws = wsMax.copy() #当前迭代的回归系数结果。
        returnMat[i, :] = ws.T
    return returnMat
# 交叉验证
def crossValidation(xArr, yArr, numVal=10):
    """
    通过交叉验证，寻找最好的参数设置；
    十折交叉验证
    :param xArr:
    :param yArr:
    :param numVal:
    :return:
    """
    m = len(yArr)
    indexList = range(m)
    errorMat = np.zeros((numVal,30))
    for i in range(numVal):
        # 划分训练集和测试集 比例 9：1
        trainX = [] ;trainY = []
        testX = []; testY = []
        random.shuffle(indexList)
        for j in range(m):
            if j < m*0.9:
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        wMat = ridgeTest(trainX,trainY) # 岭回归
        for k in range(30):# 岭回归，重复30次
            matTestX = np.mat(testX); matTrainX = np.mat(trainX)
            meanTrain = np.mean(matTrainX,0)
            varTrain = np.var(matTrainX,0)
            matTestX = (matTestX-meanTrain)/varTrain # 标准化
            yEst = matTestX * np.mat(wMat[k,:]).T + np.mean(trainY) #回复到和标签trainY的范围内，即去标准化。
            errorMat[i,k] = rssError(yEst.T.A,np.array(testY))
    meanError = np.mean(errorMat,0) # 针对不同步长lambda的numIter次循环结果取平均；
    print np.shape(errorMat) # 10 * 30
    minMean = float(min(meanError)) # 找最小的
    bestWeights = wMat[np.nonzero(meanError == minMean)] #在最后一次循环结果上取回归系数；确定步长是主要问题；
    print np.shape(wMat) # 30 * 8
    print bestWeights
    print bestWeights == wMat[22]
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    meanX = np.mean(xMat,0); varX = np.mean(xMat,0)
    unReg = bestWeights/varX
    print ("the best model from Ridge Regression is:\n" + str(unReg))
    print ("with constant term:")
    print (-1*sum(np.multiply(meanX,unReg)) + np.mean(yMat))

def originDataPlot():
    xArr, yArr = loadDataSet('ex0.txt')
    xMat = np.mat(xArr)

    fig = plt.figure()
    fig.clf()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:,1].flatten().A,yArr,c='r')
    plt.show()

def standRegresPlot():
    """
    通过画图，查看其分类效果。
    :return:
    """
    xArr, yArr = loadDataSet('ex0.txt')
    ws = standRegres(xArr,yArr)
    print np.shape(ws)
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    yHat = xMat * ws
    # 向量-->默认为列向量；
    # xMat:m * n-->m：训练样例数目；n：每条记录的特征数目；
    # ws-->列向量 n*1：n个特征数
    # 所以，经常是 xMat * ws
    # 而不是ws * xMat
    fig = plt.figure()
    fig.clf()
    ax = fig.add_subplot(111)
    # 实际结果：散点图
    ax.scatter(xMat[:,1].flatten().A,yMat.A,s=20,c='r')#x 横坐标，y纵坐标;s散点的面积
    # xCopy = xMat.copy()
    # xCopy.sort(0)# 修改了xCopy---copy xMat对象;;;无所谓排序，反正是一条直线
    ax.plot(xMat[:,1], yHat)# 分类；拟合效果
    plt.show()

# 折线图：根据x，y坐标，依次连接，形成！
# so，最好根据对应关系，排序，对排序后的结果进行画图。
def lwlrTestPlot():
    xArr, yArr = loadDataSet('ex0.txt')
    yHat = lwlrTest(xArr, xArr, yArr, 0.003)
    xMat = np.mat(xArr)

    fig = plt.figure()
    fig.clf()
    ax = fig.add_subplot(111)
    srtInd = xMat[:,1].argsort(0)
    xSort = xMat.copy()
    xSort.sort(0)
    ax.plot(xSort[:, 1], yHat[srtInd])
    ax.scatter(xMat[:, 1].flatten().A, np.mat(yArr).A, s=2, c='red')
    plt.show()

def stageWiseTestPlot():
    xArr, yArr = loadDataSet('abalone.txt')
    ws = stageWise(xArr, yArr, 0.001, 2)
    plt.plot(ws)
    plt.show()



if __name__ == '__main__':
    xArr, yArr = loadDataSet('abalone.txt')
    crossValidation(xArr,yArr)


