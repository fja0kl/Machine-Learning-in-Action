#coding:utf8
import matplotlib.pyplot as plt
import numpy as np

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

# 普通的线性回归 ---> 欠拟合
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
    for j in range(m):
        diffMat = testPoint - xMat[j,:]
        weights[j,j] = np.exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T * (weights*xMat)
    if np.linalg.det(xTx) == 0.0:
        print ("矩阵不可逆")
        return
    ws = xTx.I * (xMat.T *(weights*yMat))
    return testPoint*ws

# 对测试集进行分类
def lwlrTest(testArr, xArr, yArr, k=1.0):
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat

def rssError(yArr, yHatArr):
    return ((yArr-yHatArr)**2).sum()

#数据的特征数目 比样本点还多：处理方法
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

def ridgeTest(xArr,yArr,numTestPts = 30):
    """
    在一组lam上测试结果，find the best lambda
    :param xArr: 输入数据数组
    :param yArr: 标签数组，行方向
    :param numTestPts: 迭代次数
    :return: 不同lam上求得回归系数ws矩阵
    """
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    yMean = np.mean(yMat,0)#求列方向上（也只有列）的均值，即：每个特征的均值，然后进行 0均值处理
    yMat = yMat - yMean
    xMeans = np.mean(xMat,0)
    xStd = np.std(xMat,0)#计算xMat矩阵列方向上的标准差；
    xMat = (xMat - xMeans)/xStd#标准化，在 0~1 之间

    wMat = np.zeros((numTestPts,np.shape(xMat)[1]))#设置ws数组；lam个数 * 特征数
    for i in range(numTestPts):
        ws = ridgeRegres(xMat,yMat,np.exp(i-10)) #lambda 逐渐衰减
        wMat[i,:] = ws.T
    return wMat

def regularize(xMat):
    """
    对数据矩阵进行标准化：0均值，单位方差
    :param xMat:
    :return:
    """
    inMat = xMat.copy()
    inMeans = np.mean(inMat,axis=0)
    # inStd = np.std(inMat,axis=0)
    # inMat = (inMat - inMeans)/inStd
    inVar= np.var(inMat,axis=0)
    inMat = (inMat - inMeans)/inVar
    return inMat

# 前向逐步回归，与lasso算法得到的效果差不多，但是更加简单
def stageWise(xArr,yArr,eps=0.005,numIt=1000):
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    yMean = np.mean(yMat,axis=0)
    yMat = yMat - yMean
    xMat = (xMat-np.mean(xMat, axis=0))/np.var(xMat, 0)
    m, n = np.shape(xMat)
    returnMat = np.zeros((numIt, n))
    ws = np.zeros((n, 1)); wsTest = ws.copy(); wsMax = ws.copy()
    for i in range(numIt):# 每次迭代
        print i
        print ws.T
        lowestError = np.inf #当前最小误差为正无穷
        for j in range(n): # 每个特征
            for sign in [-1,1]: #增大 or 缩小
                wsTest = ws.copy() #防止改变ws矩阵
                wsTest[j] += eps*sign
                yTest = xMat * wsTest
                rssE = rssError(yMat.A, yTest.A)#新W下的误差
                if rssE < lowestError: #误差判断？
                    lowestError = rssE
                    wsMax = wsTest #当前特征下，增大or缩小改变情况；最优解
        ws = wsMax.copy()
        returnMat[i, :] = ws.T
    return returnMat


if __name__ == '__main__':

    xArr, yArr = loadDataSet('abalone.txt')
    ws = stageWise(xArr,yArr)
    plt.plot(ws)
    plt.show()

    # xArr, yArr = loadDataSet('ex0.txt')
    # yHat = lwlrTest(xArr,xArr,yArr,0.3)
    # print yHat
    # xMat = np.mat(xArr)
    # srtInd = xMat[:, 1].argsort()
    # xSort = xMat[srtInd][:,0,:]
    # fig = plt.figure()
    # fig.clf()
    # ax = fig.add_subplot(111)
    # ax.plot(xSort[:,1],yHat[srtInd])
    # ax.scatter(xMat[:,1].flatten().A[0], np.mat(yArr).T.flatten().A[0], s=2,c='red')
    # plt.show()
