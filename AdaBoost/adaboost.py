#coding:utf8
from numpy import *

def loadSimpData():
    dataMat = matrix([[1.,2.1],
                     [2.,1.1],
                     [1.3,1.],
                     [1., 1.],
                     [2.,1.]])
    classLabels = [1.0,1.0,-1.0,-1.0,1.0]
    return dataMat,classLabels

def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    """
    单层决策树分类器
    :param dataMatrix:数据集
    :param dimen: 维度，即分类特征。
    :param threshVal: 阈值
    :param threshIneq: 规则：less than or greater than
    :return: 分类后的结果。list
    """
    retArray = ones((shape(dataMatrix)[0],1))#赋初始值：都为1
    if threshIneq == 'lt':#根据规则进行筛选，分类
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0#根据阈值，在分类特征上进行分类
        #dataMatrix[:,dimen] <= threshVal：列表选择式：在数据集dataMatrix里的所有记录的dimen维的数值 小于 阈值，获取其下标
        #分类效果就是:小于阈值的记录 划分成 负类，即-1。
    else:#评价标准不一样
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray

#弱分类算法
def buildStump(dataArr, classLabels,D):
    """
    寻找最低错误率的决策树
    :param dataArr: 数据集
    :param classLabels: 分类标签
    :param D: 权重向量【每个记录有一个权重】
    :return: 决策树，最小误差，预测的分类结果
    """
    dataMatrix = mat(dataArr); labelMat = mat(classLabels).transpose()
    m,n = shape(dataMatrix)
    numSteps = 10.0; bestStump = {}; bestClasEst = mat(zeros((m,1)))
    #numStep 迭代次数；用来设定分类时特征阈值
    minError = inf#无穷大
    for i in range(n):#分类特征---分类特征长度可能大于1；；；所以，要在所有可能的分类特征上循环；find min
        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max()
        stepSize = (rangeMax-rangeMin)/numSteps#每次迭代的改变量；寻找特征i上的最值，除以迭代次数，求得。
        for j in range(-1,int(numSteps)+1):#循环，寻找最佳阈值
            for inequal in ['lt','gt']:#对不同规则进行循环；不同规则，即使相同的特征、相同的阈值，分类效果也不同
                threshVal = (rangeMin + float(j) * stepSize)#修改阈值
                predictVals = stumpClassify(dataMatrix,i,threshVal,inequal)
                errArr = mat(ones((m,1)))
                errArr[predictVals == labelMat] = 0#器重误分的数据；正确分类：误差为0；
                weightedError = D.T*errArr#误差：误分的样例乘权重系数，再求和，得到误差。
                print ("split: dim %d,thresh %.2f, thresh inequal: %s, the weighted error is %.3f" %\
                       (i,threshVal, inequal, weightedError))
                if weightedError < minError:#寻找最小误差；保留运算结果
                    minError = weightedError
                    bestClasEst = predictVals.copy()
                    bestStump['dim'] = i#分类特征
                    bestStump['thresh'] = threshVal#阈值
                    bestStump['ineq'] = inequal#规则
    return bestStump, minError, bestClasEst#

def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m,1))/m)#初始值：相等；之后迭代是基于上次的结果--->关注错分样本
    aggClassEst = mat(zeros((m,1)))
    for i in range(numIt):
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)
        print ("D:")
        print (D.T)
        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))# 计算alpha系数；error误分率
        bestStump['alpha'] = alpha#alpha体现分类的分类效果。效果越好，系数越大；无穷大
        weakClassArr.append(bestStump)
        print ("classEst:")
        print (classEst.T)#分类效果

        # 对权重系数进行更新 D（i+1) = D(i)*exp(-alpha*y*h(x))/Z
        # 其中，alpha = -alpha*y*h：将正确分类和误分类两种情况整合到了一起---perfect
        # Z：用来进行标准化操作
        expon = multiply(-1*alpha*mat(classLabels).T,classEst)
        D = multiply(D, exp(expon))
        D = D/D.sum()

        aggClassEst += alpha*classEst#强分类器的分类结果---由多个弱分类整合到一起，作为一个强分类器，进行分类
        # 运算结果，就是在各个分类器的分类进行汇总，统计。
        print ("aggClassEst:")
        print (aggClassEst.T)
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1)))#强分类的分类误差
        errorRate = aggErrors.sum()/m
        print ("total error: %f" % errorRate)
        if errorRate == 0.0:
            break
    return weakClassArr


def adaClassify(datToClass, classifierArr):
    dataMatrix = mat(datToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],\
                                 classifierArr[i]['thresh'],\
                                 classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha']*classEst
        print (aggClassEst)
    return sign(aggClassEst)


if __name__ == '__main__':
    dataMat, classLabels = loadSimpData()
    classifierArr = adaBoostTrainDS(dataMat,classLabels,30)
    p = adaClassify([[0,0.],[1.,5.]],classifierArr)