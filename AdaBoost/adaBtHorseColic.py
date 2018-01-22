#coding:utf8
from numpy import *

def loadDataSet(filename):
    dataMat = []; labelMat = []
    fr = open(filename)
    n = len(fr.readline().strip().split('\t'))
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        data = []
        for i in range(n-1):
            data.append(float(lineArr[i]))
        dataMat.append(data)
        labelMat.append(float(lineArr[-1]))
    return dataMat, labelMat

def stumpClassify(dataMat, dimen, threshVal,threshIneq):
    m, _ = shape(dataMat)
    predLabels = ones((m, 1))
    if threshIneq == 'lt':
        predLabels[dataMat[:,dimen] <= threshVal] = -1.0
    else:
        predLabels[dataMat[:,dimen] > threshVal] = -1.0
    return predLabels

def buildStump(dataArr,classLables, D):
    """
    构建弱分类器：决策树类型；树桩
    :param dataArr:
    :param classLables:
    :param D:
    :return: 决策树；误差；预测结果
    """
    dataMat = mat(dataArr); classMat = mat(classLables).transpose()
    m,n = shape(dataMat)
    numSteps = 10; bestStump={}; bestClasEst = mat(zeros((m,1)))
    minError = inf
    for i in range(n):
        rangeMin = dataMat[:,i].min(); rangeMax = dataMat[:,i].max()
        stepSize = (rangeMax - rangeMin)/float(numSteps)
        for j in range(-1, int(numSteps+1)):
            for ineq in ['lt','gt']:
                threshVal = rangeMin + j*stepSize
                predLabels = stumpClassify(dataMat,i,threshVal,ineq)
                errArr = mat(zeros((m,1)))
                errArr[predLabels != classMat] = 1
                weightError = D.T*errArr

                if weightError < minError:
                    minError = weightError
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = ineq
                    bestClasEst = predLabels.copy()
    return bestStump,bestClasEst,minError

def adaBoostTrainDS(dataArr,classLabels, numIters=50):
    weakClfArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m,1))/m)
    aggClassEst = mat(zeros((m,1)))
    for i in range(numIters):
        bestStump,classEst,error = buildStump(dataArr,classLabels,D)
        alpha = float(0.5*log((1-error)/max(error,1e-16)))
        bestStump['alpha'] = alpha
        weakClfArr.append(bestStump)

        expon = multiply(-1*alpha*mat(classLabels).T,classEst)

        D = multiply(D,exp(expon))#对权重系数向量进行更新
        D = D/D.sum()

        aggClassEst += alpha*classEst
        aggError = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1)))
        errorRate = aggError.sum()/m#错误率：误分的个数/全部样例个数

        if errorRate == 0.0:
            break
    return weakClfArr,aggClassEst

def adaClassify(datToClass, classifierArr):
    dataMatrix = mat(datToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix,\
                                 classifierArr[i]['dim'],\
                                 classifierArr[i]['thresh'],\
                                 classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha']*classEst
    return sign(aggClassEst)

def plotROC(predStrengths, classLabels):
    import matplotlib.pyplot as plt
    cur = (1.0, 1.0)
    ySum = 0.0
    numPosClas = sum(array(classLabels)==1.0)
    yStep = 1/float(numPosClas)#TP
    xStep = 1/float(len(classLabels)-numPosClas)#FP

    print predStrengths
    sortedIndicies = predStrengths.argsort()
    print sortedIndicies
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0; delY = yStep
        else:
            delX = xStep; delY = 0;
            ySum += cur[1]
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY], c='b')
        cur = (cur[0]-delX,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('ROC curve for AdaBoost Horse Colic Detection System')
    ax.axis([0,1,0,1])
    plt.show()
    print ("the Area Under the Curve is: %f" % float(ySum*xStep))

if __name__ == '__main__':
    dataMat, labelMat = loadDataSet('horseColicTraining2.txt')
    weakClfArr,aggClassEst = adaBoostTrainDS(dataMat,labelMat,100)
    testDataMat, testLabelMat = loadDataSet('horseColicTest2.txt')
    pred = adaClassify(testDataMat,weakClfArr)
    result = multiply(pred != mat(testLabelMat).T,ones(shape(testLabelMat)))
    print result.sum()/float(len(testLabelMat))
    plotROC(aggClassEst.T,labelMat)

