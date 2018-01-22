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
    cur = (1.0, 1.0)#起点
    ySum = 0.0
    numPosClas = sum(array(classLabels)==1.0)#正类数目
    yStep = 1/float(numPosClas)#TPR单维刻度
    xStep = 1/float(len(classLabels)-numPosClas)#FPR单维刻度

    # print predStrengths
    sortedIndicies = predStrengths.argsort()#对预测结果的得分进行升序排序
    print type(sortedIndicies)
    #绘图
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)

    print shape(sortedIndicies.tolist())#matrix to list : 多维的

    for index in sortedIndicies.tolist()[0]:#【0】：列表
        if classLabels[index] == 1.0:#真 正类：TPR情况
            delX = 0; delY = yStep
        else:
            delX = xStep; delY = 0
            ySum += cur[1]
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY], c='b')#画点：从（1,1）开始
        cur = (cur[0]-delX,cur[1]-delY)#变换当前点
    ax.plot([0,1],[0,1],'c--')#对角线：（0,0）到（1,1）；；一条虚线--dashed
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')#轴标签
    plt.title('ROC curve for AdaBoost Horse Colic Detection System')#图标签
    ax.axis([0,1,0,1])#x轴、y轴刻度范围
    plt.show()#图显示
    print ("the Area Under the Curve is: %f" % float(ySum*xStep))#AUC：ROC曲线下的面积；；；计算方法，像微积分

if __name__ == '__main__':
    dataMat, labelMat = loadDataSet('horseColicTraining2.txt')
    weakClfArr,aggClassEst = adaBoostTrainDS(dataMat,labelMat,100)
    # testDataMat, testLabelMat = loadDataSet('horseColicTest2.txt')
    # weakClfArr, aggClassEst = adaBoostTrainDS(testDataMat, testLabelMat, 100)
    # pred = adaClassify(testDataMat,weakClfArr)
    # result = multiply(pred != mat(testLabelMat).T,ones(shape(testLabelMat)))
    # print result.sum()/float(len(testLabelMat))
    plotROC(aggClassEst.T,labelMat)

