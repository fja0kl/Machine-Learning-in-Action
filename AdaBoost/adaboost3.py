#!/bin/usr/python3
import numpy as np

# AdaBoost模型集成方法: 弱分类器采用决策树;而各个决策树之间训练过程是串行训练---重点[关注]前一个分类器错分的数据样本; 
# 最后模型预测时由之前训练的所有弱分类器共同决定---各个分类器的权重不同,
# 权重的计算主要依据是分类错误率越低的分类器其对应权重越大.
# 但决策树只用一个特征进行判断,也就是说决策树只有两层; 
# 
# 关注错分数据怎么实现? 赋权重, 依据是否误分进行计算,以及上次权重结果进行计算,同时保证所有数据记录的权重和为1. 
# 此外,第一次训练时, 所有数据记录的权重系数都相同
# 
# AdaBoost构建:首先需要决策树桩的构建,然后再是基于决策树AdaBoost的构建,最后再依据训练结果对新数据进行分类. 
# 决策树桩的构建:由于只使用一个特征进行判断,所以需要遍历所有特征,然后在特征取值内进行遍历,因为这里使用的是连续数据,
# 所以在数据集特征划分时对两种离散结果都进行尝试[>=, <=], 最后找到分类效果最好的特征,以及对应的决策树.

# 对数值型特征进行分类: 注意这里的分类结果为-1和1
def stumpClassify(dataMatrix, axis, threshVal, sign):
    m, _ = np.shape(dataMatrix)
    result = np.ones((m, 1))
    # 依据sign符合判断,分类方向
    if sign == 'lt':
        result[dataMatrix[:, axis] <= threshVal] = -1
    else:
        result[dataMatrix[:, axis] >= threshVal] = -1
    
    return result

# 决策树桩构建
def buildStump(dataArr, classLabels, D):
    """
    寻找最低错误率的决策树
    :param dataArr: 数据集
    :param classLabels: 分类标签
    :param D: 权重向量【每个记录有一个权重】
    :return: 决策树，最小误差，预测的分类结果
    """
    dataMatrix = np.mat(dataArr); labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(dataMatrix)
    #对于连续数据,确定特征变化步幅;
    numSteps = 10
    # 保存最佳树桩, 以及对应的分类效果,方便后续计算误差
    bestStump = {}; bestEst = np.mat(np.zeros((m, 1)))
    minError = np.inf # 保存最小误差,依据误差确定最佳树桩
    
    for i in range(n):
        # 确定取值范围,进而确定步幅,最后依据计算结果确定,划分边界的取值
        rangeMin = dataMatrix[:, i].min(); rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin)/float(numSteps)
        for j in range(-1, numSteps+1):
            # 确定每次划分时的,划分边界
            for sign in ['lt', 'gt']:
                threshVal = rangeMin + j * stepSize
                # 依据特征,边界,符号尝试划分, 然后计算误差
                predResults = stumpClassify(dataMatrix, i, threshVal, sign)
                # 计算误差, 采用向量计算,更搞笑
                errArr = np.ones((m, 1))
                errArr[predResults == labelMat] = 0 #分类正确,设置为0
                # 误差: 误分样本对应权重系数之和
                weightedError = errArr.T * D

                print ("split: axis %d,thresh %.2f, sign: %s, the weighted error is %.3f" %\
                       (i,threshVal, sign, weightedError))
                #判断,寻找最小误差
                if weightedError < minError:
                    minError = weightedError
                    bestEst = predResults.copy()
                    bestStump['axis'] = i
                    bestStump['threshVal'] = threshVal
                    bestStump['sign'] = sign

    return bestStump, minError, bestEst

# 基于决策树的AdaBoost训练
def trainAdaBoostDS(dataArr, classLabels, numIters=50):
    """
    AdaBoost训练过程
    :param dataArr 训练数据
    :param classLabels 训练数据对应标签
    :param numIters 迭代次数,指定了树桩的上界
    :return 训练的弱分类器列表
    """
    weakClassifiers = []
    m, _ = np.shape(dataArr)
    # D 数据集样本的权重系数表
    D = np.mat(np.ones((m, 1))/m)
    # 累计误差,如果累计误差等于0, 可以提前退出,停止训练
    aggClassEst = np.mat(np.zeros((m, 1)))
    # 训练
    for i in range(numIters):
        # 构建一个最佳树桩
        stump, minErr, predResults = buildStump(dataArr, classLabels, D)
        print('样本权重矩阵D:')
        print(D.T)
        # 依据对训练数据的预测结果,计算分类器权重,然后更新样本权重列表
        alpha = float(0.5 * np.log((1-minErr)/max(minErr, 1e-16)))#1e-16是为了防止0溢出
        # 保存分类器的权重
        stump['alpha'] = alpha
        weakClassifiers.append(stump)
        # 更新D矩阵
        # 计算公式 D（i+1) = D(i)*exp(-alpha*y*h(x))/Z ; Z是归一化系数
        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, predResults) # exp的指数
        ################ multiply:乘法 element-wise逐元素相乘.可以进行广播
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()

        # 计算累计分类误差
        #首先需要计算弱分类器累计分类效果: 分类器加权
        aggClassEst += alpha * predResults
        # 计算强分类器的误差
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m,1)))
        errorRate = aggErrors.sum() / m
        print ("total error: %f" % errorRate)
        # 强分类分类误差为0, 提前退出训练过程
        if errorRate == 0.0:
            break
    return weakClassifiers

def adaBoostClassify(data, classArr):
    dataMatrix = np.mat(data)
    m, _ = np.shape(dataMatrix)
    results = np.mat(np.zeros((m, 1)))
    # 每个弱分类器进行分类, 对结果累计求和
    for model in classArr:
        predVals = stumpClassify(dataMatrix, model['axis'], model['threshVal'], model['sign'])
        results += model['alpha'] * predVals
        print(results)# 累计结果变化
    # 输出累计结果: sign取符号: -1 1
    return np.sign(results)

def loadSimpData():
    dataMat = np.mat([[1.,2.1],
                     [2.,1.1],
                     [1.3,1.],
                     [1., 1.],
                     [2.,1.]])
    classLabels = [1.0,1.0,-1.0,-1.0,1.0]
    return dataMat,classLabels

if __name__ == '__main__':
    dataMat, classLabels = loadSimpData()
    classifierArr = trainAdaBoostDS(dataMat,classLabels,30)
    p = adaBoostClassify([[0,0.],[1.,5.]],classifierArr)
    print(p)
