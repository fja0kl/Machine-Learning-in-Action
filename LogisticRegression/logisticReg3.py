import numpy as np

def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('./data/testSet.txt','r')
    for line in fr.readlines():
        line = line.strip().split()
        dataMat.append([1.0, float(line[0]), float(line[1])])
        labelMat.append(int(line[2]))
    return dataMat, labelMat

def sigmoid(x):
    return 1/(1 + np.exp(-x))

# 批次梯度下降batch gradient descent;通过迭代次数终止
def gradDescent(data, labels, maxIters=500):
    xMat = np.mat(data); yMat = np.mat(labels).T
    m, n = np.shape(xMat)
    weights = np.random.randn(n, 1)
    alpha = 0.001
    for i in range(maxIters):
        h = sigmoid(xMat * weights)
        error = yMat - h
        weights = weights - alpha * xMat.T * error
    
    return weights

# 随机梯度下降:每次梯度更新只选择一个样本计算梯度,然后更新梯度----noise,波动
def stocGradDecent(data, labels):
    xMat = np.mat(data); yMat = np.mat(labels).T
    alpha = 0.005
    m, n = np.shape(xMat)
    weights = np.random.randn(n, 1) 
    # 随机梯度更新:样本依次选择
    for i in range(m):
        # 计算结果是矩阵形式
        h = sigmoid(xMat[i, :] * weights)
        # 计算结果是矩阵形式
        error = yMat[i] - h
        print(type(error))
        print(type(error.A))#ndarray
        weights = weights - alpha * error.A * xMat[i].T.A#可以广播
    
    return weights

# 随机梯度下降: 样本随机选择,同时进行多次迭代epochs
def stocGradDescent1(data,labels,maxIters=200):
    xMat = np.mat(data); yMat = np.mat(labels).T
    m, n = np.shape(xMat)
    weights = np.random.randn(n ,1)
    for i in range(maxIters):
        dataIdx = [k for k in range(m)]
        for j in range(m):
            # alpha随着迭代次数增加,逐渐减小,收敛更快
            alpha = 4.0/(i+j+1)
            #随机选择一个样本,但是保证不会重复选择相同的样本---在dataIdx中删除选择过得样本
            randIdx = int(np.random.uniform(0, len(dataIdx)))
            h = sigmoid(xMat[randIdx] * weights)
            error = yMat[randIdx] - h
            weights = weights - alpha * error.A * xMat[randIdx].T.A
            #删除使用过的样本
            del dataIdx[randIdx]
    
    return weights

def classify(inputX, weights):
    # inputX类型 np matrix; weights: np matrix
    score = sigmoid(inputX * weights)
    print(score)
    if score.A > 0.5:
        return 1.0
    else:
        return 0.0

if __name__ == '__main__':
    data, labels = loadDataSet()
    w = stocGradDescent1(data, labels)
    print(w.shape)
    a = classify(np.mat([1.0,2.0,-3.1]),w)
    print(a)