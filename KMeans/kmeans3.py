#!/bin/usr/python3
import numpy as np

# 相似度计算函数:使用欧几里得距离[计算数据和簇中心之间的相似度]
def distEucl(vectA, vectB):
    # 欧几里得距离
    return np.sqrt(np.sum(np.power(vectA-vectB, 2)))

def randCent(dataset, k):
    #簇中心初始化方法:随机生成
    _, n = np.shape(dataset)
    # 初始化的簇中心列表
    centroids = np.mat(np.zeros((k, n)))
    for j in range(n):
        # 获取第j个特征的取值范围
        minJ = min(dataset[:, j])
        maxJ = max(dataset[:, j])
        Jrange = float(maxJ - minJ)
        # 在第j个特征上,随机生成簇列表大小的随机值
        centroids[:, j] = minJ + Jrange * np.random.rand(k, 1)
    return centroids

def randPick(dataset, k):
    # 从数据集中随机选择k个样本记录作为簇中心
    m, _ = np.shape(dataset)
    # 数据集打乱
    randIdx = np.random.permutation(m)
    dataset = dataset[randIdx]
    # 随机选择k个记录
    centroids = dataset[:k, :]
    return centroids

## KMeans流程: 随机确定k个初始点作为簇中心; 为每个数据分配簇[计算每条数据和簇中心的相似度,分配到最相似的簇上]; 
# 根据簇中的数据点对每个簇中心进行更新 
# KMeans可能会陷入局部最优解

## 伪代码: 
# 创建k个点作为起始质心;
# 当任意一个点的簇分配结果发生改变时:
#   对数据集中的每个数据点:
#       对每个质心: 
#           计算质心和当前数据点的相似度 
#       将数据点分配到最近的质心所代表的簇上 
#   对于每个簇,计算簇中所有点的均值,并将均值作为新的簇中心[质心]
def kmeans(dataset, k, calSimi=distEucl, createCents=randPick):
    m, n = np.shape(dataset)
    centroids = createCents(dataset, k)
    # 保留每条数据的簇分配结果: 簇ID, 相似度
    clusterAssment = np.mat(np.zeros((m, 2)))
    changed = True
    while changed:
        changed = False
        # 数据集进行簇分配
        for i in range(m):
            minDist = np.inf; minIdx = -1
            # 计算里当前数据最近的簇
            for j in range(k):
                distJI = calSimi(centroids[j, :], dataset[i, :])
                if distJI < minDist:
                    minDist = distJI; minIdx = j
            # 判断当前数据点的簇的分配情况是否发生改变, 进而确定算法是否继续
            if clusterAssment[i, 0] != minIdx:
                changed = True
            # 记录当前数据点的簇分配情况
            clusterAssment[i, :] = minIdx, minDist**2
        # 簇更新
        for cent in range(k):
            # 找到属于当前簇的所有数据点
            ptsInCurCluster = dataset[np.nonzero(clusterAssment[:, 0] == cent)[0], :]
            # 计算当前簇中数据点的均值,作为新的簇中心[质心]
            centroids[cent, :] = np.mean(ptsInCurCluster, axis=0)
    # 当收敛时,返回最后的质心以及分配结果
    return centroids, clusterAssment

# KMeans学习经常会陷入到局部最优解
# 二分kmeans可以避免这个问题.
# 首先将所有点作为一个簇,然后将簇一分为二;之后选择其中一个簇继续进行划分,选择哪一个簇进行划分取决于对其划分是否可以最大程度降低SSE;
# 基于SSE的划分过程不断反复,直到得到指定簇的数目为止.[SSE: sum of Square Errors]
## 伪代码: 
# 将所有数据点看成一个簇
# 当簇数目小于k时: 
#   对于每个簇: 
#       计算总误差 
#       在给定的簇上进行KMeans聚类(k=2) 
#       计算将该簇一分为二之后的总误差 
#   选择使得误差最小的拿个簇进行划分操作.


def biKMeans(dataset, k, calSimi=distEucl, createCents=randCent):
    m, _ = np.shape(dataset)
    clusterAssment = np.mat(np.zeros((m, 2)))
    # 将所有数据点看做一个簇
    centroid0 = np.mean(dataset, axis=0).tolist()
    centroids = [centroid0]
    # 对数据进行簇分配
    for j in range(m):
        clusterAssment[j, 1] = calSimi(centroid0, dataset[j, :])
    # 当
    while(len(centroids) < k):
        lowestSSE = np.inf       
        # 遍历每个簇并尝试划分,选择SSE最小的那个簇
        for cent in range(len(centroids)):
            # 选择当前簇的数据点集
            ptsInCurClust = dataset[np.nonzero(clusterAssment[:, 0] == cent)[0], :]
            # 尝试对当前数据点集进行kmeans聚类[k=2]
            centMat, splitClustAss = kmeans(ptsInCurClust, 2, calSimi, createCents)
            # 计算划分后的误差SSE
            sseSplit = np.sum(splitClustAss[:, 1])
            # 计算没有进行划分的其他簇的误差和
            sseNotSplit = np.sum(clusterAssment[np.nonzero(clusterAssment[:, 0] != cent)[0], 1])
            print ("sseSplit, and notSplit : %d, %d" % (sseSplit, sseNotSplit))
            # 计算划分后误差和最小的簇, 划分后的簇分配情况, 划分后的簇中心列表
            if (sseNotSplit + sseSplit) < lowestSSE:
                lowestSSE = sseNotSplit + sseSplit
                bestCentToSplit = cent # 对当前簇一分为二后的两个簇中心列表
                bestClustAss = splitClustAss.copy()# 划分后的簇数据集分配情况
                bestCents = centMat
        print ("the bestCentToSplit is: %d" % bestCentToSplit)
        print ("the len of bestClustAss is: %d" % len(bestClustAss))
        # 对最佳划分的簇数据集分配情况进行更新: 原来就是0 和 1, 更新到全局簇上.
        # 0: 对应最佳划分簇
        bestClustAss[np.nonzero(bestClustAss[:, 0] == 0)[0], 0] = bestCentToSplit 
        # 1: 对应新加的簇,给定一个标号.
        bestClustAss[np.nonzero(bestClustAss[:, 0] == 1)[0], 0] = len(centroids)
        # 对划分的簇对应的质心进行更新
        centroids[bestCentToSplit] = bestCents[0, :]
        # 将新簇对应的质心,添加到簇质心列表中
        centroids.append(bestCents[1, :])
        # 更新原来的簇分配结果
        clusterAssment[np.nonzero(clusterAssment[:, 0] == bestCentToSplit)[0], :] = bestClustAss
    
    return centroids, clusterAssment

def loadDataSet(filename):
	dataMat = []
	with open(filename) as fr:
		for line in fr.readlines():
			curLine = line.strip().split('\t')
			fltLine = list(map(float, curLine))
			dataMat.append(fltLine)
	return dataMat

if __name__ == '__main__':
    dataSet = np.mat(loadDataSet('./data/testSet.txt'))
    centList, clusterAssments = biKMeans(dataSet, 3)
    print ("#"*64)
    # print centList
    print ('#'*64)
    print (clusterAssments)
