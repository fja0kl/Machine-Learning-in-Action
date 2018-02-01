#coding:utf8
from numpy import *

def loadDataSet(filename):
	dataMat = []
	with open(filename) as fr:
		for line in fr.readlines():
			curLine = line.strip().split('\t')
			fltLine = map(float, curLine)
			dataMat.append(fltLine)
	return dataMat

def distEclud(vecA, vecB):
	"""
	距离计算函数--欧式距离
	:param vecA: 向量A
	:param vecB: 向量B
	:return: 两个向量之间的距离；
	"""
	return sqrt(sum(power(vecA-vecB, 2)))

# 初始簇中心-质心；不一定在数据集中，但是质心的各个特征取值一定在 特征取值范围内；
# 这和我的理解不同；


# my solution
def myRandCent(dataSet, k):
	"""
	在数据集中随机选择k个样本点作为初始簇质心；
	:param dataSet: 
	:param k: 
	:return: k个初始簇中心；
	"""
	m = shape(dataSet)[0]
	index = arange(m)
	random.shuffle(index)
	randInd = index[:k]
	centroids = dataSet[randInd]
	return centroids

def randCent(dataSet, k):
	"""
	在给定数据集上构建一个k个随机质心的集合
	:param dataSet: 数据集
	:param k: k值；
	:return: 随机挑选的k个质心集合
	"""
	n = shape(dataSet)[1]
	centroids = mat(zeros((k, n))) # k个质心的集合
	for j in range(n):
		minJ = min(dataSet[:, j])
		rangeJ = float(max(dataSet[:, j]) - minJ)
		centroids[:, j] = minJ + rangeJ*random.rand(k, 1) # rand(k, 1):k是形状参数；k个0-1之间的小数；
	return centroids

# 容易陷入 局部最小值；而不是全局最小值；
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
	"""
	kMeans聚类方法
	:param dataSet: 数据集 
	:param k: 簇的个数
	:param distMeas: 距离测量方法；
	:param createCent: 初始簇质心选择方法；
	:return: 聚类结果，即簇分配结果；
	"""
	m = shape(dataSet)[0]
	clusterAssment = mat(zeros((m, 2))) # 数据集分配结果；下标，距离--可以用来评价聚类的效果；
	centroids = createCent(dataSet, k)
	clusterChanged = True # 终止条件：当所有数据点的簇分配结果不再改变；
	while clusterChanged:
		clusterChanged = False
		for i in range(m): # 遍历数据集中的每个数据点；
			minDist = inf; minIndex = -1
			for j in range(k): # 寻找最近的质心；
				distJI = distMeas(centroids[j, :], dataSet[i, :])
				if distJI < minDist:
					minDist = distJI; minIndex = j
			if clusterAssment[i, 0] != minIndex: # 判断簇分配结果是否需要改变
				clusterChanged = True
			clusterAssment[i, :] = minIndex, minDist**2 # 分配簇：下标，距离平方；
		print centroids # 输出质心变化过程
		for cent in range(k): # 更新质心位置；取分配后簇的均值---数组筛选；
			ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]# 取行方向
			centroids[cent,:] = mean(ptsInClust, axis=0) # 更新质心位置；
	return centroids, clusterAssment

# 二分K-均值算法
def biKmeans(dataSet, k, distMeas=distEclud):
	"""
	
	:param dataSet: 数据集
	:param k: k值
	:param distMeas: 距离计算方法 
	:return: 簇划分结果；簇质心；
	"""
	m = shape(dataSet)[0]
	clusterAssment = mat(zeros((m, 2)))
	centroid0 = mean(dataSet, axis=0).tolist()[0] # 将所有点看做一个簇
	centList = [centroid0]
	for j in range(m): # 记录初始分配情况
		clusterAssment[j, 1] = distMeas(mat(centroid0), dataSet[j, :])**2
	while (len(centList) < k): # 簇数目小于k时；
		lowestSSE = inf
		for i in range(len(centList)): # 对于每个簇
			ptsInCurrCluster = \
				dataSet[nonzero(clusterAssment[:, 0].A == i)[0], :] # 筛选得到分配在当前簇上的数据记录
			centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas) # 在给定簇上进行K-均值聚类（k=2）
			sseSplit = sum(splitClustAss[:, 1]) # 计算将簇一分为二之后的总误差；
			sseNotSplit = \
				sum(clusterAssment[nonzero(clusterAssment[:, 0].A == i)[0], 1]) # 划分之前的总误差；
			print ("sseSplit, and notSplit : %d, %d" % (sseSplit, sseNotSplit))
			if (sseSplit + sseNotSplit) < lowestSSE: # find 误差最小的簇的划分方式
				bestCentToSplit = i
				bestNewCents = centroidMat # 2*2
				bestClustAss = splitClustAss.copy()
				lowestSSE = sseSplit + sseNotSplit
		# 更新簇的分配结果；只有0 和 1---二分
		bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)
		bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
		print ("the bestCentToSplit is: %d" % bestCentToSplit)
		print ("the len of bestClustAss is: %d" % len(bestClustAss))
		centList[bestCentToSplit] = bestNewCents[0, :] # 使用最佳划分簇的新簇中心更新原始簇中心
		centList.append(bestNewCents[1, :]) # 添加新的划分簇
		# 更新划分簇结果；
		clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss
	return centList, clusterAssment


if __name__ == '__main__':
	dataSet = mat(loadDataSet('testSet.txt'))
	centList, clusterAssments = biKmeans(dataSet, 3)
	print ("#"*64)
	print centList
	print ('#'*64)
	print clusterAssments

