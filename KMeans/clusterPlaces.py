#coding:utf8
from numpy import *
import matplotlib
import matplotlib.pyplot as plt

def distOCalc(vecA, vecB):
	return sqrt(sum(power(vecA-vecB, 2)))

def randCent(dataSet, k):
	n = shape(dataSet)[1]
	centroids = mat(zeros((k, n)))
	for j in range(n):
		minJ = min(dataSet[:, j])
		rangeJ = max(dataSet[:, j]) - minJ
		centroids[:, j] = minJ + rangeJ*random.rand(k, 1)
		print minJ + rangeJ*random.rand(k, 1)
	return centroids

def myRandCent(dataSet, k):
	m = shape(dataSet)[0]
	index = arange(m)
	random.shuffle(index)
	randInd = index[:k]
	centroids = dataSet[randInd]
	return centroids

def kMeans(dataSet, k, distMeas=distOCalc, createCent=myRandCent):
	m = shape(dataSet)[0]
	clusterAssment = mat(zeros((m, 2)))
	centroids = createCent(dataSet, k)
	clusterChanged = True
	while clusterChanged:
		clusterChanged = False
		for i in range(m):
			minDist = inf; minIndex = -1
			for j in range(k):
				distJI = distMeas(centroids[j], dataSet[i])
				if distJI < minDist:
					minDist = distJI
					minIndex = j
			if clusterAssment[i, 0] != minIndex:
				clusterChanged = True
			clusterAssment[i, :] = minIndex, minDist**2
		print centroids
		for cent in range(k):
			clust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]
			centroids[cent, :] = mean(clust, axis=0)
	return centroids, clusterAssment

def binKmeans(dataSet, k, distMeas=distOCalc):
	m = shape(dataSet)[0]
	clusterAssment = mat(zeros((m, 2)))
	centroid0 = mean(dataSet, axis=0).tolist()[0]
	centList = [centroid0]
	for j in range(m):
		clusterAssment[j, 1] = distMeas(mat(centroid0), dataSet[j, :])**2
	while (len(centList)<k):
		lowestSSE = inf
		for i in range(len(centList)):
			clust = dataSet[nonzero(clusterAssment[:, 0].A==i)[0], :]
			centroidMat, splitClustAss = kMeans(clust, 2, distMeas)
			sseSplit = sum(splitClustAss[:, 1])
			sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0]==i)[0], 1])
			if (sseSplit + sseNotSplit) < lowestSSE:
				bestCentToSplit = i
				bestNewCents = centroidMat
				bestClustAss = splitClustAss.copy()
				lowestSSE = sseSplit + sseNotSplit
		bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)
		bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
		centList[bestCentToSplit] = bestNewCents[0, :]  # 使用最佳划分簇的新簇中心更新原始簇中心
		centList.append(bestNewCents[1, :])  # 添加新的划分簇
		clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss
	return centList, clusterAssment


def distSLC(vecA, vecB):
	a = sin(vecA[0, 1]*pi/180) * sin(vecB[0, 1]*pi/180)
	b = cos(vecA[0, 1]*pi/180) * cos(vecB[0, 1]*pi/180) * \
			cos(pi*(vecB[0, 0]-vecA[0, 0])/180)
	return arccos(a + b) * 6371.0

def clusterClubs(numClust=5):
	datList = []
	for line in open('places.txt').readlines():
		lineArr = line.split('\t')
		datList.append([float(lineArr[4]), float(lineArr[3])])
	datMat = mat(datList)
	myCentroids, clustAssing = binKmeans(datMat, numClust, distMeas=distSLC)

	fig = plt.figure()
	rect = [0.1, 0.1, 0.8, 0.8]
	scatterMarkers = ['s','o','^','8','p','d','v','h','>','<']
	axprops = dict(xticks=[], yticks=[])
	ax0 = fig.add_axes(rect, label='ax0', **axprops)
	imgP = plt.imread('Portland.png')
	ax0.imshow(imgP)
	ax1 = fig.add_axes(rect, label='ax1', frameon=False)
	for i in range(numClust):
		ptsInCurrCluster = datMat[nonzero(clustAssing[:, 0].A == i)[0], :]
		markerStyle = scatterMarkers[i % len(scatterMarkers)]
		ax1.scatter(ptsInCurrCluster[:, 0].flatten().A[0], ptsInCurrCluster[:, 1].flatten().A[0],
					marker=markerStyle, s=90)
	# ax1.scatter(myCentroids[:, 0].flatten().A[0], myCentroids[:, 1].flatten().A[0],marker='+',s=300)
	a = []; b = []
	for cent in myCentroids:
		a.append(cent[0,0])
		b.append(cent[0,1])
	ax1.scatter(a, b, marker='+', s=300)
	# ax1.scatter(myCentroids[:,0].flatten().A[0], myCentroids[:,1].flatten().A[0], marker='+', s=300)
	plt.show()

if __name__ == '__main__':
    clusterClubs()
