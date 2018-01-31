#!/bin/usr/env python3
#coding:utf8
from numpy import *

class treeNode():
    def __init__(self, feat, val, right, left):
        featureToSplitOn = feat
        valueOfSplit = val
        rightBranch = right
        leftBranch = left

def loadDataSet(filename):
    dataMat = []
    with open(filename) as fr:
        for line in fr.readlines():
            curLine = line.strip().split('\t')
            fltLine = map(float, curLine) # 将curLine列表中的元素，变成float类型；返回list
            # map(function, sequence[, sequence, ...]) -> list 将function应用到可迭代对象的组成元素上，返回list
            dataMat.append(fltLine)
    return dataMat

def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0], :][0]
    # print (nonzero(dataSet[:,feature] > value))
    mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0], :][0]
    # 解释一下：
    # dataSet[nonzero(dataSet[:,feature] > value)[0], :][0]
    # nonzero(dataSet[:,feature] > value)[0] 返回两个维度上的结果，现在只取行维度上，满足条件的结果；得到满足条件的行坐标
    # 然后依据得到的结果对数据集进行裁剪，划分；得到两个矩阵 a，b
    return mat0, mat1

def regLeaf(dataSet):
    return mean(dataSet[:, -1])

def regErr(dataSet):
    # 总方差作为误差
    return var(dataSet[:, -1]) * shape(dataSet)[0]

def chooseBestSplit(dataSet, leafType=regLeaf, errorType=regErr, ops=(1,4)):
	"""
	选择合适的数据的最佳二元切分方法，当chooseBestSplit函数确定不再对数据进行切分时，生成叶子节点；
	找到“好”的切分方式时，返回切分特征编号和切分特征值。
	:param dataSet: 数据集
	:param leafType: 生成叶子节点；当chooseBestSplit函数确定不再对数据进行切分时，生成叶子节点；
	:param errorType: 误差函数计算方式---切分方式的评价指标；
	:param ops: 其他参数，用于控制函数的停止时机：tolS 容许的误差下降值， tolN 切分的最少样本数；
	:return: 找不到‘好’的切分，返回叶子节点；找到“好”的切分，返回 切分特征编号 和 切分特征值；
	"""
	tolS = ops[0]; tolN = ops[1]
	vals = dataSet[:, -1].T.tolist()[0]
	if len(set(vals)) == 1: # 1.所有值都相等--> 特征编号为None，生成叶子节点（均值），并退出；
		return None, leafType(dataSet)
	m, n = shape(dataSet)
	S = errorType(dataSet) # 计算初始误差
	bestS = inf; bestIndex = 0; bestValue = 0
	# 遍历所有特征及其所有的取值来找到使误差最小化的切分阈值；
	for featIndex in range(n-1): # 遍历所有特征；
		for splitVal in set(dataSet[:, featIndex]): # 遍历该特征的所有可能取值；
			mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
			if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN): # 待切分的数据集的样本数必须 大于tolN（切分最小样本数）
				continue
			newS = errorType(mat0) + errorType(mat1) # 计算划分后的误差，与初始误差值进行比较；find the best
			if newS < bestS:
				bestIndex = featIndex
				bestValue = splitVal
				bestS = newS
	if (S - bestS) < tolS: # 如果划分后误差下降太小，即：小于容许的误差下降值；直接返回，生成叶子节点
		return None, leafType(dataSet)
	# 根据找到的最好的划分方式进行划分；评估划分后的数据集的样本容量大小；太小，直接退出；合适-->返回划分特征编号和特征值；
	mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
	if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN): # 划分后每个数据集的样本容量 评估
		return None, leafType(dataSet)
	return bestIndex, bestValue

# 树的生成算法，对参数ops敏感；
# ops参数用于设定停止条件；
# 如何找到合适的停止条件，it's a question.
def createTree(dataSet, leafType=regLeaf, errorType=regErr, ops=(1,4)):
    """
    回归树，模型树；叶子类型不同，误差计算函数不同，需要其他参数的元组
    :param dataSet:
    :param leafType: 叶子类型；
    :param errorType: 误差计算函数
    :param ops: 树构建过程中需要的其他参数；以元组形式给出；
    :return: 树
    """
    feat, val = chooseBestSplit(dataSet, leafType, errorType, ops)
    if feat == None: # 不能切分（不用切分）时---情况：特征值都相同；划分后数据集样本数太少；划分后误差下降太小；
        return val
	# 递归
    retTree = {}
    retTree['spInd'] = feat #划分特征的下标
    retTree['spVal'] = val #划分特征的分类值
    lSet, rSet = binSplitDataSet(dataSet, feat, val) # 对划分后的子集，继续划分；
    retTree['left'] = createTree(lSet, leafType, errorType, ops)
    retTree['right'] = createTree(rSet, leafType, errorType, ops)
    return retTree


def isTree(obj):
	"""
	判断是否是树；用来寻找叶子节点；
	:param obj: 对象
	:return: bool
	"""
	return isinstance(obj,dict)

def getMean(tree):
	"""
	对树进行塌陷处理（即返回树的平均值）；从上到下遍历树直到叶节点为止，找到两个叶节点，则计算平均值；
	:param tree: 
	:return: 
	"""
	if isTree(tree['left']):
		tree['left'] = getMean(tree['left'])
	if isTree(tree['right']):
		tree['right'] = getMean(tree['right'])
	return (tree['left']+tree['right'])/2.0

def prune(tree, testData):
	if shape(testData)[0] == 0:# 测试集没有数据，进行塌陷处理
		return getMean(tree)
	if (isTree(tree['left']) or isTree(tree['right'])):
		lSet, rSet = binSplitDataSet(testData, tree['spInd'],tree['spVal'])
	# 递归处理子树
	if isTree(tree['left']): tree['left'] = prune(tree['left'],lSet)
	if isTree(tree['right']): tree['right'] = prune(tree['right'], rSet)
	# 到达叶子节点
	if not isTree(tree['left']) and not isTree(tree['right']):
		lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
		#未剪枝误差：
		errorNoMerge = sum(power(lSet[:,-1]-tree['left'], 2)) + \
			sum(power(rSet[:,-1]-tree['right'], 2))
		# 剪枝后误差：
		treeMean = (tree['left']+tree['right'])/2.0 # 取平均值
		errorMerge = sum(power(testData[:,-1] - treeMean, 2))
		if errorMerge < errorNoMerge:# 合并后，误差减小，--合并；merge
			print ("merging")
			return treeMean
		else:# 不合并
			return tree
	else:
		return tree

if __name__ == '__main__':
	myDat2 = loadDataSet('ex2.txt')
	print ('树回归')
	myMat = mat(myDat2)
	tree = createTree(myMat,ops=(1000,4))
	print (tree)
	myTestData = loadDataSet('ex2test.txt')
	myMat2Test = mat(myTestData)
	pruneTree = prune(tree, myMat2Test)
	print (pruneTree)
