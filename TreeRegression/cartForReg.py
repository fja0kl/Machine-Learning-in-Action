#!/bin/usr/python3
import numpy as np

# CART分类回归树:既可以用于分类,也可以用来回归.主要讲用于回归.但CART树一定是二叉树[无论是用户分类还是用户回归].

# 根据回归树叶子的类型不同,可以分为回归树和模型树: 
# 回归树叶子节点是该点数据子集的平均值; 
# 模型树叶子节点是一个线性回归参数列表,预测时需要进行一步计算,计算结果

# 原来的ID3,C4.5,或者是CART用于分类时,最佳特征选择参数是信息增益,信息增益比率.gini指数;
# CART回归树最佳特征选择使用的指标是MSE,平方误差和[不取平均],误差越小越好
# [符合常识,回归问题多用MSE,MAE作为损失函数].

def binSplitMethod(dataset, featIdx, value):
    mat0 = dataset[np.nonzero(dataset[:, featIdx] > value)[0], :]
    mat1 = dataset[np.nonzero(dataset[:, featIdx] <= value)[0], :]

    return mat0, mat1

def regLeaf(dataset):
    # 叶子节点:取平均值
    return np.mean(dataset[:,-1])

def linearSolve(dataset):
    # 模型树叶子节点---线性方程
    m, n = np.shape(dataset)
    # X 中包含一个全1列. 对应W参数中的偏置参数b
    X = np.mat(np.ones((m, n)))
    y = np.mat(np.ones((m, 1)))
    X[:, 1:n] = dataset[:, 0:n-1]; y = dataset[:, -1]
    # 直接用正规方程,计算线性回归参数
    xTx = X.T * X
    #判断是否可逆[奇异矩阵]
    if np.linalg.det(xTx) == 0.0:
        raise NameError("奇异矩阵, 矩阵不可逆,尝试修改参数")
    W = xTx.I * (X.T * y)

    return X, y, W

def modelLeaf(dataset):
    X, y, W = linearSolve(dataset)
    return W

def regErr(dataset):
    return np.var(dataset[:,-1]) * np.shape(dataset)[0]

def modelErr(dataset):
    # 模型树的误差函数
    X, y, W = linearSolve(dataset)
    yHat = X*W

    return sum(np.power(yHat-y, 2))

def chooseBestSplit(dataset, leafType=regLeaf, errorType=regErr, ops=(1,4)):
    """
	选择合适的数据的最佳二元切分方法，当chooseBestSplit函数确定不再对数据进行切分时，生成叶子节点；
	找到“好”的切分方式时，返回切分特征编号和切分特征值。
	:param leafType: 生成叶子节点；当chooseBestSplit函数确定不再对数据进行切分时，生成叶子节点；
	:param errorType: 误差函数计算方式---切分方式的评价指标；
	:param ops: 其他参数，用于控制函数的停止时机：tolS 容许的误差下降值， tolN 切分的最少样本数；
	:return: 找不到‘好’的切分，返回叶子节点；找到“好”的切分，返回 切分特征编号 和 切分特征值；
	"""
    tolS = ops[0]; tolN = ops[1]
    # 如果,节点上所有值都相等,直接返回叶子节点
    y = dataset[:, -1].T.tolist()[0]
    if len(set(y)) == 1:
        return None, leafType(dataset)
    # 其他情况,找出最佳特征, 特征值
    m, n = np.shape(dataset)
    S = errorType(dataset)
    bestS = np.inf; bestFeatIdx = -1; bestSplitVal = 0
    for featIdx in range(n-1):
        uniqueValues = set(dataset[:, featIdx].T.tolist()[0])
        for value in uniqueValues:
            lSet, rSet = binSplitMethod(dataset, featIdx, value)
            # 判断节点内样本数目
            if(np.shape(lSet)[0] < tolN or np.shape(rSet)[0] < tolN):
                continue
            newS = errorType(lSet) + errorType(rSet)
            if newS < bestS:
                bestS = newS
                bestFeatIdx = featIdx
                bestSplitVal = value
    # 判断误差更新值
    if (S - bestS) < tolS:
        return None, leafType(dataset)
    # 根据找到的最好的划分方式进行划分；评估划分后的数据集的样本容量大小；太小，直接退出；合适-->返回划分特征编号和特征值；
    mat0, mat1 = binSplitMethod(dataset, bestFeatIdx, bestSplitVal)
    if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN): # 划分后每个数据集的样本容量 评估
        return None, leafType(dataset)
    return bestFeatIdx, bestSplitVal

def createTree(dataset,leafType=regLeaf, errorType=regErr,ops=(1,4)):
    bestFeatIdx, bestValue = chooseBestSplit(dataset,leafType,errorType,ops)
    # 如果标签取值都相同；划分后数据集样本数太少；划分后误差下降太小, 直接节点值
    if bestFeatIdx == None:#bestValue 值是叶子节点的值
        return bestValue
    # 递归创建树
    tree = {}
    tree['spIdx'] = bestFeatIdx
    tree['spVal'] = bestValue
    lSet, rSet = binSplitMethod(dataset,bestFeatIdx, bestValue)
    tree['left'] = createTree(lSet,leafType,errorType,ops)
    tree['right'] = createTree(rSet,leafType, errorType, ops)

    return tree

def isTree(tree):
    return isinstance(tree, dict)

def getMean(tree):
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])
    
    return (tree['left'] + tree['right'])/2.0

#树剪枝：
#伪代码
#基于已有的树切分测试数据：
#     如果存在任一子集是一棵树，则在该子集上递归剪枝;
#     计算将当前两个[叶节点]合并后的误差
#     计算不合并的误差
#     如果合并的误差会降低的话，就将叶节点合并

def prune(tree, testdata):
    # 测试数据为空, 进行塌陷处理
    if np.shape(testdata)[0] == 0:
        return getMean(tree)
    if(isTree(tree['left']) or isTree(tree['right'])):
        lSet, rSet = binSplitMethod(testdata, tree['spIdx'], tree['spVal'])
    # 在子树上进行递归剪枝处理
    if(isTree(tree['left'])): tree['left'] = prune(tree['left'], lSet)
    if(isTree(tree['right'])): tree['right'] = prune(tree['right'], rSet)
    # 如果到叶子节点,尝试合并剪枝
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitMethod(testdata, tree['spIdx'], tree['spVal'])
        # 未剪枝误差
        errNoMerge = sum(np.power(tree['left']-lSet[:,-1], 2)) + sum(np.power(tree['right']-rSet[:,-1], 2))
        # 剪枝
        treeMean = (tree['left'] + tree['right'])/2.0
        errMerge = np.power(testdata[:,-1]-treeMean, 2)
        #返回合并后的估计值
        if errMerge < errNoMerge:
            print("Merging...")
            return treeMean 
        else:# 不用合并
            return tree
    else:
        return tree

# 回归树预测

def regPredBase(leaf, inDat):
	#回归树，返回叶子节点表示的浮点数
	return float(leaf)

def modelPredBase(ws, inDat):
    # 模型树回归预测；inDat中没有标签
    n = np.shape(inDat)[1]
    X = np.mat(np.ones((1, n+1)))
    X[:, 1:n+1] = inDat # 格式化，第一列为1

    return float(X * ws)

# 单条数据
def treeForeCast(tree, inData, modelEval=regPredBase):
	# 自顶向下遍历整棵树，直到命中 叶子节点为止；调用modelEval函数，进行预测；
	if not isTree(tree): # 叶子节点
		return modelEval(tree, inData)
    # 寻找子树中的叶子节点；找到进行评估，预测；
	if inData[tree['spIdx']] > tree['spVal']: # 左子树
		if isTree(tree['left']):# 如果子树还是是树,递归判断
			return treeForeCast(tree['left'], inData, modelEval)
		else:# 否则,直接返回节点值
			return modelEval(tree['left'], inData)
	else:#右子树
		if isTree(tree['right']):
			return treeForeCast(tree['right'],inData, modelEval)
		else:
			return modelEval(tree['right'], inData)

def dataPredict(tree, testData, modelEval=regPredBase):
	"""
	在整个测试集上进行评估，预测；
	:param tree: 回归树模型；
	:param testData: 测试集
	:param modelEval: 评估函数
	:return: 测试集上的评估值向量；
	"""
	m = len(testData)
	yHat = np.mat(np.zeros((m, 1)))
	for i in range(m):
		yHat[i, 0] = treeForeCast(tree, np.mat(testData[i]), modelEval)
	return yHat

# 数据读取
def loadDataSet(filename):
	dataMat = []
	with open(filename) as fr:
		for line in fr.readlines():
			curLine = line.strip().split('\t')
			fltLine = list(map(float, curLine)) # 将curLine列表中的元素，变成float类型；返回list
            # map(function, sequence[, sequence, ...]) -> list 将function应用到可迭代对象的组成元素上，返回list
			dataMat.append(fltLine)
	return dataMat

def testTreeEval():
	# 通过计算预测结果与标签值之间的相关系数，来评价模型的好坏；
    trainMat = np.mat(loadDataSet('./data/bikeSpeedVsIq_train.txt'))
    testMat = np.mat(loadDataSet('./data/bikeSpeedVsIq_test.txt'))
	
    regTree = createTree(trainMat, ops=(1,20))
    yHat = dataPredict(regTree, testMat[:,0])
    cor = np.corrcoef(yHat, testMat[:, 1],rowvar=0)[0,1]
    print (cor)

    modelTree = createTree(trainMat, leafType=modelLeaf, errorType=modelErr, ops=(1,20))
    yHat_model = dataPredict(modelTree, testMat[:,0], modelPredBase)
    modelCorr = np.corrcoef(yHat_model, testMat[:,1],rowvar=0)[0,1]
    print (modelCorr)


if __name__ == '__main__':
	testTreeEval()
