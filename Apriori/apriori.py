#coding:utf8
"""
关联规则：
1. 求解频繁项集；
2. 给予求解的频繁项集挖掘满足最小置信度的关联规则。
"""
def loadDataSet():
	return [[1,3,4],[2,3,5],[1,2,3,5],[2,5]]

def createC1(dataSet):
	"""
	生成大小为1的所有候选项集的一个不变集合；
	:param dataSet: 初始数据集；
	:return: C1项集；
	"""
	C1 = []
	for transaction in dataSet:
		for item in transaction:
			if [item] not in C1:
				C1.append([item])
	C1.sort()
	return map(frozenset, C1) # frozenset：集合的一种类型，创建后不能修改；frozenset可以用作字典的键值；

def scanDataSet(dataSet, Ck, minSupport):
	"""
	通过最小支持度筛选候选项集；
	:param dataSet: 数据集
	:param Ck: 候选项集；
	:param minSupport: 最小支持度；
	:return: 满足最小支持度的频繁项集列表 和 一个包含支持度值的字典；
	"""
	ssCnt = {} # 剪枝为候选项集元素；
	for tid in dataSet: # 扫描数据集
		for can in Ck: # 候选项集
			if can.issubset(tid): # 对每个候选项集元素 统计出现次数；方便计算支持度；
				if not ssCnt.has_key(can):
					ssCnt[can] = 1
				else:
					ssCnt[can] += 1
	numItems = float(len(dataSet)) # 数据集长度
	retList = [] # 筛选后的频繁项集列表；
	supportData = {} # 每个频繁项集的支持度字典；
	for key in ssCnt: # 出现次数字典；
		support = ssCnt[key]/numItems # 项集元素支持度；
		if support >= minSupport: # 大于最小支持度--> 保留；
			retList.insert(0, key)
		supportData[key] = support # 候选项集支持度字典；
	return retList, supportData # 筛选后的频繁项集列表；候选项集支持度字典；

def aprioriGen(Lk, k):
	"""
	通过频繁项集列表生成长度为k的 候选项集Ck；基于上步工作；不断迭代过程：
	列表中两个元素之间两两合并；---减少无效的合并：只选择前k-2个元素相同的列表项进行合并运算；
	:param Lk: LkSub频繁项集列表;元素长度为(k-1)
	:param k: 候选项集元素的长度；
	:return: Ck 元素长度为k的候选项集；
	"""
	retList = []
	lenLk = len(Lk)
	for i in range(lenLk): # 遍历k-1的频繁项集列表；
		for j in range(i+1, lenLk): # eg:Lk = [[0，1],[1,2],[0，2]]，生成频繁3项集
			L1 = list(Lk[i])[:k-2]; L2 = list(Lk[j])[:k-2] # 列表中两个元素的前k-2个元素
			L1.sort(); L2.sort() # 排序
			if L1 == L2: #前k-2个元素相同，其对应的列表项进行并集运算，并将运算结果添加到k频繁项集列表中；
				retList.append(Lk[i] | Lk[j])
	return retList

def LkFreqIS(LkSub):
	retList = []
	lenLkSub = len(LkSub)
	curK = len(LkSub[0])
	for i in range(lenLkSub):
		for j in range(i+1, lenLkSub):
			L1 = list(LkSub[i])[:curK-1]; L2 = list(LkSub[j])[:curK-1]
			L1.sort(); L2.sort()
			if L1 == L2:
				retList.append(LkSub[i] | LkSub[j])
	return retList

def apriori(dataSet, minSupport=0.5):
	"""
	Apriori算法，用于生成满足条件的所有频繁项集；频繁1项集，2项集，3项集...
	:param dataSet: 数据集；
	:param minSupport: 最小支持度；
	:return: 所有的频繁项集， 及其支持度字典；
	"""
	C1 = createC1(dataSet) # 初始候选项集；
	D = map(set, dataSet)
	L1, supportData = scanDataSet(D, C1, minSupport) # 筛选后的L1频繁项集；
	L = [L1]
	k = 2 # 求频繁2项集

	# 基于上次运算求得的k-1频繁项集，求k频繁项集；直到项集上次求得频繁项集长度为0；
	# 因为由k-1频繁项集求 k频繁项集时，给予k-1的前k-2个元素相同，才会求并运算；---最后，肯定会终止；
	while(len(L[k-2])>0):
		Ck = aprioriGen(L[k-2], k)
		Lk, supK = scanDataSet(D, Ck, minSupport)
		supportData.update(supK)
		L.append(Lk)
		k += 1
	return L, supportData # 最后的项集一定有一项元素为 空[]


if __name__ == '__main__':
	dataSet = loadDataSet()
	C1 = createC1(dataSet)
	print C1
	retList, supportData = apriori(dataSet, 0.5)
	print retList
	print ("#"*64)
	print supportData

