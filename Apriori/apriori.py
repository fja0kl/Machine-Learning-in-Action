#coding:utf8
"""
关联规则：
1. 求解频繁项集；
2. 给予求解的频繁项集挖掘满足最小置信度的关联规则。
"""
def loadDataSet():
	return [[1,3,4,],[2,3,6],[1,2,3,5,6],[2,5,1,6]]

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

def generateRules(L, supportData, minConf=0.7):
	"""
	生成关联规则
	:param L: 频繁项集；
	:param supportData:包含频繁项集支持度的字典； 
	:param minConf: 最小置信度
	:return: 包含可信度的关联规则列表；
	"""
	bigRulesList = []
	for i in range(1, len(L)): # 过滤掉1频繁项集；
		for freqSet in L[i]:
			H1 = [frozenset([item]) for item in freqSet] # 频繁项集的单个元素列表；
			rulesFromConseq(freqSet, H1, supportData, bigRulesList, minConf)
			# if (i > 1):
			# 	H1 = calcConf(freqSet, H1, supportData, bigRulesList, minConf)
			# 	rulesFromConseq(freqSet, H1, supportData, bigRulesList, minConf)
			# else: # i = 1；频繁二项集,不涉及右边元素再次组合;直接计算置信度；
			# 	calcConf(freqSet, H1, supportData, bigRulesList, minConf)
	return bigRulesList

def calcConf(freqSet, H, supportData, brl, minConf=0.7):
	"""
	对候选规则进行评估，筛选满足最小置信度的规则；
	:param freqSet: 一个频繁项集的元素；
	:param H: 频繁项集的单个元素列表；
	:param supportData: 频繁项集支持度字典
	:param brl: 满足条件的关联规则
	:param minConf: 最小置信度；
	:return: 
	"""
	prunedH = [] # 可以出现在关联规则右边的元素列表；
	for conseq in H: # 频繁项集中的单个元素；
		# 列举所有可能的关联规则，计算其置信度；
		# (freqSet-conseq) --> conseq;   freqSet:频繁项集（全集），我们依据频繁项集
		# 来列举出所有可能的关联规则，计算置信度，从而确定真正的满足条件的关联规则；
		conf = supportData[freqSet]/supportData[freqSet-conseq]
		if conf >= minConf:
			print freqSet-conseq,'-->',conseq,'conf:',conf
			brl.append((freqSet-conseq, conseq, conf)) # brl满足条件的关联规则列表；
			prunedH.append(conseq)
	return prunedH

def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
	"""
	生成候选规则集合
	:param freqSet:频繁项集； 
	:param H: 可以出现在规则右部的元素列表；
	:param supportData: 支持度
	:param brl: 通过检查的bigRulesList
	:param minConf: 最小置信度；
	:return: 
	"""
	m = len(H[0])
	if (len(freqSet) > m): # 判断当前频繁项集能否移除 大小为m的 子集；
		prunedTails = calcConf(freqSet, H, supportData, brl, minConf)
		if(len(prunedTails) > 1):# 满足条件的规则 多余一条；判断是否可以再次组合；
			nextTails = aprioriGen(H, m+1) # 由当前频繁项集单元素列表，生成所有的m+1元素列表（可能出现在规则右边）；
			print(nextTails)
			rulesFromConseq(freqSet,nextTails,supportData,brl,minConf)
		# 筛选后，满足条件的可以出现在规则右边的 元素列表；
		# 通过下面的筛选，实现：
		# 如果某条规则并不满足最小可信度要求，那么该规则的所有子集也不会满足最小可信度要求；
		# 我们以右边的结论为关注点；其满足条件的关联规则的子集的生成过程，
		# 就是以当前筛选过的满足条件的关联规则的右边为基础，生成高一层的子集；不断递归实现；
		# 这样，可以只计算：
		# 满足最小可信度要求的，所有子集也满足最小置信度；从而，生成所有的关联规则；
		# yeah
		

if __name__ == '__main__':
	dataSet = loadDataSet()
	C1 = createC1(dataSet)
	print C1
	print(type(C1))
	retList, supportData = apriori(dataSet, 0.5)
	# print retList
	# print ("#"*64)
	# print supportData
	rules = generateRules(retList, supportData, minConf=0.6)
	# print ('#'*64)
	# print rules

