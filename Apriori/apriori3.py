#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
关联分析:
- 求频繁项集;
- 根据频繁项集结果,计算符合confidence置信度的关联规则,保存,输出

Apriori算法:用来生成频繁项集.
一般情况下,如果要生成频繁项集,我们需要穷举所有情况,然后根据每种情况进行统计计算support支持度,筛选掉不符合条件的频繁项集;
但是,这种方法太过于复杂,时间复杂度太高,如果数据集非常大,时间成指数级增长.所以,出现了apriori原理.
主要原理是:如果一个项集是频繁的,那么它的所有子集也是频繁的;反之,如果一个项集是非频繁的,那么它的所有超集也是非频繁的.
(根据项集的支持度:一个项集的出现次数占全部记录的比例.如果一个项集是频繁的,它的子集,分母不变,分子变大,对应支持度也变大,所以也是频繁的;非频繁,类似).

利用Apriori原理,我们可以大量减少可能感兴趣的项集,减少了一部分计算量.

Apriori算法:1)生成候选项集(可能感兴趣的项集),2)统计计算支持度,3)根据支持度筛候选项集,得到频繁项集.

有一个先后顺序,先生成,后统计,筛选.
"""

def loadDataset():
    return [[1,3,4],[2,3,6],[1,2,3,5,6],[2,5,1,6]]

# 生成大小为1的候选项集
def createC1(dataset):
    C1 = []
    for transaction in dataset:
        for item in transaction:
            if [item] not in C1:
                C1.append([item])
    C1.sort()
    #frozenset:不可变对象,可以用作字典键值
    return list(map(frozenset, C1))

# 根据上次的筛选结果,生成下一长度的候选项集,eg L1-->C2,L5-->C6
# 基本方法是两两合并,求并集(结果比原来项集长度多1项). 
# 但是为了减少重复计算,我们筛选前k-2个元素相等的项集进行合并,给出结果(k是当前长度,k-1是筛选结果项集长度,k-2是频繁项集的除了最后一个外都相同,并集之后,长度为k)
def aprioriGenCk(Lprev, k):
    Ck = []
    lenLprev = len(Lprev)
    for i in range(lenLprev):
        for j in range(i+1, lenLprev):
            L1 = list(Lprev[i])[:k-2]; L2 = list(Lprev[j])[:k-2]
            L1.sort(); L2.sort()
            if L1 == L2:
                Ck.append(Lprev[i] | Lprev[j])
    
    return Ck

# 扫描一遍数据集D,求支持度,并根据最小支持度筛选候当前长度的选项集,
# 返回支持度结果计算,以及筛选后的频繁项集
# notes: dataset是set类型
def scanD(dataset, Ck, minSupport=0.666):
    # 项集计数统计结果
    supCnt = {}
    numItems = 0
    for transaction in dataset:
        numItems += 1
        for event in Ck:
            if event.issubset(transaction):
                # event可以做字典键值, 如果不存在当前键值,返回默认值0,然后+1,计数
                supCnt[event] = supCnt.get(event, 0) + 1
    # TypeError: object of type 'map' has no len() map转成list
    # numItems = len(list(dataset))
    supDict = {}
    Lk = []

    for key in supCnt:
        support = supCnt[key]/numItems
        if support >= minSupport:
            Lk.append(key)
        # 所有项集的支持度字典
        supDict[key] = support
    
    return supDict, Lk

# apriori算法,生成当前数据集的所有符合条件的频繁项集
def apriori(dataset, minSupport=0.666):
    # 从长度为1的候选项集C1开始
    C1 = createC1(dataset)
    # 将数据记录转换成set, 我们只关心每个项目是否出现,并不关注它出现了多少次--->转成set,没有影响
    D = list(map(set, dataset))
    supDict, L1 = scanD(D, C1, minSupport)
    # 存储所有长度的频繁项集
    L = [L1]
    # 接下来从k=2开始
    k = 2
    # 从上次筛选后的频繁项集开始,循环生成下一长度的候选项集
    while(len(L[k-2])>0):#如果上次频繁想次不为空----终止条件,全部项集存储结果L最后包含一个空列表
        # 生成新的候选项集
        Ck = aprioriGenCk(L[k-2], k)
        # 针对该长度的候选项集,进行筛选
        supK, Lk = scanD(D, Ck, minSupport)
        # 存储筛选结果
        L.append(Lk)
        # 存储当前长度项集的支持度字典,使用字典更新字典,supDict项增多
        supDict.update(supK)
        k += 1 # 准备生成下一长度的候选项集
    
    return L, supDict

"""
针对频繁项集生成关联规则:一般情况下,遍历所有可以出现规则右部分的项集(也就是针对当前项集求非空子集),然后生成一条候选规则,
针对这条规则,计算规则的confidence置信度,筛选符合条件的关联规则;但是这种方法太过于耗时,时间成本大,因此,我们想是否可以有apriori原理
一样的性质,可以减少感兴趣的候选规则.

针对当前频繁项集来说,一条关联规则可以由规则右部分唯一确定.

依然我们从置信度的公式出发,A->B规则的置信度计算公式:support(A|B)/support(A).语言描述来说是两个集合A,B并集(都出现)的支持度除以A出现的支持度
在A发生的情况下,B发生的概率.

我们注重观察不可能的情况(不用接着考虑的情况),如果规则A->B置信度不满足条件,那么B的任意超集(包含B的集合)对应的规则也不满足条件.
why?因为两个A|B是一样的(或者说当前频繁项集是一样的),如果右部分B变成B的超集(包含B的集合),那么左半部分对应的支持度就会增大,分母变大,分子不变-->置信度变小

和频繁项集apriori类似,如果当前项集是非频繁的,那么项集的超集也是非频繁的.

所以,关联规则的生成方法和apriori类似,<<分级进行>>:
针对当前频繁项集,右部分从一个元素C1开始,筛选后得到L1,如果L1可以进一步组合(当前项集长度比较长)得到C2,然后筛选得到L2,...,直到Lk为空.

lk-->Ck+1:可以使用aprioriGen方法,生成下一长度的候选项集.


"""
# 针对当前频繁项集,筛选符合条件关联规则; 
# tails包含可以出现在关联规则右部的所有可能项; 
# 返回筛选后的可以出现在右部的项---方便下一步计算:将右部分的项进一步组合,生成新的可以出现在右部分的项
def calcConf(freqSet, tails, supDict, rulesList, minConf=0.7):
    prunedTails = []
    for tail in tails:
        confidence = supDict[freqSet]/supDict[freqSet - tail]
        if confidence >= minConf:
            print (freqSet-tail,'-->',tail,'conf:',confidence)
            # 符合条件,添加到关联规则列表中
            rulesList.append((freqSet-tail, tail, confidence))
            prunedTails.append(tail)
    # 返回筛选后的项集
    return prunedTails

# 针对当前频繁项集,从右半部分候选集生成关联规则,根据置信度筛选,
# 一个反复的过程,直到当前规则只有一条,右半部分不能组合为止(右半部分是全集--当前频繁项集)
def rulesFromConseq(freqSet, tails, supDict, rulesList, minConf=0.666):
    # 当前候选集项目长度
    m = len(tails[0])
    # 判断当前频繁项集长度是否大于m---决定能否终止
    while(len(freqSet) > m):
        prunedTails = calcConf(freqSet, tails, supDict, rulesList, minConf)
        # 如果当前筛选后的prunedTails规则多余一条---说明可以进一步组合,生成新的候选集
        if(len(prunedTails) > 1):
            # 新的右半部分候选集tails,长度+1
            tails = aprioriGenCk(prunedTails, m+1)
            m += 1
        else:#只有一条规则,终止循环
            break

# 关联规则生成器: 遍历所有频繁项集
def generateRules(L, supDict, minConf=0.666):
    rulesList = []
    for i in range(1, len(L)): #跳过频繁1项集, 遍历长度
        # 遍历当前等级(长度)的所有频繁项集
        for freqSet in L[i]:
            # 求当前项集的长度为1的右半部分候选项集,开始着手生成规则
            H1 = [frozenset([item]) for item in freqSet]
            # TypeError: 'int' object is not iterable: 没有加[item]
            rulesFromConseq(freqSet, H1, supDict, rulesList, minConf)
    
    return rulesList


if __name__ == '__main__':
    data = loadDataset()
    retList, supportData = apriori(data)
    rules = generateRules(retList, supportData, minConf=0.6)
    print(rules)