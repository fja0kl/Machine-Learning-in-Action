#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 首先创建fptree存储结构
class treeNode:
    def __init__(self, name, count, parent):
        self.name = name # 节点名称
        self.count = count # 节点出现次数
        self.parent = parent # 双亲
        self.children = {} # 孩子节点
        self.nodeLink = None # 相似节点指针
    
    # 如果当前节点出现过,对当前节点出现次数进行更新
    def increase(self, count):
        self.count += count
    
    # 输出方便
    def disp(self, ind=1):
        print(' '*ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind+1)

'''
FPGrowth算法分为两部分:
- 构建fp-tree;
- 从FP-Tree中挖掘频繁项集.

1)构建fp-tree
FP-Tree每个节点是一个1频繁项集.
在FP-Tree存储结构中,每个节点存储节点名称,当前节点在数据集中出现次数,节点双亲(便于上溯),节点孩子以及相似节点指针.

FP-Tree自上而下,每个节点出现次数逐渐降低,同时数据集中一条记录对应FP-Tree中的一条路径,如果记录中的单个数据项没有出现在FP-Tree中,创建一个新节点;
反之,如果出现了,对这个节点的出现次数进行更新;


既然是查找频繁项集,必然有一个最小支持度.而fptree中每一个节点都是满足最小支持度的1频繁项集,如果不满足,筛选掉,筛选输入记录,因此筛选之前,我们需要对所有
单个数据项进行统计,统计出现次数,根据最小支持度对单个候选频繁项进行筛选,筛选后得到L1频繁项集;
在第一次遍历数据集时,我们需要一个数据结构保存单个数据项的统计结果,因此设计一个头指针链表保存;
头指针链表中保存符合条件单元素项,它的出现次数,相似节点指针列表.

得到fptree的所有构成节点之后,剩下的工作就是生成fptree,遍历数据集,对每一条记录根据单个数据项的全局频率进行排序,然后再把这条记录看做一条路径,在fptree上进行
遍历创建节点,循环完所有数据集记录后,就可以得到fptree.剩下的工作就是从生成fptree中挖掘频繁项集.

2)从fp-tree中挖掘频繁项集

从上面的描述中,我们可以知道fp-tree中所有节点是符合条件的1频繁项集L1,但是我们需要得到所有频繁项集(所有长度),
想法: 根据1频繁项集中元素,针对每个1频繁项集,筛选包括这个频繁项集的记录,得到新的数据集,我们使用这个新数据集生成新的fp-tree和头指针链表;
因为这是在1频繁项出现的基础上,生成的条件fp-tree,得到头指针节点中的每个单数据项是符合最小支持度的1频繁项,这是在上次频繁项基础上的新的1频繁项,我们将这个条件头指针链表中
的元素和上次的频繁项组合,就可以得到新的2频繁项集,这样逐渐递归,我们可以生成所有长度的频繁项集,get the point.

基于上的想法以及fp-tree结构,因为fp-tree中每一条路径都对应数据集中的一条记录,所以,我们可以直接对fp-tree进行遍历,就不用对数据集进行遍历了,同时fp-tree也对数据集中的记录做了
一个简单的统计,遍历效果更快,更高效---一条路径(根节点到叶子节点)的终点的count字段表明这条路径出现的次数,也就是数据集中相同记录的数目.

此外,条件数据集(筛选后的数据集,筛选条件是上次结果的1频繁项[头指针链表中的元素项])怎么表示?在fp-tree中是什么?怎么表示的?
首先肯定是以路径的形式保存的,但是这条路径一定是全路径(从根到叶子节点)吗?我们知道fp-tree中上下层是否一定关系的,根据之前创建fp-tree的规则,主要是节点更新规则,我们可以
知道上面节点对应的count比下面节点的count要大,也就是说上面节点的出现次数比下面孩子节点的出现次数要多(跟新,只要孩子出现了,双亲也要出现,更新--因为我们依据全局出现次数对数据记录进行了降序排序).
所以,这个条件数据集,或者说是条件路径,应该不是全路径,如果是全路径的话,下面节点是一定不符合最小支持度的,现在出现次数都比当前节点小,条件以后更下,所以可以筛选掉,只保留到当前节点的路径,同时,
我们也筛选掉这个路径终点,因为终点都相同.所以,条件数据集,就是以当前1频繁项集项(1频繁项集有好多,我们先选一个)结尾的路径集合,此外,我们还可以给他一个定义,叫做条件模式基.

条件模式基:以所查找元素项为结尾的路径集合(前缀路径).-----本质上就是条件数据集.

之后,以上次的fp-tree和头指针链表为基础,查找条件数据集,以这个条件数据集做输入,生成新的条件fp-tree,生成的条件fp-tree中的节点都是符合条件的1频繁项集,也是在上次元素项为1频繁项集的基础上,
这些元素项也是1频繁项集,所以两者组合,得到2频繁项集,反复递归,我们就可以得到以这个1频繁项集为基础的所有长度的频繁项集,之后我们遍历所有初始fp-tree的1频繁项集元素,就可以得到所有的频繁项集了!!!
'''


#===================1)构建FPTree==============================================#

# 更新树:
# items 根据单元素项全局频率降序排序后的列表;
# inTree fpTree
# headerTable 头指针链表
# count items数据记录出现次数 
def updateTree(items, inTree, headerTable, count):
    # 先看数据集的第一个元素项是否在fptree中
    if items[0] in inTree.children:# 在,直接更新
        inTree.children[items[0]].increase(count)
    # 不在,创建新节点,保存到树中,同时保存到头指针链表中
    else:
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        # 更新头指针链表中这个节点的相思项指针
        if headerTable[items[0]][1] == None: #之前没有出现过,fptree的第一个items[0]节点
            headerTable[items[0]][1] = inTree.children[items[0]]
        else: # 出现过,更新:遍历到指针链表尾
            updateHeaderTable(headerTable[items[0]][1], inTree.children[items[0]])
    # 如果当前记录还有剩余元素,继续更新
    if len(items) > 1:# inTree参数永远对应:双亲节点,上次末尾
        updateTree(items[1::],inTree.children[items[0]], headerTable, count)

def updateHeaderTable(nodeInTable, targetNode):
    # 遍历到当前元素项相似节点链表末尾
    while (nodeInTable.nodeLink != None):
        nodeInTable = nodeInTable.nodeLink
    nodeInTable.nodeLink = targetNode

# 创建fptree
# 参数 
# dataset 输入数据集;数据格式字典,键是当前数据记录集合,可以做字典键,是frozenset格式(不可变),值是当前记录出现次数
# minSup 最小支持度
def createFPTree(dataset, minSup=1):
    headerTable = {}
    # 第一次遍历数据集.对单个元素项出现次数做统计
    for trans in dataset:
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataset[trans]
    # 统计完成之后,我们得到1频繁项候选集C1,之后要根据最小支持度对候选集进行筛选,删除不满足条件的元素项
    for k in headerTable.copy():
        if headerTable[k] < minSup:
            del (headerTable[k])
    # 如果筛选后,1频繁项集为空,直接返回
    freqItemSet = set(headerTable.keys())
    if(len(freqItemSet) == 0): 
        return None, None
    # 对头指针链表扩展,增加一个相似指针字段,指向相似项
    for k in headerTable:
        headerTable[k] = [headerTable[k], None]
    # 开始生成fptree, 先生成一个根节点
    retTree = treeNode('Null Set', 1, None)
    # 第二次遍历数据集.生成fptree
    for trans, count in dataset.items():
        # 首先依据单元素项出现频率对每条记录做降序排序
        localD = {}# 当前记录中每个元素项出现的次数
        for item in trans:
            if item in freqItemSet:# 当前元素在1频繁项集中
                localD[item] = headerTable[item][0]
        # 筛选后,如果记录不为空
        if len(localD) > 0:
            #对这条记录排序
            orderedItems = [v[0] for v in sorted(localD.items(),key=lambda p:p[1], reverse=True)]
            # 根据这条记录更新fptree
            updateTree(orderedItems, retTree,headerTable,count)
    
    return retTree, headerTable


    
#=======================2)从FP树中挖掘频繁项集========================================#

# 给定元素项生成一个条件模式基:以所查找元素项为结尾的路径集合,每一条路径其实是一条前缀路径(介于所查找元素项与树根节点之间的所有内容). 
# treeNode是指向相似节点的指针,可以从头指针链表中获得
def findPrefixPath(item, treeNode):
    # 条件模式基
    condPats = {}
    while treeNode != None:# 遍历所有相似节点
        prefixPath = []
        # 上溯查找路径,包含这个节点本身
        ascendTree(treeNode, prefixPath)
        # 除了这个节点之外,还有其他节点
        if len(prefixPath) > 1:
            # 添加到条件模式基中
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        # 更换到下一个相似节点
        treeNode = treeNode.nodeLink
    
    return condPats

# 上溯路径: 叶子节点.保存路径列表;第一个元素是当前节点:保存的路径是从叶子到根节点
def ascendTree(leafNode, prefixPath):
    while leafNode.parent != None:
        prefixPath.append(leafNode.name)
        leafNode = leafNode.parent

# 从fp-tree中挖掘频繁项集
def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    # for i in headerTable.items():
    #     print (i[1][0])

    # 对当前头指针链表中L1频繁项集进行升序排序
    L1 = [v[0] for v in sorted(headerTable.items(), key=lambda a:a[1][0])]
    # key=lambda a:a[1] 会报错 TypeError: unorderable types: treeNode() < treeNode(); items函数将字典项转换成元组对,
    # 遍历L1频繁项集
    for basePat in L1:
        # 上次遍历的前缀---用来生成频繁项集
        newFreqSet = preFix.copy()
        # 添加当前元素生成新的频繁项集
        newFreqSet.add(basePat)
        # 保存
        freqItemList.append(newFreqSet)
        # 查找条件模式集
        condPatBases = findPrefixPath(basePat, headerTable[basePat][1])
        # 使用条件模式基,生成条件fptree,进而生成新的频繁项集
        myCondTree, myHead = createFPTree(condPatBases, minSup)
        if myHead != None:# 如果不为空,继续递归
            print ('conditional tree for :', newFreqSet)
            myCondTree.disp()
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)

# 封装定义fpgrowth算法
def fpGrowth(dataset, minSup=3):
    initSet = createInitSet(dataset)
    myFPTree, myHeaderTab = createFPTree(initSet, minSup)
    freqItems = []
    mineTree(myFPTree,myHeaderTab,minSup,set([]),freqItems)

    return freqItems


#生成数据集
def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]

    return simpDat
# 需要转换成字典形式,键是记录集,值是对应记录集合在数据集中出现的次数
def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1

    return retDict

if __name__ == '__main__':
    dataSet = loadSimpDat()
    freqItems = fpGrowth(dataSet)
    print (freqItems)