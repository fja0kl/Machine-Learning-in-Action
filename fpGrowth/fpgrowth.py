#coding:utf8
from numpy import *

"""
创建TP-Tree数据存储结构:Node信息
"""
class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue # 当前节点名称
        self.count = numOccur # 当前节点出现次数
        self.nodeLink = None # 指向相似节点(nameValue相同)
        self.parent = parentNode # 父节点,用来从叶子节点进行上溯
        self.children = {} # 子节点,下一层

    def inc(self, numOccur): # 如果当前节点出现过,对出现次数更新
        self.count += numOccur

    def disp(self, ind=1):#　对树输出方便
        print ' '*ind, self.name, ' ', self.count
        for child in self.children.values():
            child.disp(ind+1)

#=======================构建FP树========================================#

# 由数据记录(根据单个元素项出现全局频率排序)更新树
def updateTree(items, inTree, headerTable, count):
    # 如果记录中的第一个项已经在树中了,直接对该节点进行更新
    if items[0] in inTree.children:
        inTree.children[items[0]].inc(count)
    # 如果不在,创建一个新节点,并添加到树上
    else:
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        # 更新头指针链表中的相似元素链表
        if headerTable[items[0]][1] == None:
            headerTable[items[0]][1] = inTree.children[items[0]]
        else: # 如果已经在头指针链表里,找到该节点对应相似链表的末尾.然后更新
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    # 对记录中剩下的元素逐个进行更新
    if len(items) > 1:#双亲节点改变了
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)

#获取头指针表中该元素项对应的单链表的尾节点，然后将其指向新节点targetNode        
def updateHeader(nodeToTest, targetNode):
    # nodeToTest是头指针列表中对应元素项节点; targetNode是新生成的目标节点
    while(nodeToTest.nodeLink != None):# 头指针中当前节点的nodeLink不为空---已经指向了相似FP-Tree的节点
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode

# 创建FP-Tree
def createTree(dataSet, minSup=1):
    # 头指针列表
    headerTable = {}
    # 第一次遍历数据集,对单个数据项进行统计计数,放到头指针列表中
    for trans in dataSet:
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
    # 过滤掉小于最小支持度的单个数据项
    for k in headerTable.keys():
        if headerTable[k] < minSup:
            del (headerTable[k])
    # 空元素集，返回空
    freqItemSet = set(headerTable.keys())
    if len(freqItemSet) == 0: return None, None
    # 头指针链表增加一个指向相似节点的指针
    for k in headerTable:
        headerTable[k] = [headerTable[k], None]
    # 创建fpTree的根节点root
    retTree = treeNode('Null Set', 1, None)
    # 第二次遍历数据集,创建FPTree
    for tranSet, count in dataSet.items():
        # 首先需要对每条记录根据单个元素项的全局频率进行降序排序,方便生成FPTree
        localD = {}
        for item in tranSet:
            if item in freqItemSet:# 只选择符合支持度的单个频繁项
                localD[item] = headerTable[item][0]
        # 筛选后,当前记录不为空
        if len(localD) > 0:
            # 根据全局频率降序排序
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1],reverse=True)]
            updateTree(orderedItems, retTree, headerTable, count)
    return retTree, headerTable

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

#=======================从FP树中挖掘频繁项集========================================#

# 给定元素项生成一个条件模式基:以所查找元素项为结尾的路径集合,每一条路径其实是一条前缀路径(介于所查找元素项与树根节点之间的所有内容).
# basePat表示输入的给定元素频繁项(1频繁项)，treeNode为当前FP树中对应的第一个节点(通过headerTable[basePat][1]获取)
def findPrefixPath(basePat, treeNode):
    # 条件模式基:字典形式,值是count计数
    condPats = {}
    # 通过遍历basePat的相似项,查找所有的这个节点结尾的路径,最后转换成前缀路径,保存在conPats中
    while treeNode != None:
        prefixPath = []
        # 查找一条路径
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1:# 如果路径长度大于1,当前路径包括叶子节点本身;保存这条条件模式基
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        # 对下一个相似节点进行遍历,查找到所有的条件模式基
        treeNode = treeNode.nodeLink
    
    return condPats
# 辅助函数,对树上溯,从叶子节点到根节点,找到一条路径:保存形式list,from leaf node to root. 
# 不保存null set根节点
def ascendTree(leafNode, prefixPath):
    while leafNode.parent != None:
        prefixPath.append(leafNode.name)
        leafNode = leafNode.parent
        # ascendTree(leafNode.parent, prefixPath)

# 递归查找频繁项集:一个递归方法
# 对于每一个频繁项都要创建一个条件FP树,比如:x,{t,y}构建条件树.使用刚才发现的条件模式基作为输入数据,构建这些FP树,
# 因为是在上次频繁项结果下进行的,或者说输入数据是上次频繁项的条件模式基,所以生成的FP树叫做条件模式树,
# 随着递归的不断深度,保存所有频繁项的列表不断增大.
# 
# 参数:
# inTree和headerTable是由createTree()函数生成的数据集的FP树以及对应的头指针链表
# minSup表示最小支持度,根据最小支持度对频繁项集进行筛选
# preFix请传入一个空集合（set([])），将在函数中用于保存当前前缀
# freqItemList请传入一个空列表（[]），将用来储存生成的频繁项集
def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    # 对头指针的1频繁项根据全局频率进行升序排序:头指针列表中的单个元素都是符合条件的1频繁项集
    bigL = [v[0] for v in sorted(headerTable.items(),key=lambda p: p[1])]
    # 因为头指针列表中的节点本身都是1频繁项集,又因为是条件模式基,也就是说这些输入是保证上次项是频繁的基础上进行的
    # 所以直接将当前1频繁项添加到前缀集合preFix中,得到的就是更高一级的频繁项集
    # 递归方法
    for basePat in bigL:
        # preFix上次循环的前缀集合
        newFreqSet = preFix.copy()
        # 将当前频繁项保存上次的前缀结果中,得到新的频繁项,然后保存到频繁列表中.
        newFreqSet.add(basePat)
        # 添加到所有频繁列表中
        freqItemList.append(newFreqSet)
        # 查找当前频繁项的条件模式基,用于下次递归的输入
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        # 根据输入和最小支持度,创建条件模式树
        myConTree, myHead = createTree(condPattBases, minSup)
        # 如果创建的条件模式树为空:myHead头指针列表不为空---so, The conditional fp-tree is NOT empty.
        if myHead != None:
            print 'conditional tree for :', newFreqSet
            myConTree.disp()
            # 递归创建条件模式树
            mineTree(myConTree,myHead,minSup,newFreqSet,freqItemList)

# 封装定义fpgrowth算法
def fpGrowth(dataset, minSup=3):
    initSet = createInitSet(dataset)
    myFPTree, myHeaderTab = createTree(initSet, minSup)
    freqItems = []
    mineTree(myFPTree,myHeaderTab,minSup,set([]),freqItems)

    return freqItems

if __name__ == '__main__':
    '''
    data = loadSimpDat()
    data = createInitSet(data)
    myFPTree, myHeaderTab = createTree(data, 3)
    myFPTree.disp()
    print(myHeaderTab)
    '''
    dataSet = loadSimpDat()
    freqItems = fpGrowth(dataSet)
    print freqItems
