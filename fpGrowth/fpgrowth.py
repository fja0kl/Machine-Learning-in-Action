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


if __name__ == '__main__':
    data = loadSimpDat()
    data = createInitSet(data)
    myFPTree, myHeaderTab = createTree(data, 3)
    myFPTree.disp()
    print(myHeaderTab)
