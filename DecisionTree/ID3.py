from math import log

"""
决策树生成过程:
1. 判断数据集是否是同一类别,如果是停止;
2. 判断特征集是否为空,如果是进行投票表决,决定当前叶子节点的类别归属;否则,继续划分;
3. 在当前数据集中选择[最佳属性]进行[数据集划分],然后在子数据集上继续划分.

[最佳属性]:ID3判断方法是使用信息增益为衡量方法,信息增益越大,越好;
[数据集划分]:多分支划分,每种属性值对应一个决策树分支;取值情况越多,分支数目越多.

ID3决策树划分时,趋向于选择取值情况多的特征[bad];同时,ID3处理数据为名词标称型数据/连续数据, 但是连续数据要进行离散化,本质上还是标称型[离散型]数据.

"""

"""
dataset 数据集格式:list(list)
每条记录前n-1个元素为属性值,最后一个为标签信息(类别信息)
"""
class ID3(object):
    def calEntropy(self, dataset):
        "计算信息熵"
        n = len(dataset)
        res = {}

        for line in dataset:
            res[line[-1]] = res.get(line[-1], 0) + 1

        entropy = 0.0
        for key in res:
            p = res[key]/(n*1.0)
            entropy -= p*log(p, 2)
        
        return entropy

    def splitMethod(self, dataset, axis, value):
        """
        数据集划分方法:在数据集的某个特征上选择等于目标特征取值的数据记录,返回这些记录值
        :param dataset:数据集
        :param axis: 选择的特征，在该特征上对数据集进行划分
        :param value: 特征取值;
        :return: 划分后的子集
        """
        retDataSet = []
        for line in dataset:
            if line[axis] == value:
                newLine = line[:axis]
                newLine.extend(line[axis+1:])
                retDataSet.append(newLine)
        
        return retDataSet

    def chooseBestFeature(self, dataset):
        """
        选择最好的特征对数据集进行划分;
        最好的评价是:信息增益最大.
        :param dataSet: 训练数据集
        :return: bestFeatureIdx 选择划分数据特征的下标
        """
        originEntropy = self.calEntropy(dataset)
        bestFeatureIdx = -1
        bestGainInfo = 0.0
        numFeatures = len(dataset[0]) -1 
        for i in range(numFeatures):
            #确定特征取值,根据取值尝试划分[多分支,每个取值一个分支]
            featureRanges = [line[i] for line in dataset]
            featureRanges = set(featureRanges)

            #根据取值范围划分数据,同时计算条件信息熵
            iFeatureEntropy = 0.0
            for value in featureRanges:
                retDataset = self.splitMethod(dataset, i, value)
                # 计算划分后数据集的熵
                iValueEnt = self.calEntropy(retDataset) 
                iFeatureEntropy += len(retDataset)/len(dataset) * iValueEnt # 加权
            gainInfo = originEntropy - iFeatureEntropy

            if gainInfo > bestGainInfo:
                bestGainInfo = gainInfo
                bestFeatureIdx = i
        
        return bestFeatureIdx

    def majorCnt(self, dataset):
        """
        对数据集类别进行投票,确定节点分类信息
        """
        labelsDict = {}
        for record in dataset:
            label = record[-1]
            labelsDict[label] = labelsDict.get(label, 0) + 1
        
        #根据类别统计次数进行降序排序[sortDict是一个元祖]
        sortDict = sorted(labelsDict.items(), key=lambda a:a[1], reverse=True)
        # 返回类别出现次数最多的类别信息
        return sortDict[0][0]

    def createDTree(self, dataset, featuresList):
        """
        创建决策树: 一个递归方法
        :param dataset 数据集
        :param featuresList 特征列表
        """
        #1. 终止条件一: 数据集都是同一类别
        classList = [record[-1] for record in dataset]
        if classList.count(classList[0]) == len(dataset):
            return classList[0]
        #2. 终止条件二: 属性集列表为空[每次划分,ID3都会从特征列表中删除当前划分特征,因此特征列表长度逐渐减少]
        # 叶子节点分类由投票表决
        if len(featuresList) == 1:
            return self.majorCnt(dataset)
        # 其他情况
        bestFeatIdx = self.chooseBestFeature(dataset)
        bestFeat = featuresList[bestFeatIdx]
        # 使用字典表示决策树
        DTree = {bestFeat : {}}
        # 在特征列表中删除该特征
        del(featuresList[bestFeatIdx])

        #确定当前划分特征的取值情况,进行多分支划分
        featRanges = [record[bestFeatIdx] for record in dataset]
        featRanges = set(featRanges)

        #3. 一般情况,决策树生成过程
        for value in featRanges:
            # 复制删除划分特征后的新特征列表
            subFeatList = featuresList[:]
            # 递归划分:在子集上继续进行决策树生成
            subDataSet = self.splitMethod(dataset, bestFeatIdx, value)
            DTree[bestFeat][value] = self.createDTree(subDataSet, subFeatList)

        return DTree

    def predict(self, tree, featList, inputRecord):
        """
        新纪录划分
        :param tree 决策树字典
        :param featList 特征列表
        :param inputRecord 输入记录
        """
        firstFeat = list(tree.keys())[0]
        firstIdx = featList.index(firstFeat)
        secondTree = tree[firstFeat]
        # 判断
        for key in secondTree:
            if inputRecord[firstIdx] == key:
                # 如果决策树中对应节点还是一棵树,继续在子树中进行判断,否则直接返回叶子节点的值
                if isinstance(secondTree[key], dict):
                    classLabel = self.predict(secondTree[key], featList, inputRecord)
                else:
                    classLabel = secondTree[key]

        return classLabel

    def storeDTree(self, tree, fileName):
        """
        保存模型，方便下次使用
        :param tree:生成的决策树模型
        :param fileName: 保存文件的名称
        :return:
        """
        import pickle
        with open(fileName, 'w') as fw:
            pickle.dump(tree,fw)

    def loadDTree(fileName):
        import pickle
        with open(fileName) as f:
            return pickle.load(f)

def read2DataSet(filename):
    fr = open(filename,'r')
    dataSet = [example.strip().split('\t') for example in fr.readlines()]
    lenseFeats = ['age','prescript','astigmatric','tearRate']

    return dataSet,lenseFeats


if __name__ == "__main__":
    dataSet, labels = read2DataSet("./data/lenses.txt")
    label = labels[:]#防止后面的操作影响原始数据
    id3 = ID3()
    myTree = id3.createDTree(dataSet,label)
    predict = id3.predict(myTree,labels,['pre','myope','no','normal'])
    print(myTree)
    print(predict)
