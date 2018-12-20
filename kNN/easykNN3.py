import numpy as np

# kNN:计算量比较大,没有显性的学习过程,根据输入样例与数据集之间相似性,选择k个最相似的,然后投票确定输入所属分类.
class kNN:
    def __init__(self):
        pass
    
    def fit(self):
        pass

    def predict(self, data, labels, inX, K=3):
        X = np.array(data)
        inX = np.array(inX)
        diffMat = X - inX
        sqDiffMat = diffMat ** 2
        sqDiffMat = np.sum(sqDiffMat, axis=1)#按照行求和,计算距离平方
        distances = sqDiffMat ** 0.5 #开平方,得到距离
        #根据距离排序,得到由小到大对应的下标;不能直接对距离数组排序,因为排序后和labels就对应不上了.
        sortDistIdxs = distances.argsort()#返回数组由小到大排序后对应原数组的下标
        # 选择最近的k个记录点
        classCnt = {}#投票记录
        for i in range(K):
            voteLabel = labels[sortDistIdxs[i]]
            classCnt[voteLabel] = classCnt.get(voteLabel, 0) + 1
        # 对投票结果统计,少数服从多数
        sortedClassCnt = sorted(classCnt.items(), key=lambda a: a[1], reverse=True)
        # 返回预测结果
        return sortedClassCnt[0][0]

def createDataSet():
    group = [[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]]
    lables = ['A','A','B','B']
    return group, lables

if __name__ == '__main__':
    dataSet, labels = createDataSet()
    inX = [0,0.2]
    cate = kNN().predict(dataSet,labels, inX)
    print (cate)
