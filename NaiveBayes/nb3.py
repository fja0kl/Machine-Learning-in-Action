import numpy as np
import re

"""
To-Do:

- BernoulliNB;
- MultinomialNB;
- GaussianNB;

三种贝叶斯模型的比较以及代码实现;同时完成代码优化,增强代码通用性.

"""

# 用作二分类---to 多分类
class BernoulliNB:
    def __init__(self):
        pass
    
    def fit(self, X, y):
        m, n = np.shape(X)
        pAbusive = np.sum(y)/m
        # 拉普拉斯平滑
        p0Sum = np.ones(n); p1Sum = np.ones(n)
        p0Denom = 2.0; p1Denom = 2.0

        # 遍历数据,计算p(xi|yi)
        for i in range(m):
            if y[i] == 1:
                p1Sum += X[i]
                p1Denom += np.sum(X[i])
            else:
                p0Sum += X[i]
                p0Denom += np.sum(X[i])
        # 取log防止下溢
        p0Vect = np.log(p0Sum / float(p0Denom))
        p1Vect = np.log(p1Sum / float(p1Denom))

        return p0Vect, p1Vect, pAbusive
    
    def predict(self, p0Vect, p1Vect, p1, inputX):
        p1 = sum(inputX * p1Vect) + np.log(p1)
        p0 = sum(inputX * p0Vect) + np.log(1-p1)
        
        if p1 > p0:
            return 1
        else:
            return 0


def loadDataSet():
    postingList = [['my','dog','has','flea','problems','help','please'],
                   ['maybe','not','take','him','to','dog','park','stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVec

def createVocabList(dataSet):
    vocabSet = set()
    for doc in dataSet:
        vocabSet = vocabSet | set(doc)
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print ("the word: %s is not in my Vocabulary!" % word)
    return returnVec

def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocab = createVocabList(dataSet=listOPosts)

    trainMatrix = []
    for doc in listOPosts:
        trainMatrix.append(setOfWords2Vec(myVocab,doc))
    ob = BernoulliNB()
    p0Vec, p1Vec, pAbusive = ob.fit(np.array(trainMatrix),listClasses)
    testEntry = ['love','my','dog']
    thisDoc = np.array(setOfWords2Vec(myVocab,testEntry))
    print (str(testEntry) + " classified as: %s" % ob.predict(p0Vec,p1Vec,pAbusive,thisDoc))

    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWords2Vec(myVocab, testEntry))
    print (str(testEntry) + " classified as: %s" % ob.predict(p0Vec,p1Vec,pAbusive,thisDoc))

if __name__ == '__main__':
    testingNB()