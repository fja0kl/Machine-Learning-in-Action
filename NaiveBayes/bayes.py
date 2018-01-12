#coding:utf8
from numpy import log,array,ones,zeros
import random
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
def bagOfWords2VecMN(vocabList, inputList):
    returnVec = [0]*len(vocabList)
    for word in inputList:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(len(trainCategory))
    p0Sum = ones(numWords); p1Sum = ones(numWords)
    p0Denom = 2.0; p1Denom = 2.0

    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Sum += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])#sum(trainMatrix[i])：文章i的词汇数
        else:
            p0Sum += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vec = log(p1Sum/float(p1Denom))
    p0Vec = log(p0Sum/float(p0Denom))

    return p1Vec, p0Vec, pAbusive

def classifyNB(vec2Classify, p1Vec, p0Vec, pAbusive):
    p1 = sum(vec2Classify * p1Vec) + log(pAbusive)
    p0 = sum(vec2Classify * p0Vec) + log(1-pAbusive)

    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocab = createVocabList(dataSet=listOPosts)

    trainMatrix = []
    for doc in listOPosts:
        trainMatrix.append(setOfWords2Vec(myVocab,doc))
    p1Vec, p0Vec, pAbusive = trainNB0(array(trainMatrix),listClasses)
    testEntry = ['love','my','dog']
    thisDoc = array(setOfWords2Vec(myVocab,testEntry))
    print (str(testEntry) + " classified as: %s" % classifyNB(thisDoc,p1Vec,p0Vec,pAbusive))

    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocab, testEntry))
    print (str(testEntry) + " classified as: %s" % classifyNB(thisDoc, p1Vec, p0Vec, pAbusive))

def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*',bigString)
    return [tok.lower() for tok in listOfTokens if len(tok)>2]

def spamTest():
    docList = []; classList = []; fullText = []
    for i in range(1,26):
        bigString = open('email/spam/%d.txt' % i).read()
        wordList = textParse(bigString)
        docList.append(wordList)
        classList.append(1)
        fullText.extend(wordList)

        bigString = open('email/ham/%d.txt' %i).read()
        wordList = textParse(bigString)
        docList.append(wordList)
        classList.append(0)
        fullText.extend(wordList)
    vocabList = createVocabList(docList)
    #划分训练集和测试集
    trainSet = range(50); testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainSet)))
        testSet.append(trainSet[randIndex])
        del trainSet[randIndex]
    trainMat = []; trainClasses = []
    for docIndex in trainSet:
        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p1Vec, p0Vec, pSpam = trainNB0(trainMat,trainClasses)
    errorCount = 0
    for docIndex in testSet:
        wordVec = setOfWords2Vec(vocabList,docList[docIndex])
        predict = classifyNB(wordVec,p1Vec,p0Vec,pSpam)
        if predict != classList[docIndex]:
            errorCount += 1
    errorRate = float(errorCount)/len(testSet)
    print ("the error rate is %f" % errorRate)
    return errorRate
if __name__ == '__main__':
    error = zeros(100)
    for i in range(100):
        error[i] = spamTest()
    print error
    print error.mean()