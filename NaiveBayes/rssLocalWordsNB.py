#coding:utf8
import re
from numpy import *
import feedparser

def createVocabList(dataSet):
    vocabSet = set()
    for doc in dataSet:
        vocabSet = vocabSet | set(doc)
    return list(vocabSet)


def textParse(bigString):
    wordsList = re.split(r'\W*',bigString)
    retWordsList = [word.lower() for word in wordsList if len(word) > 2]
    return retWordsList

def calcMostFreq(vocabList,fullText,k):
    """
    返回频率最高的k个词
    :param vocabList:字典
    :param fullText: 所有用例的大文本
    :return: 频率最高的k个词
    """
    freqDict = {}
    for word in vocabList:
        freqDict[word] = fullText.count(word)
    sortedFreq = sorted(freqDict.items(),key=lambda a:a[1],reverse=True)
    return sortedFreq[:k]

def bagOfWords2Vect(inputList,vocabList):
    retVec = [0]*len(vocabList)
    for word in inputList:
        if word in vocabList:
            index = vocabList.index(word)
            retVec[index] += 1
    return retVec

def trainNB(trainMatrix,trainClasses):
    numOfDocs = len(trainMatrix)
    numOfFeatures = len(trainMatrix[0])
    p1 = sum(trainClasses)/float(numOfDocs)
    p0Sum = ones(numOfFeatures); p1Sum = ones(numOfFeatures)
    p0Denom = 2.0; p1Denom = 2.0
    for i in range(numOfDocs):
        if trainClasses[i] == 1:
            p1Sum += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Sum += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])

    p0Vec = log(p0Sum/float(p0Denom))
    p1Vec = log(p1Sum/float(p1Denom))

    return p0Vec, p1Vec, p1

def classifyNB(inputVec,p0Vec,p1Vec,pNegative):
    p1 = sum(inputVec*p1Vec) + log(pNegative)
    p0 = sum(inputVec*p0Vec) + log(1-pNegative)
    if p1 > p0:
        return 1
    else:
        return 0

def localWords(feed1,feed0):
    docList =[]; classList = []; fullText = []
    minLen = min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):
        bigString = feed1['entries'][i]['summary']
        wordList = textParse(bigString)
        docList.append(wordList)
        classList.append(1)
        fullText.extend(wordList)

        bigString = feed0['entries'][i]['summary']
        wordList = textParse(bigString)
        docList.append(wordList)
        classList.append(0)
        fullText.extend(wordList)
    vocabList = createVocabList(docList)
    top30Words = calcMostFreq(vocabList,fullText,30)
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])

    trainSet = range(minLen*2); testSet = []
    for i in range(20):
        randIndex = int(random.uniform(0,len(trainSet)))
        testSet.append(trainSet[randIndex])
        del trainSet[randIndex]

    trainMatrix = []; trainClasses = []
    for docIndex in trainSet:
        trainMatrix.append(bagOfWords2Vect(docList[docIndex],vocabList))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB(array(trainMatrix),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        docVec = bagOfWords2Vect(docList[docIndex],vocabList)
        predict = classifyNB(docVec,p0Vec=p0V,p1Vec=p1V,pNegative=pSpam)
        if predict != classList[docIndex]:
            errorCount += 1
    errorRate = float(errorCount)/len(testSet)
    print("error rate is %f" % errorRate)
    return vocabList, p0V, p1V

def getTopWords(ny,sf):
    vocabList, p0V, p1V = localWords(ny,sf)
    topNY = []; topSF = []
    for i in range(len(p0V)):
        if p0V[i] > -6.0: topSF.append((vocabList[i],p0V[i]))
        if p1V[i] > -6.0: topNY.append((vocabList[i],p1V[i]))
    sortedSF = sorted(topSF,key=lambda a:a[1],reverse=True)
    print ("SF**"*32)
    for item in sortedSF:
        print item[0]
    sortedNY = sorted(topNY,key=lambda a:a[1],reverse=True)
    print ("NY**"*32)
    for item in sortedNY:
        print item[0]

if __name__ == '__main__':
    ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
    sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
    getTopWords(ny,sf)