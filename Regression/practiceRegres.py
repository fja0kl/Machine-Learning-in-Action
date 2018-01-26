#coding:utf8
from time import sleep
import json
import urllib2
import numpy as np
import random

def searchForSet(retX,retY,setNum,yr,numPce,origPrc):
    sleep(10)
    myAPIstr = 'get from code.google.com'
    searchURL = 'https://www.googleapis.com/shopping/search/v1/public/products?key=%s&country=US&q=lego+%d&alt=json' % (myAPIstr,setNum)
    pg = urllib2.urlopen(searchURL)
    retDict = json.loads(pg.read())
    print retDict
    for i in range(len(retDict['items'])):
        try:
            currItem = retDict['items'][i]
            if currItem['product']['condition'] == 'new':
                newFlag = 1
            else:
                newFlag = 0
            listOfInv = currItem['product']['inventories']
            for item in listOfInv:
                sellingPrice = item['price']
                if sellingPrice > origPrc*0.5:
                    print ("%d\t%d\t%d\t%f\t%f" %\
                           (yr, numPce,newFlag,origPrc,sellingPrice))
                    retX.append([yr,numPce,newFlag,origPrc])
                    retY.append(sellingPrice)
        except:
            print ("problem with item %d " % i)

def setDataCollect(retX,retY):
    searchForSet(retX,retY,8288,2006,800,49.99)
    searchForSet(retX,retY,10030,2002,3096,269.99)
    searchForSet(retX,retY,10179,2007,5195,499.99)
    searchForSet(retX,retY,10181,2007,3428,199.99)
    searchForSet(retX,retY,10189,2008,5922,299.99)
    searchForSet(retX,retY,10196,2009,3263,249.99)

def crossValidation(xArr, yArr, numVal=10):
    m = len(yArr)
    indexList = range(m)
    errorMat = np.zeros((numVal,30))
    for i in range(numVal):
        trainX = [] ;trainY = []
        testX = []; testY = []
        random.shuffle(indexList)
        for j in range(m):
            if j < m*0.9:
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        wMat = ridgeTest(trainX,trainY)
        for k in range(30):
            matTestX = np.mat(testX); matTrainX = np.mat(trainX)
            meanTrain = np.mean(matTrainX,0)
            varTrain = np.var(matTrainX,0)
            matTestX = (matTestX-meanTrain)/varTrain
            yEst = matTestX* np.mat(wMat[k,:]).T + np.mean(trainY)
            errorMat[i,k] = rssError(yEst.T.A,np.array(testY))
    meanError = np.mean(errorMat,0)
    minMean = float(min(meanError))
    bestWeights = wMat[np.nonzero(meanError==minMean)]
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    meanX = np.mean(xMat,0); varX = np.mean(xMat,0)
    unReg = bestWeights/varX
    print ("the best model from Ridge Regression is:\n" + str(unReg))
    print ("with constant term: %d" % -1*sum(np.multiply(meanX,unReg)) + np.mean(yMat))



if __name__ == '__main__':
    lgX = []; lgY=[]
    setDataCollect(lgX,lgY)