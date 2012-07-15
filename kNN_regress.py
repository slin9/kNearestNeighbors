from numpy import *
import operator
import sys

def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())
    returnMat = zeros((numberOfLines, 3))
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        index += 1
    return returnMat

def createDataSet():
    group = array([[1, 1.1, 3], [1, 1, 4], [0, 0, -1], [0, 0.1, -2]])
    return group

def minmaxNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))
    return normDataSet, ranges, minVals

# Normalizes data to zero mean, 1 standard dev, takes optional inputs of existing values (e.g., from training set)
def standardGaussNorm(dataSet, meanValsIn=None, stdValsIn=None):
    if(meanValsIn is None and stdValsIn is None):
        meanVals = dataSet.mean(0)
        stdVals = dataSet.std(0)
    else:
        meanVals = meanValsIn
        stdVals = stdValsIn
    m = dataSet.shape[0]
    normDataSet = zeros(shape(dataSet))
    normDataSet = dataSet - tile(meanVals, (m,1))
    normDataSet = normDataSet/tile(stdVals, (m,1))
    return normDataSet, meanVals, stdVals

# performs simple average of targets
def regress0(inX, dataSet, targets, k):
	dataSetSize = dataSet.shape[0]
	diffMat = tile(inX, (dataSetSize, 1)) - dataSet
	sqDiffMat = diffMat**2
	sqDistances = sqDiffMat.sum(axis=1)
	distances = sqDistances**0.5
	sortedDistIndices = distances.argsort()
	estimate = 0.0
        for i in range(k):
		target_i = targets[sortedDistIndices[i]]
		estimate += target_i/k
	return estimate

# performs weighted average based on distances to test point
def regress1(inX, dataSet, targets, k):
        dataSetSize = dataSet.shape[0]
        diffMat = tile(inX, (dataSetSize, 1)) - dataSet
        sqDiffMat = diffMat**2
        sqDistances = sqDiffMat.sum(axis=1)
        distances = sqDistances**0.5
        sortedDistIndices = distances.argsort()
        estimate = 0.0
        weights = 0.0
        for i in range(k):
                target_i = targets[sortedDistIndices[i]]
                weights += 1/targets[sortedDistIndices[i]]
                # the closer the point is, the larger the weight it is given
                estimate += target_i/targets[sortedDistIndices[i]]
        finalEstimate = estimate/weights
        return finalEstimate
    
# test the method
# note that kNN-based regression can never obtain an estimate  > max(targets) in the targets using these approaches
# i.e., no extrapolation can ever been done
k = 2
datMat = createDataSet()
targets = datMat[:,2]
regressors = datMat[:,0:2]
testPoint = array([[10, 10]])
normTrainMat, meanVals, stdVals = standardGaussNorm(regressors)
normTestMat = standardGaussNorm(testPoint, meanVals, stdVals)[0]
regressResult0 = regress0(normTestMat, normTrainMat, targets, k)
regressResult1 = regress1(normTestMat, normTrainMat, targets, k)
print datMat
print 'Simple Average: ' + str(regressResult0)
print 'Weighted Average: ' + str(regressResult1)