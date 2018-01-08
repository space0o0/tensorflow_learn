# coding=utf-8
import matplotlib
import matplotlib.pyplot as plt
import operator
from numpy import *


def file2matrix(fileName):
    fr = open(fileName)
    numberOfLines = len(fr.readlines())  ##数据行数
    print("the number of lines is : %d" % numberOfLines)
    returnMat = zeros(shape=[numberOfLines, 3])
    classLabelVector = []
    index = 0
    fr = open(fileName)
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1

    return returnMat, classLabelVector


def autoNorm(dataSet):
    """
    归一公式
    Y = (X-Xmin)/(Xmax-Xmin)
    """
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)

    ranges = maxVals - minVals  ##极差
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def showData():
    datingDataMat, datingLabels = file2matrix('knn_dataset.txt')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
    plt.show()


def test():
    s = array([[1., 2., 3.], [1., 2., 6.]])
    print(s)
    normMat, range, minVals = autoNorm(s)
    print(normMat)
    print(range)
    print(minVals)

if __name__ == '__main__':
    test()
