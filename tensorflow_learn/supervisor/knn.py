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


def showData():
    datingDataMat, datingLabels = file2matrix('knn_dataset.txt')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
    plt.show()


if __name__ == '__main__':
    showData()
