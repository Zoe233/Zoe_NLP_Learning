import csv
import random
import math
import operator

DEBUG = False

class HandWritngKnn(object):
    '''
    根据KNN算法的一般步骤，自定义函数方法，手写Iris数据集的KNN算法
    '''
    def __init__(self):
        self.filename ='irisdata.txt'
        self.split = 0.67
        self.trainSet = []
        self.testSet =[]
        self.k = 3

    def run(self):
        self.trainSet,self.testSet = self.loadDataset(filename=self.filename,split=self.split)
        if DEBUG:
            print(self.trainSet)
            print(len(self.trainSet))
            print('*' * 50)
            print(self.testSet)
            print(len(self.testSet))

        # 单条测试数据调用。用于测试函数
        # neighbors =self.getNeighbors(self.trainSet,self.testSet[0],3)
        # print('neighbors:',neighbors)
        #
        # sortedVotes = self.getResponse(neighbors)
        # print('sortedVotes:',sortedVotes)

        # 批量测试测试集与训练模型结果。
        correct = 0
        for x in range(len(self.testSet)):
            neighbors = self.getNeighbors(self.trainSet, self.testSet[x], self.k)
            result = self.getResponse(neighbors)
            if result == self.testSet[x][-1]:
                correct += 1
            else:
                correct = correct
        accuracy = correct/len(self.testSet)*100.0
        print('Accuracy:', accuracy)

    def loadDataset(self,filename,split,trainSet=[],testSet=[]):
        '''
        读取数据集,并设置 训练集 和 测试集
        :param filename: 读取数据集文件名，此处为irisdata.txt
        :param split: 用于将原始数据随机分隔为 trainSet 和 TestSet 的[0,1]之间的数据。
        :param trainSet: 用于存放训练集数据
        :param testSet: 用于存放测试集数据
        :return:trainSet, TestSet
        '''
        with open(filename,'r') as Iris:
            lines = csv.reader(Iris)
            dataset = list(lines)  # [['5.1', '3.5', '1.4', '0.2', 'Iris-setosa'], ['4.9', '3.0', '1.4', '0.2', 'Iris-setosa'], ...]

            for x in range(len(dataset)-1):
                for y in range(4):
                    dataset[x][y] = float(dataset[x][y])
                if random.random() < split: # ra ndom.random()产生[0,1]之间的随机数
                    trainSet.append(dataset[x])
                else:
                    testSet.append(dataset[x])
        return trainSet,testSet

    def __euclideanDistance(self,instance1,instance2,dimensions):
        '''
        Euclidean Distance距离计算公式，计算两个实例之间的距离
        :param instance1: 实例1
        :param instance2: 实例2
        :param dimensions: 维度个数
        :return: 返回Euclidean Distance距离
        '''
        distance = 0
        for x in range(dimensions):
            distance += pow((instance1[x]-instance2[x]),2)
        return math.sqrt(distance),instance2[-1]

    def getNeighbors(self,trainingSet,testInstance,k):
        '''
        取最近的k个实例
        :param trainingSet: 训练集
        :param testInstance: 测试实例
        :param k: 取的k值
        :return: 最近的实例列表
        '''
        distances = [] # 存储测试实例与训练集所有数据的距离结果
        dimensions = len(testInstance)-1

        res_dic ={}
        for x in range(len(trainingSet)):
            dist,instanceClassName = self.__euclideanDistance(testInstance,trainingSet[x],dimensions)
            distances.append(dist)

            res_dic[dist] = instanceClassName
        distances.sort()
        if DEBUG:
            print('+'*100)
            print(distances)
            print(res_dic)
        min_k_dist =  sorted(res_dic.keys())[0:k]
        neighbors = []
        for v in min_k_dist:
            neighbors.append(res_dic[v])
        return neighbors

    def getResponse(self,neighbors):
        '''
        判断k个neighbors的结果，按照少数服从多数的原则进行判断。
        :param neighbors: self.getNeigbors返回的neighbors对象。
        :return:返回最多的那个值的结果
        '''
        classVotes = {}
        for x in range(len(neighbors)):
            response = neighbors[x]
            if response in classVotes:
                classVotes[response] += 1
            else:
                classVotes[response] = 1
        sortedVotes = sorted(classVotes.items())
        return sortedVotes[0][0]



if __name__=='__main__':
    knnObj = HandWritngKnn()
    knnObj.run()




