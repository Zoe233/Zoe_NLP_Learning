# -*- coding:utf-8 -*-

# 第一步：获取原始数据集
# 读取csv文件数据
import os
import csv

base_dir = os.path.dirname(os.path.abspath(__file__))

feature_list =[]  # 用于存储特征属性的值
label_list =[]  # 用于存储目标变量Class_buys_computer的值


# 读取
with open(os.path.join(base_dir,'AllElectronics.csv'),'r',encoding='utf-8',newline='') as f:
    # 由于csv库有自己的换行符，需要把open方法的newline设置为空，否则在Excel中会出现数据隔行。
    reader = csv.reader(f) # <_csv.reader object at 0x10e609a58> csv对象
    headers = reader.__next__()  # headers 中存储的为第一行Titles ['RID', 'age', 'income', 'student', 'credit_rating', 'class_buys_computer']

    for row in reader: # 上面读取了headers之后，此处的for循环读取数据则是从数据行开始读取。
        label_list.append(row[-1])
        rowDict = {}
        for i in range(1,len(row)-1):
            rowDict[headers[i]] = row[i]
        feature_list.append(rowDict)  #rowDict 的格式为：{'age': 'youth', 'income': 'high', 'student': 'no', 'credit_rating': 'fair'},
print(feature_list)

# 不用csv，自己解析。后续还需要根据情况修改格式。
# with open(os.path.join(base_dir,'AllElectronics.csv'), 'r') as f:
#     rows = [line.strip().split(',') for line in f.readlines()]
# print('***'*50)
# print(rows)


# 第二步：数据预处理，格式化输出符合sklearn调用的格式。
# sklearn要求输入端的数据都是数值型的，而不是字符串类型的。
# 比方age要将{"youth":1,"high":2,"senior":3}

# Vectorize features
from sklearn.feature_extraction import DictVectorizer  # 数据格式转换工具。
# 可以将字典的数据直接转换为0，1的类型，字典数据向量化
vec = DictVectorizer()
dummyX = vec.fit_transform(feature_list).toarray()
print('dummyX:',dummyX) # 将其转换为14*10(14行*10列）的数组

print(vec.get_feature_names())

print(dir(vec))
print(vec.vocabulary_)  # 标签与向量化的值对应



# vectorize class labels
from sklearn import preprocessing  # 目标变量的转换工具
lb = preprocessing.LabelBinarizer()  # 由于此例中的目标变量为是或否，二值化
dummyY = lb.fit_transform(label_list)
print("dummyY: " ,dummyY)  # 14*1的数组



# 第三步：调用sklearn，构建模型
from sklearn import tree

# Using decision tree for classification
# clf = tree.DecisionTreeClassifier()
clf = tree.DecisionTreeClassifier(criterion='entropy')
# 具体的参数调用，去http://scikit-learn.org/stable/modules/tree.html#classification官网文档查看。
clf = clf.fit(dummyX, dummyY)
print("clf: " ,clf)


# 第四步：将训练模型结果输出，并使用graphviz画出决策树图像
# Visualize model
with open("allElectronicInformationGainOri.dot", 'w') as f:
    f = tree.export_graphviz(clf, feature_names=vec.get_feature_names(), out_file=f)


# 第五步：数据测试。
# 此处是将数据集中的第一条数据的age参数向量化的值改掉后，预测的结果。
oneRowX = dummyX[0,:]
print("oneRowX:",oneRowX)

newRowX = oneRowX
newRowX[0] = 1
newRowX[2] = 0
import numpy as np
newRowX = np.array(newRowX).reshape(1,10)
print("newRowX: ",newRowX)
print(newRowX.shape)
print(dummyX.shape)

print(dir(clf))
predictedY = clf.predict(newRowX)  # 注意要求输入的格式与算法训练输入的格式一样。通过reshape格式化。

print("predictedY: ",predictedY)  # 测试结果为1
