from sklearn import neighbors
from sklearn import datasets

# 加载iris公开数据集
iris = datasets.load_iris()

# save data
# f = open("iris.data.csv", 'wb')
# f.write(str(iris))
# f.close()

# 生成KNN分类器实例
knn = neighbors.KNeighborsClassifier()

# 加载训练集
knn.fit(iris.data, iris.target)

# 验证单条数据
predictedLabel = knn.predict([[0.1, 0.2, 0.3, 0.4]])

print ("predictedLabel is :", predictedLabel)
