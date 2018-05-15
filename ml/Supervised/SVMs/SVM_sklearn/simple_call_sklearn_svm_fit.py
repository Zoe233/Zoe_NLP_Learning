from sklearn import svm

# 获取或定义 实例X向量化 和 目标变量向量化
X = [[2,0],[1,1],[2,3]]  # 3个2维实例
y = [0,0,1]

# 实例化svm
clf = svm.SVC(kernel = 'linear')
clf.fit(X,y)
print(clf)

# 获取支持向量support_vectors
print(clf.support_vectors_)

# 获取X实例中是支持向量的索引位置
print(clf.support_)

# 获取每个类的支持向量数目
print(clf.n_support_)