**算法原理：**
    如果一个样本空间中的k个最近邻的样本中的大多数属于某一个类别，
    则该样本也属于该类别，并且最有这个类别的特性。

**步骤：**
    1.为了判断未知实例的类别，以所有已知类别的实例作为参照；
    2.选择参数K
    3.计算未知实例与所有已知实例的距离
    4.选择最近K个已知实例
    5.根据少数服从多数的投票法则，让未知实例归类为K个最近邻样本中最多数的类别

**细节：**
    关于距离的衡量方法：
        Euclidean Distance定义：
            $d = \sqrt{(x_{2}-x_{1})^2 + (y_{2}-y_{1})^2}$

​	$E(x,y) = \sqrt{\sum_{i=0}^{n}(x_{i}-y_{i})^2}$

​	

​	其他距离衡量：

​		余弦值 cos

​		相关度 correlation

​		曼哈顿距离 Manhattan distance



**算法优缺点：**

优点：

 	1. 简单
	2. 易于理解
	3. 容易实现
	4. 通过对K的选择可具备丢噪音数据的健壮性（一般K值为1，3，5，7，…奇数个，一般不大于20）—对异常值不敏感
	5. 精度高
	6. 无数据输入假定

缺点：

 1. 计算复杂度高

 2. 空间复杂度高

 3. 易受样本分布不平衡的影响。

    ​