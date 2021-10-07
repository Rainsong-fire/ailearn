from sklearn.neighbors import KNeighborsClassifier

#get the data
x = [[1],[2],[0],[0]]
# this is the 特征空间
y = [1, 1, 0, 0]
#this is the expectation

# 实例化一个训练模型。
estimator = KNeighborsClassifier(n_neighbors = 2)
estimator.fit(x, y)

ret0 = estimator.predict([[-122]])
ret1 = estimator.predict([[-1]])
ret2 = estimator.predict([[3]])
ret3 = estimator.predict([[10]])
ret4 = estimator.predict([[12]])

print(ret0, ret1, ret2, ret3, ret4)
