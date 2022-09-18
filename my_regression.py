import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt


plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
path = 'data/regress_data1.csv'
data = pd.read_csv(path)
data.head()
data.describe()
data.plot(kind='scatter', x='人口', y='收益')
plt.xlabel('人口', fontsize=18)
plt.ylabel('收益', rotation=0, fontsize=18)
plt.show()


def computeCost(X, y, w):
    inner = np.power(((X * w.T) - y), 2)# (m,n) @ (n, 1) -> (n, 1)
#     return np.sum(inner) / (2 * len(X))
    return np.sum(inner) / (2 * X.shape[0])


data.insert(0, 'Ones', 1)
# set X (training data) and y (target variable)
cols = data.shape[1]
X = data.iloc[:,:cols-1]#X是所有行，去掉最后一列
y = data.iloc[:,cols-1:]#X是所有行，最后一列
X = np.matrix(X.values)
y = np.matrix(y.values)
w = np.matrix(np.array([0,0]))
computeCost(X, y, w)


def batch_gradientDescent(X, y, w, alpha, iters):
    temp = np.matrix(np.zeros(w.shape))
    parameters = int(w.ravel().shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X * w.T) - y

        for j in range(parameters):
            term = np.multiply(error, X[:, j])
            temp[0, j] = w[0, j] - ((alpha / len(X)) * np.sum(term))

        w = temp
        cost[i] = computeCost(X, y, w)

    return w, cost


alpha = 0.01
iters = 1000
g, cost = batch_gradientDescent(X, y, w, alpha, iters)
computeCost(X, y, g)
x = np.linspace(data['人口'].min(), data['人口'].max(), 100)
f = g[0, 0] + (g[0, 1] * x)
fig, ([ax1, ax2]) = plt.subplots(1, 2, figsize=(12, 8))
ax1.plot(x, f, 'r', label='预测值')
ax1.scatter(data['人口'], data['收益'], label='训练数据')
ax1.legend(loc=2)
ax1.set_xlabel('人口', fontsize=8)
ax1.set_ylabel('收益', rotation=0, fontsize=8)
ax1.set_title('预测收益和人口规模', fontsize=8)
# plt.show()
# fig, ax = plt.subplots(figsize=(12, 8))
ax2.plot(np.arange(iters), cost, 'r')
ax2.set_xlabel('迭代次数', fontsize=8)
ax2.set_ylabel('代价', rotation=0, fontsize=8)
ax2.set_title('误差和训练Epoch数', fontsize=8)
plt.show()
