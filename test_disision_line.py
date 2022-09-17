import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
import pandas as pd

from sklearn.datasets.samples_generator import make_classification
# X为样本特征，y为样本类别输出， 共200个样本，每个样本2个特征，输出有3个类别，没有冗余特征，每个类别一个簇
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                             n_clusters_per_class=1, n_classes=3)
#之所以生成2个特征值是因为需要在二维平面上可视化展示预测结果，所以只能是2个，3个都不行
plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)
plt.show() #根据随机生成样本不同，图形也不同
clf = neighbors.KNeighborsClassifier(n_neighbors = 15 , weights='distance')
clf.fit(X, y)  #用KNN来拟合模型，我们选择K=15，权重为距离远近
h = .02  #网格中的步长
#确认训练集的边界
#生成随机数据来做测试集，然后作预测
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h)) #生成网格型二维数据对
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF']) #给不同区域赋以颜色
cmap_bold = ListedColormap(['#FF0000', '#003300', '#0000FF'])#给不同属性的点赋以颜色
#将预测的结果在平面坐标中画出其类别区域
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
# 也画出所有的训练集数据
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

plt.show()
