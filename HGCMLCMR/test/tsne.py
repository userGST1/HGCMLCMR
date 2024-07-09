import scipy
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
data1 = []
data2 = []
data3 = []
test_txt = scipy.io.loadmat('/data/mirflickr/test_txt.mat')
test_lab = scipy.io.loadmat('/data/mirflickr/test_lab.mat')
test_img = scipy.io.loadmat('/data/mirflickr/test_img.mat')

data1 = np.array(test_txt.get('test_txt'))
data2 = np.array(test_img.get('test_img'))
data2 = (data2 - data2.mean()) / data2.std()
data3 = np.array(test_lab.get('test_lab'))
# data1 = data1[:1000]
# data2 = data2[:1000]
x_data = data1
y_data = data2
z_data = data3
# 使用TSNE进行降维处理。从4维降至2维。
tsne = TSNE(n_components=2, learning_rate=100).fit_transform(x_data)
tsne2 = TSNE(n_components=2, learning_rate=100).fit_transform(y_data)
# 使用PCA 进行降维处理
pca = TSNE(n_components=2, learning_rate=1).fit_transform(x_data)
pca2 = TSNE(n_components=2, learning_rate=1).fit_transform(y_data)
# plt.scatter(tsne[0, 0], tsne[0, 1], c='red', label='Class 0')
# plt.scatter(tsne[1, 0], tsne[1, 1], c='blue', label='Class 1')
plt.figure()
plt.xlim(-60,60)
plt.ylim(-60,60)
plt.scatter(tsne[:, 0], tsne[:, 1],marker='.',c='r',label='image')
plt.scatter(tsne2[:, 0], tsne2[:, 1],marker='.',c='b',label='text')
plt.legend()

plt.figure()
plt.xlim(-60,60)
plt.ylim(-60,60)
plt.scatter(pca[:, 0], pca[:, 1],marker='.',c='r',label='image')
plt.scatter(pca2[:, 0], pca2[:, 1],marker='.',c='b',label='text')
plt.legend()
plt.show()