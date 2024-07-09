import scipy
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

data = []
labels = []
#dataset_path =""
test_txt = scipy.io.loadmat('/data/MS-COCO/test_txt.mat')
test_lab = scipy.io.loadmat('/data/MS-COCO/test_lab.mat')
print()
data = np.array(test_txt.get('test_txt'))
print(data.shape)
labels = np.array(test_lab.get('test_lab'))
print(labels.shape)

#iris = load_iris()

#print(iris.data.shape)#(150,4)

#print(iris.target.shape)#(150,)

tsne = TSNE(n_components=2, learning_rate=1).fit_transform(data)
labels = TSNE(n_components=1, learning_rate=1).fit_transform(labels)


#pca = PCA().fit_transform(data)

# plt.figure(figsize=(12, 6))
# plt.subplot(121)
plt.scatter(tsne[:, 0], tsne[:, 1], c=labels)
# plt.subplot(122)
# plt.scatter(pca[:, 0], pca[:, 1], c=labels)
plt.colorbar()
plt.show()

# pca = PCA().fit_transform(data)
#
# plt.figure(figsize=(12, 6))
# plt.subplot(121)
# for i in range(labels.shape[1]):
#     plt.scatter(tsne[labels[:, i] == 1, 0], tsne[labels[:, i] == 1, 1], label=f'Class {i}')
# # plt.scatter(tsne[:, 0], tsne[:, 1], c=labels)
# plt.subplot(122)
# for i in range(labels.shape[1]):
#     plt.scatter(pca[labels[:, i] == 1, 0], pca[labels[:, i] == 1, 1], label=f'Class {i}')
# # plt.scatter(pca[:, 0], pca[:, 1], c=labels)
# plt.colorbar()
# plt.show()