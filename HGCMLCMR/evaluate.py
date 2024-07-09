import numpy as np
import scipy
import scipy.spatial


def fx_calc_map_label(image, text, label, k=0, dist_method='COS'):
    if dist_method == 'L2':
        dist = scipy.spatial.distance.cdist(image, text, 'euclidean')
        # 该函数计算两个输入集合中每一对之间的距离。
        # 通过metric参数指定计算距离的不同方式得到不同的距离度量值。
        # metric=euclidean----------欧几里得距离（欧氏距离）
    elif dist_method == 'COS':
        dist = scipy.spatial.distance.cdist(image, text, 'cosine')
        # cosine----------余弦距离
    ord = dist.argsort()  # argsort()函数是对数组中的元素进行从小到大排序，并返回相应序列元素的数组下标。
    numcases = dist.shape[0]
    # 使用shape[0]读取矩阵第一维度的长度，即行数；
    # 使用shape[1]读取矩阵第二维度的长度，即列数。
    sim = (np.dot(label, label.T) > 0).astype(float)  # 通过astype函数显式转换数组的数据类型
    tindex = np.arange(numcases, dtype=float) + 1
    if k == 0:
        k = numcases
    res = []
    for i in range(numcases):
        order = ord[i]
        sim[i] = sim[i][order]  #
        num = sim[i].sum()
        a = np.where(sim[i]==1)[0]
        sim[i][a] = np.arange(a.shape[0], dtype=float) + 1
        res += [(sim[i] / tindex).sum() / num]

    return np.mean(res)


def calc_r_at_k(image, text, label, k=0, dist_method='COS'):
    if dist_method == 'L2':
        dist = scipy.spatial.distance.cdist(image, text, 'euclidean')
    elif dist_method == 'COS':
        dist = scipy.spatial.distance.cdist(image, text, 'cosine')

    ord = dist.argsort()[:, :k]
    num_cases = dist.shape[0]
    sim = (np.dot(label, label.T) > 0).astype(float)

    if k == 0:
        k = num_cases

    res = []
    for i in range(num_cases):
        order = ord[i]
        sim[i] = sim[i][order]
        num = sim[i].sum()
        a = np.where(sim[i] == 1)[0]
        sim[i][a] = np.arange(a.shape[0], dtype=float) + 1
        p_at_k = (sim[i][:k] <= k).sum() / min(k, num)
        res.append(p_at_k)

    return np.mean(res)
