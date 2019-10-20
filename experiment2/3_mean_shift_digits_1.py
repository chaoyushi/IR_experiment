import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
from sklearn.manifold import TSNE
from matplotlib.pyplot import style
from itertools import cycle
def evaluation(labels_true,labels_pred):
    print("Normalized Mutual Information:%0.3f"%
          metrics.normalized_mutual_info_score(labels_true,labels_pred))
    print("Homogeneity: %0.3f" %
          metrics.homogeneity_score(labels_true, labels_pred))
    print("Completeness: %0.3f" %
          metrics.completeness_score(labels_true, labels_pred))
    print()

style.use('ggplot')
digits = load_digits()
data = scale(digits.data) #数据标准化

#data=digits.data
n_samples, n_features = data.shape
n_digits = len(np.unique(digits.target))
labels_true=digits.target

X=data
bandwidth = estimate_bandwidth(X, quantile=0.4, n_samples=500)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)
labels = ms.labels_
# for i in labels:
#     print(i)
cluster_centers = ms.cluster_centers_
print(cluster_centers.shape)

labels_unique = np.unique(labels)
print(labels_unique)
n_clusters_ = len(labels_unique)
evaluation(labels_true,labels)



print("number of estimated clusters : %d" % n_clusters_)


print('Estimated number of clusters: %d' % n_clusters_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(labels_true, labels,
                                           average_method='arithmetic'))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels, metric='sqeuclidean'))


data_tsne = TSNE(n_components=2, init='pca', random_state=0).fit_transform(data)
print(data_tsne.shape)
X=data_tsne
cluster_tsne=TSNE(n_components=2,init = "pca", random_state = 0).fit_transform(cluster_centers)
print(cluster_tsne.shape)

plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k

    cluster_center = cluster_tsne[k]
    # print(cluster_center)
    plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()

