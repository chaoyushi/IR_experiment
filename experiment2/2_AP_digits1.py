from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
def evaluation(labels_true,labels_pred):
    print("Normalized Mutual Information:%0.3f"%
          metrics.normalized_mutual_info_score(labels_true,labels_pred))
    print("Homogeneity: %0.3f" %
          metrics.homogeneity_score(labels_true, labels_pred))
    print("Completeness: %0.3f" %
          metrics.completeness_score(labels_true, labels_pred))
    print()
digits = load_digits()
data = scale(digits.data) #数据标准化
n_samples, n_features = data.shape
n_digits = len(np.unique(digits.target))
X=data
af = AffinityPropagation(preference=-3100).fit(X)
cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_
# centroids = af.cluster_centers_
# print(centroids)
n_clusters_ = len(cluster_centers_indices)
labels_true=digits.target    #样本真实标签
print(labels)
print(labels.shape)
print(n_clusters_)
evaluation(labels_true,labels)

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

import matplotlib.pyplot as plt

data_tsne = TSNE(learning_rate=100).fit_transform(data)
colors = [['red','green','blue','grey','yellow',
           'cyan','black','white','blue','black'][i] for i in labels]
'''绘制聚类图'''
plt.scatter(data_tsne[:,0],data_tsne[:,1],c=colors,s=10)
plt.title('AP_digits')
plt.show()

