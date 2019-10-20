import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from itertools import cycle
from sklearn.manifold import TSNE
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA,SparsePCA
from sklearn.mixture import GaussianMixture
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
reduced_data = PCA(n_components=2).fit_transform(data)
# print(reduced_data.shape)
X=reduced_data
labels_true=digits.target
gmm = GaussianMixture(n_components=10)
gmm.fit(X)
labels = gmm.predict(X)
evaluation(labels_true,labels)
labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)
# labels =SpectralClustering().fit_predict(X)



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

plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    # print(cluster_center)
    plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
