import pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.datasets import load_breast_cancer as load
from sklearn.preprocessing import StandardScaler as SS
from sklearn.cluster import KMeans as KM
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix as cm, classification_report as cr
X, y = load(return_X_y=True)
X = SS().fit_transform(X)
k = KM(n_clusters=2, random_state=42).fit(X)
y_k = k.labels_
print("Confusion Matrix:\n", cm(y, y_k))
print("\nClassification Report:\n", cr(y, y_k))
X2 = PCA(n_components=2).fit_transform(X)
df = pd.DataFrame(X2, columns=['PC1', 'PC2'])
df['Cluster'], df['True'] = y_k, y
for h, t in zip(['Cluster', 'True'], ['K-Means Clustering', 'True Labels']):
    sns.scatterplot(data=df, x='PC1', y='PC2', hue=h, s=100, edgecolor='k', alpha=0.7).set_title(t)
    plt.show()

sns.scatterplot(data=df, x='PC1', y='PC2', hue='Cluster', s=100, edgecolor='k', alpha=0.7)
c = PCA(2).fit_transform(k.cluster_centers_)
plt.scatter(c[:,0], c[:,1], s=200, c='red', marker='X', label='Centroids')
plt.title('K-Means with Centroids')
plt.legend(); plt.show()
