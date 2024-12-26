# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 23:12:24 2024

@author: om
"""
"""
Problem Statement: -
Perform hierarchical and K-means clustering on the dataset. 
After that, perform PCA on the dataset and extract the first 3 principal components and make 
a new dataset with these 3 principal components as the columns. Now, on this new dataset, perform 
hierarchical and K-means clustering. Compare the results of clustering on the original dataset and 
clustering on the principal components dataset (use the scree plot technique to obtain the optimum 
number of clusters in K-means clustering and check if you’re getting similar results with and without PCA).
"""

"""
Business problem:
    Perform hierarchical and K-means clustering on the
    dataset. After that, perform PCA on the dataset and 
    extract the first 3 principal components and make a
    new dataset with these 3 principal components as the
    columns.

"""

import pandas as pd
import matplotlib.pylab as plt
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

# Load the dataset
wine = pd.read_csv("C:/Assignments(DS)/wine.csv")
print(wine.head())

# Normalization function
def norm_func(i):
    return (i - i.min()) / (i.max() - i.min())

# Normalize the original dataset
df_norm = norm_func(wine.iloc[:, :])
print(df_norm.describe())

# Hierarchical clustering on the original dataset
z = linkage(df_norm, method="complete", metric="euclidean")
plt.figure(figsize=(15, 8))
plt.title("Hierarchical Clustering Dendrogram (Original Dataset)")
plt.xlabel("Index")
plt.ylabel("Distance")
sch.dendrogram(z, leaf_rotation=0, leaf_font_size=10)
plt.show()

# K-means clustering on the original dataset
TWSS = []
k = list(range(2, 8))

for i in k:
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)

# Scree plot for K-means clustering on the original dataset
plt.figure(figsize=(10, 6))
plt.plot(k, TWSS, 'ro-')
plt.title("Scree Plot for K-means Clustering (Original Dataset)")
plt.xlabel("Number of Clusters")
plt.ylabel("Total Within Sum of Squares (TWSS)")
plt.grid()
plt.show()

# Fit K-means with the optimal number of clusters
optimal_clusters = 3  # Adjust this based on your scree plot observation
model = KMeans(n_clusters=optimal_clusters, random_state=42)
model.fit(df_norm)
wine['clust'] = model.labels_
print(wine.head())

# PCA
wine_normal = scale(wine.iloc[:, :])  # Normalizing the data
pca = PCA(n_components=3)  # Extracting the first 3 components
pca_values = pca.fit_transform(wine_normal)

# Explained variance ratio
var = pca.explained_variance_ratio_
print(f'Explained Variance Ratio: {var}')

# Creating a DataFrame for PCA values
pca_df = pd.DataFrame(pca_values, columns=[f'PC{i+1}' for i in range(3)])
pca_df.to_csv("C:/Assignments(DS)/pca_wine_data.csv", index=False, encoding='utf-8')
print(pca_df.head())

# Hierarchical clustering on the PCA dataset
z_pca = linkage(pca_df, method="complete", metric="euclidean")
plt.figure(figsize=(15, 8))
plt.title("Hierarchical Clustering Dendrogram (PCA Dataset)")
plt.xlabel("Index")
plt.ylabel("Distance")
sch.dendrogram(z_pca, leaf_rotation=0, leaf_font_size=10)
plt.show()

# K-means clustering on the PCA dataset
TWSS_pca = []
for i in k:
    kmeans_pca = KMeans(n_clusters=i, random_state=42)
    kmeans_pca.fit(pca_df)
    TWSS_pca.append(kmeans_pca.inertia_)

# Scree plot for K-means clustering on the PCA dataset
plt.figure(figsize=(10, 6))
plt.plot(k, TWSS_pca, 'ro-')
plt.title("Scree Plot for K-means Clustering (PCA Dataset)")
plt.xlabel("Number of Clusters")
plt.ylabel("Total Within Sum of Squares (TWSS)")
plt.grid()
plt.show()

# Fit K-means with the optimal number of clusters for PCA dataset
model_pca = KMeans(n_clusters=optimal_clusters, random_state=42)
model_pca.fit(pca_df)
pca_df['clust'] = model_pca.labels_
print(pca_df.head())

# Comparing results
print("Cluster labels from original dataset:")
print(wine['clust'].value_counts())

print("\nCluster labels from PCA dataset:")
print(pca_df['clust'].value_counts())

"""
Benifits:
This problem helps improve data analysis by simplifying 
the dataset through PCA, allowing for easier interpretation and 
clustering. It ensures the robustness of clustering results by 
comparing outcomes with and without PCA, optimizes clustering performance 
using the scree plot, and uncovers hidden patterns in the data for better decision-making.
"""