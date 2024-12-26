# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 10:28:29 2024

@author: om
"""
"""
Problem Statement: -

A pharmaceuticals manufacturing company is conducting 
a study on a new medicine to treat heart diseases
. The company has gathered data from its secondary 
sources and would like you to provide high level 
analytical insights on the data. Its aim is to segregate
 patients depending on their age group and other factors 
 given in the data. Perform PCA and clustering algorithms
 on the dataset and check if the clusters formed before 
 and after PCA are the same and provide a brief report 
 on your model. You can also explore more ways to improve your model. 
"""

"""
Business problem:
The business problem is to analyze patient data to identify 
meaningful patterns and segments, enabling the pharmaceutical 
company to effectively target specific age groups and 
characteristics for the new heart disease treatment. By 
leveraging PCA and clustering techniques, the company aims to 
optimize its marketing strategies, improve treatment personalization, 
and allocate resources efficiently, ultimately enhancing the success of the new medication 
in addressing patient needs.

"""

import pandas as pd
import matplotlib.pylab as plt

heart=pd.read_csv("C:/Assignments(DS)/heart disease.csv")
heart.head()
heart.info()
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

df_norm=norm_func(heart.iloc[:,:])
df_norm

b=df_norm.describe()
b

#hierarchical clustering
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

z=linkage(df_norm,method="complete",metric="euclidean")
plt.figure(figsize=(15,8));
plt.title("Hierachical Clustering dendrogram");
plt.xlabel("Index");
plt.ylabel("Distance")
sch.dendrogram(z,leaf_rotation=0,leaf_font_size=10)
plt.show()

#kmeans
from sklearn.cluster import KMeans
TWSS=[]
k=list(range(2,8))
k
for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS

# Scree plot to visualize the elbow point
plt.figure(figsize=(10, 6))
plt.plot(k, TWSS, 'ro-')
plt.title("Scree Plot for K-means Clustering")
plt.xlabel("Number of Clusters")
plt.ylabel("Total Within Sum of Squares (TWSS)")
plt.grid()
plt.show()




# Based on the scree plot, let's assume the optimal number of clusters is 3 
model=KMeans(n_clusters=3)
model.fit(df_norm)


# Assigning cluster labels to the original data
heart['clust'] = pd.Series(model.labels_)
heart.head()


#PCA


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
import pandas as pd

# Assuming 'wine' is already loaded
heart.data = heart.iloc[:, :]  # Using all columns

# Normalizing numerical data
heart_normal = scale(heart.data)


# Performing PCA with 3 components
pca = PCA(n_components=3)
pca_values = pca.fit_transform(heart_normal)

# Explained variance ratio (variance captured by each component)
var = pca.explained_variance_ratio_

# Creating DataFrames for PCA values and explained variance
pca_df = pd.DataFrame(pca_values, columns=[f'PC{i+1}' for i in range(3)])
variance_df = pd.DataFrame(var, columns=['Explained_Variance'])

# Saving PCA values to a CSV file
pca_df.to_csv("C:/Assignments(DS)/pca_heart_diseases_data.csv", index=False, encoding='utf-8')

# Displaying the first few rows of the PCA data
pca_df.head()



import pandas as pd
import matplotlib.pylab as plt
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
from sklearn.cluster import KMeans

# Load the dataset
wine = pd.read_csv("C:/Assignments(DS)/pca_heart_diseases_data.csv")

# Normalization function
def norm_func(i):
    x = (i - i.min()) / (i.max() - i.min())
    return x

# Normalize the dataset
df_norm = norm_func(wine.iloc[:, :])

# Hierarchical clustering
z = linkage(df_norm, method="complete", metric="euclidean")
plt.figure(figsize=(15, 8))
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Index")
plt.ylabel("Distance")
sch.dendrogram(z, leaf_rotation=0, leaf_font_size=10)
plt.show()

# K-means clustering to find the optimum number of clusters
TWSS = []
k = list(range(2, 8))

for i in k:
    kmeans = KMeans(n_clusters=i, random_state=42)  # Added random_state for reproducibility
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)

# Scree plot
plt.figure(figsize=(10, 6))
plt.plot(k, TWSS, 'ro-')
plt.title("Scree Plot for K-means Clustering")
plt.xlabel("Number of Clusters")
plt.ylabel("Total Within Sum of Squares (TWSS)")
plt.grid()
plt.show()

# Fit K-means with the optimal number of clusters for PCA dataset
model_pca = KMeans(n_clusters=3, random_state=42)
model_pca.fit(pca_df)
pca_df['clust'] = model_pca.labels_
print(pca_df.head())

# Comparing results
print("Cluster labels from original dataset:")
print(heart['clust'].value_counts())

print("\nCluster labels from PCA dataset:")
print(pca_df['clust'].value_counts())

"""
benifits:
    This problem enables the pharmaceutical company to identify 
    distinct patient groups for targeted treatments, streamline 
    data analysis using PCA, and enhance clustering accuracy. It
    helps optimize resource allocation and supports data-driven 
    decision-making to improve drug development and patient care
    outcomes.
"""