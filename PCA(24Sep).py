# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 08:33:29 2024

@author: om
"""

import pandas as pd
import numpy as np
univ1=pd.read_csv("C:/7-clustering/University_Clustering.csv")
univ1.shape
univ1.describe()
univ1.info()
univ=univ1.drop(["State"],axis=1)
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
univ.data=univ.iloc[:,1:]
#normalizing numerical data
uni_normal=scale(univ.data)
uni_normal
pca=PCA(n_components=6)
pca_values=pca.fit_transform(uni_normal)
var=pca.explained_variance_ratio_
var

"""
#PCA weight
pca.components_
pca.components_[0]
#to check cumulative variance
var1=np.cumsum(np.round(var,decimals=4)*100)
var1
"""