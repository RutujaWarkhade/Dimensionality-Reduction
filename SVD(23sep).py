# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 08:15:13 2024

@author: om
"""
#SVD=singular value decomposition
import numpy as np
from numpy import array
from scipy.linalg import svd
A=array([[1,0,0,0,2],[0,0,0,0,0],[0,4,0,0,0]])
print(A)
#svd
U,d,Vt=svd(A)
print(U)
print(d)
print(Vt)
print(np.diag(d))
#svd applying to a dataset
import pandas as pd
data=pd.read_csv("C:/7-clustering/University_Clustering.csv")
data.head()
data=data.iloc[:,2:]
#remove non numeric data
data
from sklearn.decomposition import TruncatedSVD
svd=TruncatedSVD(n_components=3)
svd.fit(data)
result=pd.DataFrame(svd.transform(data))
result.head()
result.columns="pc0","pc1","pc2"
result.head()
#scatter diagram
import matplotlib.pylab as plt
plt.scatter(x=result.pc0,y=result.pc1)

"""
import matplotlib.pyplot as plt

plt.scatter(result["pc0"], result["pc1"])
plt.xlabel('PC0')
plt.ylabel('PC1')
plt.title('SVD Scatter Plot')
plt.show()

"""




