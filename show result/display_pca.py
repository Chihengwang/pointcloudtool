import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.decomposition import PCA
from point_cloud_function import *
from mpl_toolkits.mplot3d import Axes3D
"""
這個文件主要是用來測試四種point cloud的資料前處理會不會對
point cloud的principal axes造成影響，假如會的話 就很容易
影響後面用nn去近似時可能會造成的誤差。
"""
# =============================
# Testing data
np.random.seed(2)
noise=np.random.rand(1000,1)
X_=np.random.rand(1000,1)
Y=X_
# Z=1-X_-Y
Z=1-X_-Y+noise
X=np.hstack((X_,Y,Z))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


ax.set_xlabel('X Label(unit:m)')
ax.set_ylabel('Y Label(unit:m)')
ax.set_zlabel('Z Label(unit:m)')
plt.title("Z=1-X-Y(after adding noise)")
# print(X)
pca = PCA(n_components=3)
pca.fit(X)
ax.scatter(X[:,0], X[:,1], X[:,2], c='y',s=1)
print("Principal vectors:\n ",pca.components_)
print("Singular values:\n ",pca.explained_variance_)

xm,ym,zm=get_centroid_from_pc(X)
ax.scatter(xm, ym, zm, c='r',s=10)
discount=1
for length, vector in zip(pca.explained_variance_, pca.components_):
    ax.quiver(xm,ym,zm,vector[0],vector[1],vector[2], length=0.5*discount)
    discount/=2

plt.show()
# =============================
# show the comparison between partial point cloud
# X=get_ply_file_points('.','banana.ply')
# fig = plt.figure()
# ax = fig.add_subplot(221, projection='3d')


# ax.set_xlabel('X Label(unit:m)')
# ax.set_ylabel('Y Label(unit:m)')
# ax.set_zlabel('Z Label(unit:m)')
# plt.title("pca demo")
# # print(X)
# pca = PCA(n_components=3)
# pca.fit(X)
# ax.scatter(X[:,0], X[:,1], X[:,2], c='y',s=1)
# print("Principal vectors:\n ",pca.components_)
# print("Singular values:\n ",pca.explained_variance_)

# xm,ym,zm=get_centroid_from_pc(X)
# ax.scatter(xm, ym, zm, c='r',s=10)
# discount=1
# for length, vector in zip(pca.explained_variance_, pca.components_):
#     ax.quiver(xm,ym,zm,vector[0],vector[1],vector[2], length=0.2, normalize=True)

# # ==================================================
# # pca with norm
# ax = fig.add_subplot(222, projection='3d')
# # after normalize
# pc_norm=normalize_point_cloud(X.copy())
# pca2 = PCA(n_components=3)
# pca2.fit(pc_norm)
# ax.scatter(pc_norm[:,0], pc_norm[:,1], pc_norm[:,2], c='y',s=1)
# plt.title("pca_norm demo")
# print("Principal vectors:\n ",pca2.components_)
# print("Singular values:\n ",pca2.explained_variance_)
# xm,ym,zm=get_centroid_from_pc(pc_norm)
# ax.scatter(xm, ym, zm, c='r',s=10)
# for length, vector in zip(pca2.explained_variance_, pca2.components_):

#     ax.quiver(xm,ym,zm,vector[0],vector[1],vector[2], length=0.2, normalize=True)
# # plt.show()

# # ==================================================
# # pca with sample
# function={
#     'method':'uniform',
#     'every_k_points':8
# }
# pc_sample=point_cloud_down_sample_from_pc(X.copy(),function)
# pc_sample=np.asarray(pc_sample.points)
# ax = fig.add_subplot(223, projection='3d')

# pca3 = PCA(n_components=3)
# pca3.fit(pc_sample)
# ax.scatter(pc_sample[:,0], pc_sample[:,1], pc_sample[:,2], c='y',s=1)
# plt.title("pca_sample demo")
# print("Principal vectors:\n ",pca3.components_)
# print("Singular values:\n ",pca3.explained_variance_)
# xm,ym,zm=get_centroid_from_pc(pc_sample)
# ax.scatter(xm, ym, zm, c='r',s=10)
# for length, vector in zip(pca3.explained_variance_, pca3.components_):
#     ax.quiver(xm,ym,zm,vector[0],vector[1],vector[2], length=0.02, normalize=True)

# # ==================================================
# # pca with outlier removal
# function={
#     'method':'statistical',
#     'nb_neighbors':10,
#     'std_ratio':0.005
# }
# pc_removed=point_cloud_outlier_removal(X.copy(),function=function)
# ax = fig.add_subplot(224, projection='3d')
# pc_removed=np.asarray(pc_removed.points)
# pca4 = PCA(n_components=3)
# pca4.fit(pc_removed)
# ax.scatter(pc_removed[:,0], pc_removed[:,1], pc_removed[:,2], c='y',s=1)
# plt.title("pca_removed demo")
# print("Principal vectors:\n ",pca4.components_)
# print("Singular values:\n ",pca4.explained_variance_)
# xm,ym,zm=get_centroid_from_pc(pc_removed)
# ax.scatter(xm, ym, zm, c='r',s=10)
# for length, vector in zip(pca4.explained_variance_, pca4.components_):
#     ax.quiver(xm,ym,zm,vector[0],vector[1],vector[2], length=0.02, normalize=True)

# plt.show()