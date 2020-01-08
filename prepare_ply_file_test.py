from point_cloud_function import *
from ra605.arm_kinematic import inverse_kinematic,forward_kinematic

dirname='./pointnet_data'
# 香蕉：06-12-2019-15-14-55.ply,02-01-2020-14-18-41
# 盒子：14-12-2019-14-28-51.ply
# 貧果：14-12-2019-14-56-32.ply,02-01-2020-14-38-31
# tape: 06-12-2019-15-29-40.ply,02-01-2020-14-29-18,02-01-2020-14-23-58
# 杯子：14-12-2019-14-33-52.ply,02-01-2020-14-55-34
filename='14-12-2019-14-56-32.ply'
show_ply_file(dirname,filename)
pcs=o3d.io.read_point_cloud(dirname+"/"+filename)
print("original------------------------")
print(np.asarray(pcs.points).shape)
origin_pcs_vec,_=cal_pca(np.asarray(pcs.points),is_show=False,title="origin")
print(origin_pcs_vec)
function={
    'method':'voxel',
    'voxel_size':0.0025
}
# function={
#     'method':'uniform',
#     'every_k_points':5
# }
print("down_sampling-------------------")
down_pcd=point_cloud_down_sample_from_pc(pcs,function)
print(np.asarray(down_pcd.points).shape)
down_pcd_vec,_=cal_pca(np.asarray(down_pcd.points),is_show=False,title="downpcd")
print(down_pcd_vec)
o3d.visualization.draw_geometries([down_pcd])

function={
    'method':'statistical',
    'nb_neighbors':20,
    'std_ratio':2.0
}
print("removed_pcd----------------------------------------")
remove_pcd=point_cloud_outlier_removal(down_pcd,is_show=False,function=function)
print(np.asarray(remove_pcd.points).shape)
# o3d.visualization.draw_geometries([remove_pcd])
# show_centriod(np.asarray(remove_pcd.points),"sampling")
vec_removed,_=cal_pca(np.asarray(remove_pcd.points),is_show=False,title="sampling")
print(vec_removed)
print("fps-----------------------------------------")
point_number=1024
fps_pcs=furthest_point_sampling(np.asarray(remove_pcd.points),point_number)
print(fps_pcs.shape)
# show_centriod(fps_pcs,"fps")
# 需要將單位從m換到mm
centroid=np.asarray(get_centroid_from_pc(fps_pcs))*1000
normed_pcs=normalize_point_cloud(fps_pcs)
# show_centriod(normed_pcs,"norm")
normed_vec,covarience=cal_pca(normed_pcs,is_show=True,title="Norm")
print("*"*30)
print("PCA的長度為:")
print(covarience)
print("*"*30)
print("PCA的向量為:")
print(normed_vec)
print(type(normed_vec))
print("*"*30)
print("中心點為:")
print(centroid)
# 將PCA出來的vector轉置成x,y,z的rotation matrix之後，把mean的x,y,z組成tf matrix帶入IK求解
t_matrix=np.zeros((4,4))
rotation_matrix=normed_vec.T
t_matrix[0:3,0:3]=rotation_matrix
t_matrix[0:3,3]=centroid.T
t_matrix[3,:]=np.array([0,0,0,1])
print("*"*30)
print("目標的tf matrix是:")
print(t_matrix)
print("*"*30)
print("ik計算:")
print(inverse_kinematic(t_matrix))
t_matrix_ik=forward_kinematic(inverse_kinematic(t_matrix))
print("*"*30)
print("tf矩陣是否相同：")
print(np.allclose(t_matrix,t_matrix_ik))
# ==================================
# 法蘭面+夾爪厚度+離物體的距離
D=5+35.1+100
D_list=np.asarray([0,0,D])
print(D_list)
# off_set=centroid-rotation_matrix.dot(D_list)
print(D_list.shape)
offset_matrix=np.zeros((4,4))
offset_matrix[0:3,0:3]=np.eye(3)
offset_matrix[0:3,3]=D_list.T
offset_matrix[3,:]=np.array([0,0,0,1])
ready_tf=t_matrix.dot(np.linalg.inv(offset_matrix))
print("-"*30)
print("Ready IK")
print(ready_tf)
print(inverse_kinematic(ready_tf))