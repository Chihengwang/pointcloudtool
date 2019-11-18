"""
這份文件再測試join map在ra605的情況
順便測試手眼校正的精準程度
會把點雲都轉到base座標下觀察！
"""
import ra605.arm_kinematic as ROBOT_KINE
import sys
import os
from point_cloud_function import *

# =======================================================
# Hand-eye-matrix
# 單位:meter
HAND_EYE_TFMATRIX=np.array([
    [0,-1,0,0.11865],
    [1,0,0,-0.035 ],
    [0,0,1,0.018],
    [0,0,0,1]
])
# HAND_EYE_TFMATRIX=np.array([
#     [-0.07026,-0.99653,0.044588,0.1192251],
#     [0.996989,-0.06868,0.035999,-0.0491859 ],
#     [0.046983,0,1,0.03143297],
#     [0,0,0,1]
# ])
print(HAND_EYE_TFMATRIX)
# =======================================================
cx = 316.001
cy = 244.572
fx = 616.391
fy = 616.819
scalingfactor = 0.0010000000474974513
camera=RGBDCamera(cx,cy,fx,fy,scalingfactor)
# =======================================================
pose_file_name="pose_ra605.txt"
ply_file="ra605_joinmap_with_mask.ply"
fp = open(pose_file_name, "r")
pose_list=[]
depth_list=[]
color_list=[]
mask_list=[]
# 變數 lines 會儲存 filename.txt 的內容
lines = fp.readlines()
# close file
fp.close()
# Join map裡面用米為單位 所以需要將translation vector/1000
for i in range(len(lines)):
    split_term=lines[i].split()
    six_dof_dic={
        'j1':float(split_term[0]),
        'j2':float(split_term[1]),
        'j3':float(split_term[2]),
        'j4':float(split_term[3]),
        'j5':float(split_term[4]),
        'j6':float(split_term[5]),
    }
    t0_6=ROBOT_KINE.forward_kinematic(six_dof_dic)
    t0_6[0:3,3]=t0_6[0:3,3]/1000
    print(t0_6)
    # T0_6*HAND_EYE_MATRIX
    T0_EYE=t0_6.dot(HAND_EYE_TFMATRIX)
    pose_list.append(T0_EYE)

# for i in range(len(pose_list)):
#     depth = Image.open('./dataset_for_cal_pos/depth/'+str(i+1)+'.png')
#     rgb = Image.open('./dataset_for_cal_pos/color/'+str(i+1)+'.png')
#     mask=Image.open('./dataset_for_cal_pos/mask/'+str(i+1)+'.png')
#     depth_list.append(depth)
#     color_list.append(rgb)
#     mask_list.append(mask)
# =================================================================================
# Dealing with join map without mask
# =================================================================================
# savePoints_to_ply('.',ply_file,join_map(pose_list,color_list,depth_list,camera))
# show_ply_file('.',ply_file)

# pcd = o3d.io.read_point_cloud(ply_file)
# function={
#     'method':'uniform',
#     'every_k_points':8
# }
# down_pcd=point_cloud_down_sample_from_pc(pcd,function)
# o3d.visualization.draw_geometries([down_pcd])
# # print(get_centroid_from_pc(down_pcd.points))
# function={
#     'method':'statistical',
#     'nb_neighbors':3,
#     'std_ratio':0.01
# }
# pc_after_removal=point_cloud_outlier_removal(down_pcd,function=function)
# print(np.asarray(down_pcd.points).shape)
# o3d.visualization.draw_geometries([pc_after_removal])
# print(np.asarray(pc_after_removal.points).shape)


# =================================================================================
# Dealing with join map with mask
# =================================================================================
"""
模擬一下從labview到restful api的拍照以及轉換檔案格式的過程==>因為color,depth,mask檔案格式皆為cv2格式
先從realsense讀取資料(讀color,depth照片)，藉由MaskRCNN轉換成mask的深度圖之後
分別存成COLOR,DEPTH,MASK 的list
直到list裡面包含兩組以上的資料，開始進行join map with mask的function 出partial point cloud
"""
for i in range(len(pose_list)):
    color=cv2.imread('./dataset_for_cal_pos/color/'+str(i+1)+'.png')
    # brg to rgb
    color=color[...,::-1]
    depth=cv2.imread('./dataset_for_cal_pos/depth/'+str(i+1)+'.png',cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    mask=cv2.imread('./dataset_for_cal_pos/mask/'+str(i+1)+'.png',cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    color_img = Image.fromarray(color.astype('uint8'), 'RGB')
    depth_img = Image.fromarray(depth)
    mask_img = Image.fromarray(mask)
    depth_list.append(depth_img)
    color_list.append(color_img)
    mask_list.append(mask_img)
points,xyz_list=join_map_with_mask(pose_list,color_list,depth_list,mask_list,camera)
savePoints_to_ply('.',ply_file,points)
show_ply_file('.',ply_file)
pcd = o3d.io.read_point_cloud(ply_file)
function={
    'method':'uniform',
    'every_k_points':8
}
down_pcd=point_cloud_down_sample_from_pc(pcd,function)
o3d.visualization.draw_geometries([down_pcd])
# print(get_centroid_from_pc(down_pcd.points))
function={
    'method':'statistical',
    'nb_neighbors':3,
    'std_ratio':0.01
}
pc_after_removal=point_cloud_outlier_removal(down_pcd,function=function)
print(np.asarray(down_pcd.points).shape)
o3d.visualization.draw_geometries([pc_after_removal])
print(np.asarray(pc_after_removal.points).shape)
K=2048
furthest_points=furthest_point_sampling(np.asarray(pc_after_removal.points), K)
print(furthest_points.shape)
print(get_centroid_from_pc(furthest_points))
show_centriod(furthest_points,'furthest')