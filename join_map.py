from scipy.spatial.transform import Rotation as R
import numpy as np
from PIL import Image
import time
# ==============================================
# 相機參數
cx = 325.5
cy = 253.5
fx = 518.0
fy = 519.0
scalingFactor = 1000.0
# ==============================================

ply_file='output.ply'
pose_file_name='pose.txt'
depth_list=[]
rgb_list=[]
for i in range(5):
    depth = Image.open('./depth/'+str(i+1)+'.pgm')
    rgb = Image.open('./color/'+str(i+1)+'.png')
    depth_list.append(depth)
    rgb_list.append(rgb)


## Open file
fp = open(pose_file_name, "r")
# 變數 lines 會儲存 filename.txt 的內容
lines = fp.readlines()
# close file
fp.close()

past=time.time()

pose_list=[]
# print content
# ===============================================
# 這邊主要將所有的quaterion轉成rotation matrix
for i in range(len(lines)):
    split_term=lines[i].split()
    # [x,y,z,qx,qy,qz,qw]
    # print(split_term)
    t_matrix=np.zeros((4,4))
    translation_vector=np.array(split_term[:3])
    r = R.from_quat(split_term[3:])
    r_matrix=r.as_dcm()
    t_matrix[0:3,0:3]=r_matrix
    t_matrix[0:3,3]=translation_vector.T
    t_matrix[3,:]=np.array([0,0,0,1])
    pose_list.append(t_matrix)

# ================================================
# 做座標轉換 轉至世界座標下面
points=[]
# print(len(pose_list))
for i in range(len(pose_list)):
    rgb = rgb_list[i]
    depth = depth_list[i]
    # 這邊的v=480 u=640
    for v in range(rgb.size[1]):
        for u in range(rgb.size[0]):
            color = rgb.getpixel((u,v))
            Z = depth.getpixel((u,v)) / scalingFactor
            if Z==0: continue
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            point_in_camera=np.array([X,Y,Z,1])
            world_point=pose_list[i].dot(point_in_camera)
            points.append("%f %f %f %d %d %d 0\n"%(world_point[0],world_point[1],world_point[2],color[0],color[1],color[2]))

# now=time.time()
# cal_time=now-past
# print(cal_time)
file = open(ply_file,"w")
file.write('''ply
    format ascii 1.0
    element vertex %d
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    property uchar alpha
    end_header
    %s
    '''%(len(points),"".join(points)))


