"""
這個檔案最主要是藉由ModelNet40的資料庫，去產生出Partial point cloud
藉由ax+by+cz+d=0的平面方程式去當成切面
當中，我令normal vector的z需要朝下且
"""

import provider
import os
import sys
import numpy as np
import msvcrt
from point_cloud_function import cal_pca,show_centriod,furthest_point_sampling,R,normalize_point_cloud,get_centroid_from_pc,plt,R
def rotate_point_cloud_through_x_negative90(point_cloud):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          Nx3 array, point clouds
        Return:
          Nx3 array, point clouds
    """
    rotated_data = np.zeros(point_cloud.shape, dtype=np.float32)
    rotation_angle = -np.pi/2
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[1,0, 0 ],
                                [0, cosval, -sinval],
                                [0, sinval, cosval]])
    rotated_data = np.dot(point_cloud.reshape((-1, 3)), rotation_matrix)
    return rotated_data
# 給非特定軸的旋轉
def rotate_perturbation_point_cloud(pointcloud, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(pointcloud.shape, dtype=np.float32)
    angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
    Rx = np.array([[1,0,0],
                    [0,np.cos(angles[0]),-np.sin(angles[0])],
                    [0,np.sin(angles[0]),np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                    [0,1,0],
                    [-np.sin(angles[1]),0,np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                    [np.sin(angles[2]),np.cos(angles[2]),0],
                    [0,0,1]])
    R = np.dot(Rz, np.dot(Ry,Rx))
    rotated_data = np.dot(pointcloud.reshape((-1, 3)), R)
    return rotated_data

def switch_pca_yz_axis(pca_vec,covarience):
    # 避免list 被copy 位址
    tmp_vec=np.copy(pca_vec[1,:])
    pca_vec[1,:]=np.copy(pca_vec[2,:])
    pca_vec[2,:]=tmp_vec
    tmp_covarience=covarience[1]
    covarience[1]=covarience[2]
    covarience[2]=tmp_covarience
    # 再次確認X軸
    print(pca_vec)
    x_axis_matrix=np.outer(pca_vec[1,:],pca_vec[2,:])
    x_axis=np.asarray([x_axis_matrix[1,2]-x_axis_matrix[2,1],x_axis_matrix[2,0]-x_axis_matrix[0,2],x_axis_matrix[0,1]-x_axis_matrix[1,0]])
    pca_vec[0,:]=x_axis
    return pca_vec,covarience


def draw_switch_result(pointcloud,pca_axis,covarience,title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(title)
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1,1)
    ax.set_zlim3d(-1,1)
    ax.scatter(pointcloud[:,0], pointcloud[:,1], pointcloud[:,2], c='y',s=1)
    xm,ym,zm=get_centroid_from_pc(pointcloud)
    ax.scatter(xm, ym, zm, c='r',s=10)
    discount=1
    for length, vector in zip(covarience,pca_axis):
        ax.quiver(xm,ym,zm,vector[0],vector[1],vector[2], length=discount)
        discount/=3
    plt.show()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
# ===========================================================================
# 看是否要不要旋轉這個點雲
ROTATING_PERTURBATION_MODE=False
# 看要查看整體還非整體的點雲
PARTIAL_MODE=True
# 儲存幾組Partial point cloud
LIST_NUMBER_OF_POINTCLOUD_SET=5
# FPS參數：
NUMBER_OF_POINTS=1024
# 切割面的閥值：
CUTTING_THRESHOLD=0.05
# 對稱閥值 小於為對稱 大於為非對稱
SYMMETRIC_THRESHOLD=0.05
# 啟動旋轉模式：可能需要倒下來(保持原狀)或者站立著(True)
ROTATING_MODE=True
# 儲存的路徑與名字
SAVED_FILE_DIR=os.path.join(BASE_DIR,'data/modelnet40_ply_hdf5_1024_pose')
FILE_NAME='ply_data_train0_pose.h5'   #這個隨時都可以改
SAVE_FILES_PATH=os.path.join(SAVED_FILE_DIR,FILE_NAME)
if not os.path.exists(SAVED_FILE_DIR):
    os.mkdir(SAVED_FILE_DIR)

print("目前儲存的檔案名稱路徑為："+SAVE_FILES_PATH)
print("-"*30)

# ===========================================================================

# ModelNet40 official train/test split
TRAIN_FILES = provider.getDataFiles( \
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
# print(TRAIN_FILES)

# Shuffle train files
train_file_idxs = np.arange(0, len(TRAIN_FILES))
np.random.shuffle(train_file_idxs)
# print(train_file_idxs)
SAVE_PARTIAL_PCS_LIST=[]
MODE_LIST=[]
QUATERNION_LABEL_LIST=[]

# 存到2048組點雲組才停止
while(len(SAVE_PARTIAL_PCS_LIST)<LIST_NUMBER_OF_POINTCLOUD_SET):
    for fn in range(len(TRAIN_FILES)):
        if(len(SAVE_PARTIAL_PCS_LIST)==LIST_NUMBER_OF_POINTCLOUD_SET):
            break
        print("目前讀取的檔案是： %s" %(TRAIN_FILES[train_file_idxs[fn]],))
        current_data, current_label = provider.loadDataFile(TRAIN_FILES[train_file_idxs[fn]])
        current_data, current_label, _ = provider.shuffle_data(current_data, np.squeeze(current_label))
        # print(current_label)
        current_data = current_data[:,:,:]
        # label:(2048,1)
        # print(current_data.shape)
        
        for i in range(len(current_data)):
            print("目前已經取得 %d 組點雲" %(len(SAVE_PARTIAL_PCS_LIST),))
            """
            這邊規定normal vector的z要為負，使內積出來小於零的點會出現在平面之上。
            這樣也代表我們觀察的點是以平面以上的partial point cloud為主。
            參考公式：
            ax+by+cz=0;
            (a,b,c)是我們隨機產生的normal vector, 其z我們規定要小於零
            (x,y,z)為 點雲模型中的座標
            藉此切割出partial point cloud
            """
            
            if(len(SAVE_PARTIAL_PCS_LIST)==LIST_NUMBER_OF_POINTCLOUD_SET):
                break
            if(ROTATING_MODE):
                current_data[i]=rotate_point_cloud_through_x_negative90(current_data[i]) #轉到z朝上
            show_centriod(current_data[i],"original point cloud")
            cal_pca(current_data[i],False,title="original point cloud")
            # normalized the random normal vector(x,y,z) 
            # 盡量讓normal vector可以是我眼睛看出去的方向，所以才規定這些參數
            rand_normal_vec_X=np.random.uniform(0, 1, 1)
            rand_normal_vec_Z=np.random.uniform(-1, -0.3, 1)
            rand_normal_vec_Y=np.random.uniform(-1, 1, 1)
            rand_normal_vec=np.hstack((rand_normal_vec_X,rand_normal_vec_Y,rand_normal_vec_Z))
            normalized_vec=rand_normal_vec/np.linalg.norm(rand_normal_vec)
            print("Normal vector： %s" %(normalized_vec,))

            inner_outcome=np.inner(current_data[i][:],normalized_vec) # 對每個點內積
            partial_point_cloud=current_data[i][np.where(inner_outcome-CUTTING_THRESHOLD<0)] #挑出那些點小於零
            print("取出來的partial point cloud數量： %d" %(partial_point_cloud.shape[0],))
            if(partial_point_cloud.shape[0]<NUMBER_OF_POINTS):
                continue
            # 資料前處理
            partial_point_cloud=normalize_point_cloud(partial_point_cloud)
            partial_point_cloud=furthest_point_sampling(partial_point_cloud,NUMBER_OF_POINTS)
            # partial_point_cloud=rotate_perturbation_point_cloud(partial_point_cloud)
            show_centriod(partial_point_cloud,"partial point cloud")
            pca_vec,covarience=cal_pca(partial_point_cloud,False,title="partial point cloud")
            print("-"*30)
            print("PCA的covarience為：")
            print(covarience)
            print(pca_vec)
            # 對稱的圖建議用三指抓，非對稱可以考慮用兩指即可
            if( covarience[0] - covarience[1] < SYMMETRIC_THRESHOLD):
                print("-"*30)
                print("偏向對稱圖，其第一二軸var差距：%f" %(covarience[0]-covarience[1],))
                print("-"*30)
                pca_vec,covarience=cal_pca(partial_point_cloud,True,title="Symmetric point cloud")
                # draw_switch_result(partial_point_cloud,pca_vec,covarience,title="Symmetric point cloud")
            else:
                print("-"*30)
                print("非對稱圖，其第一二軸var差距：%f" %(covarience[0]-covarience[1],))
                print("-"*30)
                pca_vec,covarience=cal_pca(partial_point_cloud,True,title="Asymmetric point cloud")
                # draw_switch_result(partial_point_cloud,pca_vec,covarience,title="Non-symmetric point cloud")
            print(pca_vec)
            print("-"*30)
            print("Press 'C' to rotate pca-axis")
            print("Press 'D' to leave rotation mode")
            while True:
                # print(ord(msvcrt.getch()))
                press_btn=ord(msvcrt.getch())
                if press_btn in [67, 99]:
                    axis=str(input("Enter an axis:(x,y or z)預設為x: "))
                    angle=input("Enter an angle:(預設90度) ")
                    if(not axis):
                        axis='x'
                    if(not angle):
                        angle='90'
                    angle=int(angle)
                    print("-"*30)
                    print("轉換前： %s" %(pca_vec,))
                    r = R.from_euler(axis,angle, degrees=True)
                    print(r.as_dcm())
                    r3=np.dot(pca_vec.transpose(),r.as_dcm())
                    pca_vec=r3.transpose()
                    print("轉換後： %s" %(pca_vec,))
                    draw_switch_result(partial_point_cloud,pca_vec,covarience,title="rotation result")
                    print("-"*30)
                    print("Press 'C' to rotate pca-axis")
                    print("Press 'D' to leave rotation mode")
                elif press_btn in [68, 100]:

                    break
            rotation_matrix=pca_vec.T
            quaternion= R.from_dcm(rotation_matrix).as_quat()
            print("-"*30)
            print("Press 'C' to save as two-finger mode...")
            print("Press 'D' to save as three-finger mode...")
            print("Press 'E' to skip...")
            while True:
                # print(ord(msvcrt.getch()))
                press_btn=ord(msvcrt.getch())
                if press_btn in [67, 99]:
                    SAVE_PARTIAL_PCS_LIST.append(partial_point_cloud.tolist())
                    MODE_LIST.append(0)
                    QUATERNION_LABEL_LIST.append(quaternion)
                    print("save as two-finger mode...")
                    break
                elif press_btn in [68, 100]:
                    SAVE_PARTIAL_PCS_LIST.append(partial_point_cloud.tolist())
                    MODE_LIST.append(1)
                    QUATERNION_LABEL_LIST.append(quaternion)
                    print("save as three-finger mode...")
                    break
                elif press_btn in [69, 101]:
                    print("skip this one...")
                    break
# convert all data to numpy array
SAVE_PARTIAL_PCS_LIST=np.asarray(SAVE_PARTIAL_PCS_LIST)
QUATERNION_LABEL_LIST=np.asarray(QUATERNION_LABEL_LIST)
MODE_LIST=np.asarray(MODE_LIST)
print("目前point cloud的size為：%s" %(SAVE_PARTIAL_PCS_LIST.shape,))
print("目前finger mode的size為：%s" %(MODE_LIST.shape,))
print("目前pose label的size為：%s" %(QUATERNION_LABEL_LIST.shape,))
