from PIL import Image
import open3d as o3d
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import cv2
from sklearn.decomposition import PCA
# ===================================================
# Camera config
# ===================================================
"""
之後在restful api要用global去讀這個值
"""
class RGBDCamera():
    def __init__(self, centerx,centery,focalx,focaly,scalingfactor):
        self.cx =centerx
        self.cy=centery
        self.fx=focalx
        self.fy=focaly
        self.scalingfactor=scalingfactor

# ====================================================
# 將color 跟 depth map轉成 points
# 將color mask depth 轉成 partial point cloud
# ====================================================
'''
這邊要記得傳入camera的config才能夠計算
傳入的file是指 filename
todo:
這邊可能會加入如何計算join map 可以參考join_map.py的文件
'''
def color_and_depth_to_ply(rgb_file,depth_file,camera):
    depth = Image.open(depth_file)
    rgb = Image.open(rgb_file)
    if rgb.size != depth.size:
        raise Exception("Color and depth image do not have the same resolution.")
    if rgb.mode != "RGB":
        raise Exception("Color image is not in RGB format")
    if depth.mode != "I":
        raise Exception("Depth image is not in intensity format")
    print(depth.size)

    points = []
    #if the size of photo is (480,640) in opencv, then its size would be (640,480) in PIL 
    for v in range(rgb.size[1]):
        for u in range(rgb.size[0]):
            color = rgb.getpixel((u,v))
            # scaling factor can transform the unit from minimeter to meter
            Z = depth.getpixel((u,v)) * camera.scalingfactor
            if Z==0: continue
            X = (u - camera.cx) * Z / camera.fx
            Y = (v - camera.cy) * Z / camera.fy
            points.append("%f %f %f %d %d %d 0\n"%(X,Y,Z,color[0],color[1],color[2]))
    return points
"""
Mask+depth+color to partial point cloud
讀入都是numpy array的值(因為是從realsense讀過來的)
這裡因為depth是從nparray過來的所以mode上會從原本的I變成是I;16 也因此跟上面純粹用Image.open的結果不一樣

return:
list(for saving)),nparray(for dealing with data)
"""
def mask_to_partial_pointcloud(color,depth,mask,camera):
    color_img = Image.fromarray(color.astype('uint8'), 'RGB')
    depth_img = Image.fromarray(depth)
    mask_img = Image.fromarray(mask)
    if color_img.size != depth_img.size:
        raise Exception("Color and depth image do not have the same resolution.")
    if color_img.mode != "RGB":
        raise Exception("Color image is not in RGB format")
    if depth_img.mode != "I;16":
        raise Exception("Depth image is not in intensity format")
    # gray image 0-255
    if mask_img.mode != "L":
        raise Exception("Depth image is not in gray image format")
    points = []
    xyz_points=[]
    #if the size of photo is (480,640) in opencv, then its size would be (640,480) in PIL 
    for v in range(color_img.size[1]):
        for u in range(color_img.size[0]):
            color = color_img.getpixel((u,v))
            # scaling factor can transform the unit from minimeter to meter
            Z = depth_img.getpixel((u,v)) * camera.scalingfactor
            # mask==0 或者z==0 都是略過
            if Z==0 or mask_img.getpixel((u,v))==0: continue
            X = (u - camera.cx) * Z / camera.fx
            Y = (v - camera.cy) * Z / camera.fy
            xyz_points.append([X,Y,Z])
            points.append("%f %f %f %d %d %d 0\n"%(X,Y,Z,color[0],color[1],color[2]))
    return points,np.array(xyz_points)
# ==========================================================
# Saving/read pc, basic manipulation to pc
# ==========================================================
"""
這裡可以寫一些簡單的數據處理 跟 處理顯示
"""
# 存points 變成點雲文件
def savePoints_to_ply(dirname,filename,points):
    file = open(dirname+'/'+filename,"w")
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
    file.close()
def show_ply_file(dirname,filename):
    pcd = o3d.io.read_point_cloud(dirname+"/"+filename)
    o3d.visualization.draw_geometries([pcd])

# 從ply 檔案得到所有點
def get_ply_file_points(dirname,filename):
    pcd = o3d.io.read_point_cloud(dirname+"/"+filename)
    # print(pcd)
    points=np.asarray(pcd.points)
    return points
# 上面的point cloud傳入的是numpy
# 這邊傳入純粹就是x,y,z的值
"""
input: numpy or list
output: x,y,z的平均
"""
def get_centroid_from_pc(points):
    if(type(points)==list):
        points=np.array(points)
        points_mean=np.mean(points,axis=0)
    else:
        points_mean=np.mean(points,axis=0)
    return points_mean[0],points_mean[1],points_mean[2]
"""
input:
xyz_points:numpy
"""
def show_centriod(xyz_points,title_name):
    x_mean,y_mean,z_mean=get_centroid_from_pc(xyz_points)
    ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
    #  将数据点分成三部分画，在颜色上有区分度
    ax.scatter(xyz_points[:,0], xyz_points[:,1], xyz_points[:,2], c='b',s=1)  # 绘制数据点
    ax.scatter(x_mean, y_mean, z_mean, c='r',s=10)
    ax.set_zlabel('Z')  # 坐标轴
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.title(title_name)
    plt.show()
"""
Normalizing the point cloud into the unit sphere
refer to pointnet
"""
def normalize_point_cloud(points):
    centroid = np.mean(points, axis=0)
    points -= centroid
    furthest_distance = np.max(np.sqrt(np.sum(abs(points)**2,axis=-1)))
    points /= furthest_distance
    return points
# ===========================================================
# Downsizing function list
# ===========================================================
"""
這邊提供兩種方式一種是voxel的方式 voxel size越大代表取的量越少
參數method傳入字典的格式:
注意這邊傳入的是open3d point cloud的形式 is_show參數選擇要不要看差異
假如是要傳入numpy形式 要先轉成open3d的形式

input:
numpy or open3d point cloud
output:
open3d::pointcloud

ex:
function={
    'method':'voxel',
    'voxel_size':0.02
}
or
function={
    'method':'uniform',
    'every_k_points':5
}
"""
def point_cloud_down_sample_from_file(dirname,filename,function={}):
    if len(function)<2:
        raise SystemExit('You should pass the third parameter as dict')
    pcd = o3d.io.read_point_cloud(dirname+"/"+filename)
    if(function['method']=='voxel'):
        down_pcd = pcd.voxel_down_sample(voxel_size=function['voxel_size'])
    elif(function['method']=='uniform'):
        down_pcd = pcd.uniform_down_sample(every_k_points=function['every_k_points'])
    else:
        raise SystemExit('Make sure the name of method is correct!!!')
    return down_pcd
def point_cloud_down_sample_from_pc(cloud,function={}):
    if len(function)<2:
        raise SystemExit('You should pass the third parameter as dict')
    if(type(cloud).__module__==np.__name__):
        print("你傳入numpy array形式")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud)
        if(function['method']=='voxel'):
            down_pcd = pcd.voxel_down_sample(voxel_size=function['voxel_size'])
        elif(function['method']=='uniform'):
            down_pcd = pcd.uniform_down_sample(every_k_points=function['every_k_points'])
        else:
            raise SystemExit('Make sure the name of method is correct!!!')
    else:
        if(function['method']=='voxel'):
            down_pcd = cloud.voxel_down_sample(voxel_size=function['voxel_size'])
        elif(function['method']=='uniform'):
            down_pcd = cloud.uniform_down_sample(every_k_points=function['every_k_points'])
    return down_pcd
# ===========================================================
# Outlier removal function
# ===========================================================
"""
Statistical outlier removal
Radius outlier removal
注意這邊傳入的是open3d point cloud的形式 is_show參數選擇要不要看差異
假如是要傳入numpy形式 要先轉成open3d的形式
ex:
function={
    'method':'statistical',
    'nb_neighbors':20,
    'std_ratio':2.0
}
or
function={
    'method':'radius',
    'nb_points':16,
    'radius':0.05
}
"""
def display_inlier_outlier(cloud, ind,is_show):
    inlier_cloud = cloud.select_down_sample(ind)
    outlier_cloud = cloud.select_down_sample(ind, invert=True)

    if(is_show):
        outlier_cloud.paint_uniform_color([1, 0, 0])
        inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
        o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
    return inlier_cloud
"""
這邊的removal 用到的是open3d裡面的outlier removal 所以
return 的形式會是pcd的object
因此要把點取出來的話要用np.asarray(pc_after_removal.points)
將points的array用np的方式取出來

output:
open3d:pointcloud
"""
def point_cloud_outlier_removal(cloud,is_show=False,function={}):
    if len(function)==0:
        raise SystemExit('You should pass the third parameter as dict')
    if(type(cloud).__module__==np.__name__):
        print("你傳入numpy array形式")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud)      
        if(function['method']=='statistical'):
            cl, ind = pcd.remove_statistical_outlier(nb_neighbors=function['nb_neighbors'],std_ratio=function['std_ratio'])
        elif(function['method']=='radius'):
            cl, ind = pcd.remove_radius_outlier(nb_points=function['nb_points'], radius=function['radius'])

        pc_after_removal=display_inlier_outlier(pcd, ind,is_show)
    # 傳入open3d 的point cloud 形式
    else:
        if(function['method']=='statistical'):
            cl, ind = cloud.remove_statistical_outlier(nb_neighbors=function['nb_neighbors'],std_ratio=function['std_ratio'])
        elif(function['method']=='radius'):
            cl, ind = cloud.remove_radius_outlier(nb_points=function['nb_points'], radius=function['radius'])
        pc_after_removal=display_inlier_outlier(cloud, ind,is_show)
    return pc_after_removal
# ===========================================================
# The tool to analyze the shape of the point cloud
"""
cal_pca's parameters
point_cloud: numpy array type
is_show: initially False to display the result of the point cloud and show the vectors in figure.
desired_num_of_feature: we set up as three because we analyze the shape of point cloud,
which contains three dimensions (x,y,z).

return:
first: principal axes of pointcloud
second: 
singular values.
"""
def cal_pca(point_cloud,is_show=False,desired_num_of_feature=3):
    pca = PCA(n_components=desired_num_of_feature)
    pca.fit(point_cloud)
    # print("Principal vectors: ",pca.components_)
    # print("Singular values: ",pca.explained_variance_)
    if is_show:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X Label(unit:m)')
        ax.set_ylabel('Y Label(unit:m)')
        ax.set_zlabel('Z Label(unit:m)')
        plt.title("pca demo")
        ax.scatter(point_cloud[:,0], point_cloud[:,1], point_cloud[:,2], c='y',s=1)
        xm,ym,zm=get_centroid_from_pc(point_cloud)
        ax.scatter(xm, ym, zm, c='r',s=10)
        discount=1
        for length, vector in zip(pca.explained_variance_, pca.components_):
            ax.quiver(xm,ym,zm,vector[0],vector[1],vector[2], length=0.05*discount)
            discount/=2
        plt.show()
    return pca.components_,pca.explained_variance_
# ===========================================================
if __name__ == "__main__":
    # 當前目錄就只要傳入'.'即可
    # show_ply_file('.','banana_wo_partial.ply')
    # show_ply_file('.','banana.ply')
    # show_ply_file('.','output.ply')
    # 如何將registed pair 轉成 點雲形式並儲存
    # depth map colorimg to point clouds
    cx = 316.001
    cy = 244.572
    fx = 616.391
    fy = 616.819
    scalingfactor = 0.0010000000474974513
    camera=RGBDCamera(cx,cy,fx,fy,scalingfactor)
    depth_file='./dataset/depth.png'
    rgb_file='./dataset/color.png'

    points=color_and_depth_to_ply(rgb_file,depth_file,camera)
    savePoints_to_ply('.','banana_wo_partial.ply',points)
    # show_ply_file('.','banana_wo_partial.ply')
    # 如何使用down sample
    # function={
    #     'method':'uniform',
    #     'every_k_points':8
    # }
    # down_pcd=point_cloud_down_sample_from_file('.','banana.ply',function=function)
    # o3d.visualization.draw_geometries([down_pcd])

    # 如何使用removal
    # function={
    #     'method':'statistical',
    #     'nb_neighbors':5,
    #     'std_ratio':0.001
    # }

    # pc_after_removal=point_cloud_outlier_removal(down_pcd,function=function)
    # o3d.visualization.draw_geometries([pc_after_removal])

    # 2019/9/23 testing completed
    # 這邊先模擬讀入是numpy的數值 所以先用opencv讀入測試(如realsense stream讀入一樣)
    color=cv2.imread('./dataset/color.png')
    # brg to rgb
    color=color[...,::-1]
    depth=cv2.imread('./dataset/depth.png',cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    mask=cv2.imread('./dataset/mask.png',cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    # print(color.shape)
    # print(depth.shape)
    # print(mask.shape)
    # color_img = Image.fromarray(color.astype('uint8'), 'RGB')
    # depth_img = Image.fromarray(depth)
    # mask_img = Image.fromarray(mask)
    # print(color_img.size)
    # print(depth_img.size)
    # print(mask_img.size)
    """
    color_img.size==depth_img.size==mask_img.size==(640,480)
    這邊的color藉由get_pixel可以得到裡面3個channel的值所以size那樣是正常的
    """
    points,xyz_points=mask_to_partial_pointcloud(color,depth,mask,camera)

    savePoints_to_ply('.','banana.ply',points)
    # show_ply_file('.','banana.ply')

    # 對partial point cloud做 remove outlier看效果(加上down sizing的效果)
    pcd = o3d.io.read_point_cloud('banana.ply')
    # print(type(pcd).__module__)
    # print(type(np.asarray(pcd.points)).__module__)
    function={
        'method':'uniform',
        'every_k_points':8
    }
    # down_pcd=point_cloud_down_sample_from_pc(pcd,function)
    # 測試傳入numpy
    down_pcd=point_cloud_down_sample_from_pc(np.asarray(pcd.points),function)
    print(down_pcd)
    function={
        'method':'statistical',
        'nb_neighbors':3,
        'std_ratio':0.01
    }
    pc_after_removal=point_cloud_outlier_removal(np.asarray(down_pcd.points),function=function)
    o3d.io.write_point_cloud("pc_after_removal.ply", pc_after_removal)
    print(pc_after_removal)
    # o3d.visualization.draw_geometries([pc_after_removal])
    # print(np.asarray(pc_after_removal.points))
    
    removed_pc=np.asarray(pc_after_removal.points)
    norm_removed_pc=normalize_point_cloud(removed_pc.copy())
    # show_centriod(np.asarray(down_pcd.points),'Sampling')
    # show_centriod(np.asarray(pc_after_removal.points),'Removal')
    # show_centriod(norm_removed_pc,'Normalization')
    
    # =================================================
    # test pca function
    is_show=True
    vectors,vals=cal_pca(removed_pc,is_show)
