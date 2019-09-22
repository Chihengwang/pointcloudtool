from PIL import Image
import open3d as o3d
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
# ===================================================
# Camera config
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
# =================================================
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

# ==========================================================

def show_ply_file(dirname,filename):
    pcd = o3d.io.read_point_cloud(dirname+"/"+filename)
    o3d.visualization.draw_geometries([pcd])

# 從ply 檔案得到所有點
def get_ply_file_points(dirname,filename):
    pcd = o3d.io.read_point_cloud(dirname+"/"+filename)
    # print(pcd)
    points=np.asarray(pcd.points)
    return points


# ===========================================================
# Downsizing function list
"""
這邊提供兩種方式一種是voxel的方式 voxel size越大代表取的量越少
參數method傳入字典的格式:
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
    if len(function)<3:
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

    if(function['method']=='voxel'):
        down_pcd = cloud.voxel_down_sample(voxel_size=function['voxel_size'])
    elif(function['method']=='uniform'):
        down_pcd = cloud.uniform_down_sample(every_k_points=function['every_k_points'])
    else:
        raise SystemExit('Make sure the name of method is correct!!!')
    return down_pcd
# ===========================================================
# Outlier removal function
"""
Statistical outlier removal
Radius outlier removal
注意這邊傳入的是point cloud的形式 is_show參數選擇要不要看差異
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

def point_cloud_outlier_removal(cloud,is_show=False,function={}):
    if len(function)==0:
        raise SystemExit('You should pass the third parameter as dict')

    if(function['method']=='statistical'):
        cl, ind = cloud.remove_statistical_outlier(nb_neighbors=20,std_ratio=2.0)
    elif(function['method']=='radius'):
        cl, ind = cloud.remove_radius_outlier(nb_points=16, radius=0.05)

    pc_after_removal=display_inlier_outlier(cloud, ind,is_show)
    return pc_after_removal
# ===========================================================
if __name__ == "__main__":
    # 當前目錄就只要傳入'.'即可
    # show_ply_file('dataset','output.ply')

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
    ply_file='./dataset/output2.ply'
    points=color_and_depth_to_ply(rgb_file,depth_file,camera)
    savePoints_to_ply('.','test.ply',points)
    show_ply_file('.','test.ply')

    # 如何使用down sample
    # function={
    #     'method':'weef',
    #     'every_k_points':5
    # }
    # down_pcd=point_cloud_down_sample('.','output.ply',function=function)
    # o3d.visualization.draw_geometries([down_pcd])

    # 如何使用removal
    # function={
    #     'method':'statistical',
    #     'nb_neighbors':20,
    #     'std_ratio':2.0
    # }
    # pcd = o3d.io.read_point_cloud('output.ply')
    # pc_after_removal=point_cloud_outlier_removal(pcd,function=function)
    # o3d.visualization.draw_geometries([pc_after_removal])