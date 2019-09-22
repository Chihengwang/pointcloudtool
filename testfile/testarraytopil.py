import numpy as np
from PIL import Image
import cv2

"""
這個文件是用來了解PIL與opencv怎麼做轉換的
"""
cx = 316.001
cy = 244.572
fx = 616.391
fy = 616.819
scalingFactor = 0.0010000000474974513
# ----------------------------------------
# 預設讀進來是bgr
color=cv2.imread('./dataset/color.png')
# bgr to rgb
color=color[...,::-1]
# opencv開啟原本的深度圖
depth=cv2.imread('./dataset/depth.png',cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
# 讀出來都是nparray
print(type(color))
print(type(depth))
# (480,640,3)
print(color.shape)
# (480,640)
print(depth.shape)

color_img = Image.fromarray(color.astype('uint8'), 'RGB')
depth_img = Image.fromarray(depth)
depth_img2 = Image.open('./dataset/depth.png')
# print(color_img.getpixel((1,1)))
# print(depth_img.getpixel((1,1)))
# print(depth_img2.getpixel((1,1)))
# print(depth_img2)


ply_file='second.ply'
points = []    
for v in range(color_img.size[1]):
    for u in range(color_img.size[0]):
        color = color_img.getpixel((u,v))
        # scaling factor can transform the unit from minimeter to meter
        Z = depth_img.getpixel((u,v)) * scalingFactor
        if Z==0: continue
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        points.append("%f %f %f %d %d %d 0\n"%(X,Y,Z,color[0],color[1],color[2]))
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
file.close()