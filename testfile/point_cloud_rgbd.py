from PIL import Image

# // 计算点云并拼接
# // 相机内参 
cx = 316.001
cy = 244.572
fx = 616.391
fy = 616.819
scalingFactor = 0.0010000000474974513
depth_file='./dataset/depth.png'
rgb_file='./dataset/color.png'
ply_file='./dataset/output.ply'
depth = Image.open(depth_file)
rgb = Image.open(rgb_file)
if rgb.size != depth.size:
    raise Exception("Color and depth image do not have the same resolution.")
if rgb.mode != "RGB":
    raise Exception("Color image is not in RGB format")
if depth.mode != "I":
    raise Exception("Depth image is not in intensity format")

points = []    
for v in range(rgb.size[1]):
    for u in range(rgb.size[0]):
        color = rgb.getpixel((u,v))
        # scaling factor can transform the unit from minimeter to meter
        Z = depth.getpixel((u,v)) * scalingFactor
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