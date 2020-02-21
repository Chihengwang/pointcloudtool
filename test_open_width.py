from point_cloud_function import *


def open_width_algorithm(point_cloud,pca_axis,isVisualize=False,inner_product_threshold=0.97):
    """
    input point cloud should be the original data but already do the point cloud preprocessing
    (without normalization)
    input:
        point cloud: 1024X3
        pca_axis: [-pca_x-]
                [-pca_y-]
                [-pca_z-]
        type: np array
    return:
        length of gripper would open 
        (unit:mm)
        because unit of the point cloud is meter, we need to change it into mm.
    """
    # translate the point cloud but not divide the furthest length
    centroid = np.mean(point_cloud, axis=0)
    point_cloud -= centroid
    show_centriod(point_cloud,"")

    # pca_axis 是PCA坐標系到世界坐標系的旋轉矩陣
    new_point_cloud= np.dot(pca_axis,point_cloud.T)

    show_centriod(new_point_cloud.T,"")

    new_point_cloud=new_point_cloud.T*1000
    # mapping onto plane
    xm,ym,zm=get_centroid_from_pc(new_point_cloud)

    # normalize the point(change into normalized vector)
    length_of_norm=np.linalg.norm(np.copy(new_point_cloud[:,:2]),axis=1,keepdims=True)
    normalized_vector_pc=np.true_divide(np.copy(new_point_cloud[:,:2]),length_of_norm)

    pca_y_axis_after_transform=np.asarray([0,1]).reshape(2,1)
    # 內積計算 取最接近1,-1
    inner_product_result=np.dot(normalized_vector_pc,pca_y_axis_after_transform)
    filter_point_condition=np.where(np.abs(inner_product_result)>=inner_product_threshold)[0]
    filter_point=new_point_cloud[filter_point_condition,:].copy()

    # 只要考慮pca-y軸
    index_max=np.argmax(filter_point,axis=0)[1]
    index_min=np.argmin(filter_point,axis=0)[1]
    distance_vector=filter_point[index_max,:2]-filter_point[index_min,:2]
    distance=np.linalg.norm(distance_vector)
    if(isVisualize):
        plt.scatter(filter_point[:,0], filter_point[:,1], c='g',s=50)#繪製散佈圖
        plt.scatter(new_point_cloud[:,0], new_point_cloud[:,1],c='b',s=8)#繪製散佈圖
        plt.scatter(xm, ym, c='r',s=30)
        plt.scatter(filter_point[index_max,0], filter_point[index_max,1], c='r',s=30)
        plt.scatter(filter_point[index_min,0], filter_point[index_min,1], c='r',s=30)
        # plt.title("finger width should be:"+str(round(distance,2))+"mm")
        plt.show()

    return distance


if __name__ == "__main__":
    dirname='./pointnet_data'
    # 香蕉：06-12-2019-15-14-55.ply,02-01-2020-14-18-41
    # 盒子：14-12-2019-14-28-51.ply,02-01-2020-13-40-29.ply
    # 貧果：14-12-2019-14-56-32.ply,02-01-2020-14-38-31
    # tape: 06-12-2019-15-29-40.ply,02-01-2020-14-29-18,02-01-2020-14-23-58
    # 杯子：14-12-2019-14-33-52.ply,02-01-2020-14-55-34
    filename='14-12-2019-14-33-52.ply'
    # show_ply_file(dirname,filename)
    pcs=o3d.io.read_point_cloud(dirname+"/"+filename)

    # do data pre-processing but don't use normalization function

    function={
        'method':'voxel',
        'voxel_size':0.0025
    }
    print("down_sampling-------------------")
    down_pcd=point_cloud_down_sample_from_pc(pcs,function)
    print(np.asarray(down_pcd.points).shape)
    # o3d.visualization.draw_geometries([down_pcd])

    print("removed_pcd----------------------------------------")
    function={
        'method':'statistical',
        'nb_neighbors':20,
        'std_ratio':2.0
    }
    remove_pcd=point_cloud_outlier_removal(down_pcd,is_show=False,function=function)
    print(np.asarray(remove_pcd.points).shape)

    print("fps-----------------------------------------")
    point_number=1024
    fps_pcs=furthest_point_sampling(np.asarray(remove_pcd.points),point_number)
    print(fps_pcs.shape)
    # show_centriod(fps_pcs,"furthest point sampling")
    origin_pcs_vec,_=cal_pca(fps_pcs,is_show=False,title="")
    normalized_point=normalize_point_cloud(fps_pcs.copy())
    visualize_q_pointnet(normalized_point,origin_pcs_vec,"Q-PointNet")
    # translate the point cloud but not divide the furthest length
    # centroid = np.mean(fps_pcs, axis=0)
    # fps_pcs -= centroid
    # show_centriod(fps_pcs,"translation of point cloud")
    open_widht_length= open_width_algorithm(fps_pcs,origin_pcs_vec,isVisualize=True)
    print("-----------Gripper open width-----------")
    print("The width gripper should open: %f mm" %(open_widht_length))
    