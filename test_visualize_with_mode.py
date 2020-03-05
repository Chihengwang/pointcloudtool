from point_cloud_function import *
from ra605.arm_kinematic import inverse_kinematic,forward_kinematic

if __name__ == "__main__":
    dirname='./pointnet_data'
    # 香蕉：06-12-2019-15-14-55.ply,02-01-2020-14-18-41
    # 盒子：14-12-2019-14-28-51.ply,02-01-2020-13-40-29.ply
    # 貧果：14-12-2019-14-56-32.ply,02-01-2020-14-38-31
    # tape: 06-12-2019-15-29-40.ply,02-01-2020-14-29-18,02-01-2020-14-23-58
    # 杯子：14-12-2019-14-33-52.ply,02-01-2020-14-55-34
    filename='06-12-2019-15-29-40.ply'
    # show_ply_file(dirname,filename)
    pcs=o3d.io.read_point_cloud(dirname+"/"+filename)

    # do data pre-processing but don't use normalization function
    FINGER_MODE_LIST = ['two_finger_mode','three_finger_mode']
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
    point_number=256
    fps_pcs=furthest_point_sampling(np.asarray(remove_pcd.points),point_number)
    print(fps_pcs.shape)
    origin_pcs_vec,_=cal_pca(fps_pcs.copy(),is_show=False,title="")
    open_widht_length= open_width_algorithm(fps_pcs.copy(),origin_pcs_vec,isVisualize=False)
    print(open_widht_length)
    normalized_point=normalize_point_cloud(np.copy(fps_pcs))
    # visualize_q_pointnet(normalized_point,origin_pcs_vec,"Q-PointNet")
    
    visualize_q_pointnet_with_mode(fps_pcs,origin_pcs_vec,open_widht_length,FINGER_MODE_LIST[1],"")


