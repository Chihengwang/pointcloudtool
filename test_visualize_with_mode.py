from point_cloud_function import *
from ra605.arm_kinematic import inverse_kinematic,forward_kinematic

def visualize_q_pointnet_with_mode(pointcloud,axes,width,mode,title):
    """
    input:
        point cloud 1024X3 
        axes: [-pca_x-]
                [-pca_y-]
                [-pca_z-]
        type: np array
        width: from open width algorithm
        mode: two_finger_mode or three_finger_mode
    return:
        length of gripper would open 
        (unit:mm)
        because unit of the point cloud is meter, we need to change it into mm.
    """
    width=width/1000+0.01 #0.01是誤差容許範圍 除以1000換成米
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X Label(unit:m)')
    ax.set_ylabel('Y Label(unit:m)')
    ax.set_zlabel('Z Label(unit:m)')
    plt.title(title)
    # translate the point cloud but not divide the furthest length
    centroid = np.mean(pointcloud, axis=0)
    pointcloud -= centroid
    ax.scatter(pointcloud[:,0], pointcloud[:,1], pointcloud[:,2], c='y',s=1)
    xm,ym,zm=get_centroid_from_pc(pointcloud)

# ======================================================
    if(mode=='two_finger_mode'):
        # 計算夾爪位置
        D_ready=0.05 #平移距離(可自由調整)
        D_ready_list=np.asarray([0,0,-D_ready])
        offset_position=np.dot(np.transpose(axes),D_ready_list)
        # ax.scatter(offset_position[0], offset_position[1], offset_position[2], c='red',s=10)
        D_wrist=0.07 #手腕位置
        D_wrist_list=np.asarray([0,0,-D_wrist])
        wrist_offset_position=np.dot(np.transpose(axes),D_wrist_list)
        # ax.scatter(wrist_offset_position[0], wrist_offset_position[1], wrist_offset_position[2], c='red',s=10)
        y_axis_half_width=axes[1]*width/2
        finger_position_x=[xm-y_axis_half_width[0],offset_position[0]-y_axis_half_width[0],offset_position[0]+y_axis_half_width[0],xm+y_axis_half_width[0]]
        finger_position_y=[ym-y_axis_half_width[1],offset_position[1]-y_axis_half_width[1],offset_position[1]+y_axis_half_width[1],ym+y_axis_half_width[1]]
        finger_position_z=[zm-y_axis_half_width[2],offset_position[2]-y_axis_half_width[2],offset_position[2]+y_axis_half_width[2],zm+y_axis_half_width[2]]
        ax.plot(finger_position_x, finger_position_y, finger_position_z, c='black')
        ax.plot([wrist_offset_position[0],offset_position[0]],[wrist_offset_position[1],offset_position[1]],[wrist_offset_position[2],offset_position[2]], c='black')
    elif(mode=='three_finger_mode'):
        # 計算夾爪位置
        D_ready=0.05 #平移距離(可自由調整)
        D_ready_list=np.asarray([0,0,-D_ready])
        offset_position=np.dot(np.transpose(axes),D_ready_list)
        # ax.scatter(offset_position[0], offset_position[1], offset_position[2], c='red',s=10)
        D_wrist=0.07 #手腕位置
        D_wrist_list=np.asarray([0,0,-D_wrist])
        wrist_offset_position=np.dot(np.transpose(axes),D_wrist_list)
        # ax.scatter(wrist_offset_position[0], wrist_offset_position[1], wrist_offset_position[2], c='red',s=10)
        y_axis_half_width=axes[1]*width/2
        first_finger_position_x=[xm-y_axis_half_width[0],offset_position[0]-y_axis_half_width[0],offset_position[0]]
        first_finger_position_y=[ym-y_axis_half_width[1],offset_position[1]-y_axis_half_width[1],offset_position[1]]
        first_finger_position_z=[zm-y_axis_half_width[2],offset_position[2]-y_axis_half_width[2],offset_position[2]]
        ax.plot(first_finger_position_x, first_finger_position_y, first_finger_position_z, c='black')
        ax.plot([wrist_offset_position[0],offset_position[0]],[wrist_offset_position[1],offset_position[1]],[wrist_offset_position[2],offset_position[2]], c='black')
        # second finger
        r = R.from_euler('z',300, degrees=True)
        r3=np.dot(np.transpose(axes),r.as_dcm())
        pca_axis_after_rotating_60=r3.transpose()
        y_axis_half_width_after60=pca_axis_after_rotating_60[1]*width/2
        # ax.scatter(offset_position[0]+y_axis_half_width_after60[0], offset_position[1]+y_axis_half_width_after60[1], offset_position[2]+y_axis_half_width_after60[2], c='red',s=10)
        # ax.scatter(xm+y_axis_half_width_after60[0],ym+y_axis_half_width_after60[1],zm+y_axis_half_width_after60[2], c='red',s=10)
        second_finger_position_x=[offset_position[0],offset_position[0]+y_axis_half_width_after60[0],xm+y_axis_half_width_after60[0]]
        second_finger_position_y=[offset_position[1],offset_position[1]+y_axis_half_width_after60[1],ym+y_axis_half_width_after60[1]]
        second_finger_position_z=[offset_position[2],offset_position[2]+y_axis_half_width_after60[2],zm+y_axis_half_width_after60[2]]
        ax.plot(second_finger_position_x, second_finger_position_y, second_finger_position_z, c='black')
        # third finger
        r = R.from_euler('z',60, degrees=True)
        r3=np.dot(np.transpose(axes),r.as_dcm())
        pca_axis_after_rotating_60=r3.transpose()
        y_axis_half_width_after60=pca_axis_after_rotating_60[1]*width/2
        # ax.scatter(offset_position[0]+y_axis_half_width_after60[0], offset_position[1]+y_axis_half_width_after60[1], offset_position[2]+y_axis_half_width_after60[2], c='red',s=10)
        # ax.scatter(xm+y_axis_half_width_after60[0],ym+y_axis_half_width_after60[1],zm+y_axis_half_width_after60[2], c='red',s=10)
        third_finger_position_x=[offset_position[0],offset_position[0]+y_axis_half_width_after60[0],xm+y_axis_half_width_after60[0]]
        third_finger_position_y=[offset_position[1],offset_position[1]+y_axis_half_width_after60[1],ym+y_axis_half_width_after60[1]]
        third_finger_position_z=[offset_position[2],offset_position[2]+y_axis_half_width_after60[2],zm+y_axis_half_width_after60[2]]
        ax.plot(third_finger_position_x, third_finger_position_y, third_finger_position_z, c='black')
    else:
        pass
    ax.scatter(xm, ym, zm, c='r',s=10)

    plt.show()
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
    point_number=1024
    fps_pcs=furthest_point_sampling(np.asarray(remove_pcd.points),point_number)
    print(fps_pcs.shape)
    origin_pcs_vec,_=cal_pca(fps_pcs.copy(),is_show=False,title="")
    open_widht_length= open_width_algorithm(fps_pcs.copy(),origin_pcs_vec,isVisualize=False)
    print(open_widht_length)
    normalized_point=normalize_point_cloud(np.copy(fps_pcs))
    # visualize_q_pointnet(normalized_point,origin_pcs_vec,"Q-PointNet")
    
    visualize_q_pointnet_with_mode(fps_pcs,origin_pcs_vec,open_widht_length,FINGER_MODE_LIST[1],"Q-PointNet with mode")


