from tkinter import *
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import matplotlib.pyplot as plt
import PIL.Image, PIL.ImageTk
import tkinter.messagebox
import numpy as np
from tkinter import ttk
import os
from tkinter.filedialog import askdirectory,askopenfilenames
import provider
from point_cloud_function import (get_centroid_from_pc, cal_pca_for_pose_data_generator, normalize_point_cloud,furthest_point_sampling
									,R)
import tkinter.font as tkFont
import h5py

class App:
	def __init__(self, window, window_title):
		# ===================================================
		# Parameter save into h5 file
		self.SAVE_PARTIAL_PCS_LIST=[]
		self.MODE_LIST=[]
		self.QUATERNION_LABEL_LIST=[]

		# Setting parameter
		self.POINT_CLOUD_SET_LIST=np.asarray([])   #之後會用這個來抽point cloud model
		self.current_pca_axis=np.asarray([])
		self.pca_axis_after_rotating=np.asarray([])
		self.current_point_cloud=np.asarray([])
		self.current_partial_point_cloud=np.asarray([])
		self.covarience=np.asarray([])
		self.length_of_point_cloud_set=0
		# point 數量
		self.NUMBER_OF_POINTS = 1024
		# 切割面的閥值
		self.CUTTING_THRESHOLD=0.05
		# 對稱閥值 小於為對稱 大於為非對稱
		self.SYMMETRIC_THRESHOLD=0.05
		# ===================================================
		self.window = window
		self.window.title(window_title)
		self.window.protocol("WM_DELETE_WINDOW", self.quitApplication)
	# =================tab control==========================

		self.tabControl = ttk.Notebook(self.window)          # Create Tab Control
		self.tab1 = ttk.Frame(self.tabControl)            # Create a tab 
		self.tabControl.add(self.tab1, text='PCA Axis adjustment')      # Add the tab
		self.tab2 = ttk.Frame(self.tabControl)            # Add a second tab
		self.tabControl.add(self.tab2, text='Other function')      # Make second tab visible

		self.tabControl.pack(expand=1, fill="both")  # Pack to make visible
	# =================tab control1==========================
		self.frame1 = ttk.LabelFrame(self.tab1, text=' Function List ')
		self.frame1.grid(row=0,column=0, padx=8, pady=4,sticky=N)
		# ====================================================
		# Frame1_1
		self.frame1_1 = ttk.LabelFrame(self.frame1, text='Load ModelNet40')
		self.frame1_1.pack(anchor='center')
		self.path_label = ttk.Label(self.frame1_1, text="Choose the train_files.txt in Modelnet:")
		self.path_label.pack()
		# Button that lets the user take a loadModelNet40
		self.btn_loadModelNet40=ttk.Button(self.frame1_1, text="loadModelNet40", command=self.loadModelNet40)
		self.btn_loadModelNet40.pack()
		self.path_labelText=StringVar()
		self.path_labelText.set('----------')
		self.show_path=Label(self.frame1_1,textvariable=self.path_labelText)
		self.show_path.pack()

		# ============ Mode packing =======================
		self.mode_packing_frame = ttk.LabelFrame(self.frame1, text=' Mode packing frame')
		self.mode_packing_frame.pack(anchor='center')

		# ============ Rotating_perturbation ===============
		self.frame1_1 = ttk.LabelFrame(self.mode_packing_frame, text=' Rotating perturbation mode ')
		self.frame1_1.pack(side=LEFT)
		self.Rotating_perturbation_Value=IntVar()
		self.Rotating_perturbation_Value.set(1)
		Radiobutton(self.frame1_1,text='On',value=1,variable=self.Rotating_perturbation_Value).pack(side=LEFT)
		Radiobutton(self.frame1_1,text='Off',value=2,variable=self.Rotating_perturbation_Value).pack(side=RIGHT)


		# ============ Restore mode ===============
		self.frame1_2 = ttk.LabelFrame(self.mode_packing_frame, text=' Restore mode ')
		self.frame1_2.pack(side=LEFT)
		self.Restore_mode_Value=IntVar()
		self.Restore_mode_Value.set(1)
		Radiobutton(self.frame1_2,text='On',value=1,variable=self.Restore_mode_Value).pack(side=LEFT)
		Radiobutton(self.frame1_2,text='Off',value=2,variable=self.Restore_mode_Value).pack(side=RIGHT)
		# ========== Button (Catch / confirm) ============
		self.frame1_6 = ttk.LabelFrame(self.frame1, text=' Catching / Confirm & Do Processing ')
		self.frame1_6.pack(anchor='center')
			# ------- Start catching Button -------
		self.Catch_btn = Button(self.frame1_6, text='Start catching', width=15,
					height=1, command = self.start_catching)
		self.Catch_btn.pack(padx=15, pady=10, side=LEFT)
			# ------- Confirm mode Button -------
		self.Confirm_btn = Button(self.frame1_6, text='Confirm mode', width=15,
					height=1, command = self.confirm_mode)
		self.Confirm_btn.pack(padx=15, pady=10, side=LEFT)

		# =========== path selection =============
		self.frame1_3 = ttk.LabelFrame(self.frame1, text='Saving Path ')
		self.frame1_3.pack(anchor='center')
		self.h5_file_dir_label = ttk.Label(self.frame1_3, text="DirName of h5 file:")
		self.h5_file_dir_label.pack(padx=5, pady=10, side=LEFT)
		self.var_Path = StringVar()       # Path Variable
		self.Entry_Path = Entry(self.frame1_3,width=10,textvariable=self.var_Path)     
		self.Entry_Path.pack(padx=5, pady=10, side=LEFT)
		self.path_btn = Button(self.frame1_3, text='select', width=8,
					height=1, command = self.SelectPath)
		self.path_btn.pack(padx=5, pady=10, side=LEFT)

		self.h5_file_name_label = ttk.Label(self.frame1_3, text="Name of h5 file:")
		self.h5_file_name_label.pack(padx=5, pady=10, side=LEFT)
		self.h5_file_name_var = StringVar()       # Path Variable
		self.H5_File_Name_Entry = Entry(self.frame1_3,textvariable=self.h5_file_name_var)     
		self.H5_File_Name_Entry.pack(padx=5, pady=10, side=LEFT)

		# ========== slider for x,y,z axis rotation ==========
		self.frame1_4 = ttk.LabelFrame(self.frame1, text=' Rotation angle ')
		self.frame1_4.pack(anchor='center')
		self.save_pca_btn = Button(self.frame1_4, text='Save PCA', width=8,
					height=1, command = self.savePCA)
		self.save_pca_btn.pack(padx=5, pady=10, side=LEFT)
					# ------- x -------
		self.x_rot_bar_value = IntVar()
		self.x_rot_bar_value.set(0)  # default value = 0
		slider_x = Scale(self.frame1_4,
							from_=360, to=0,
							label='x', variable=self.x_rot_bar_value, command=self.on_change_x
							)
		slider_x.pack(padx=5, pady=10, side=LEFT)

					# ------- y -------
		self.y_rot_bar_value = IntVar()
		self.y_rot_bar_value.set(0)  # default value = 0
		slider_y = Scale(self.frame1_4,
							from_=360, to=0,
							label='y', variable=self.y_rot_bar_value, command=self.on_change_y
							)
		slider_y.pack(padx=5, pady=10, side=LEFT)

					# ------- z -------
		self.z_rot_bar_value = IntVar()
		self.z_rot_bar_value.set(0)  # default value = 0
		slider_z = Scale(self.frame1_4,
							from_=360, to=0,
							label='z', variable=self.z_rot_bar_value, command=self.on_change_z
							)
		slider_z.pack(padx=5, pady=10, side=LEFT)

		# get the sliderbar value
		self.x_angle = self.x_rot_bar_value.get()
		self.y_angle = self.y_rot_bar_value.get()
		self.z_angle = self.z_rot_bar_value.get()
		# =========== Finger mode selection =============
		self.frame1_7 = ttk.LabelFrame(self.frame1_4, text=' Finger selection mode ')
		self.frame1_7.pack(anchor='center')
		self.Finger_selection_Value=IntVar()
		self.Finger_selection_Value.set(1)
		Radiobutton(self.frame1_7,text='Two finger mode',value=0,variable=self.Finger_selection_Value).pack(side=LEFT)
		Radiobutton(self.frame1_7,text='Three finger mode',value=1,variable=self.Finger_selection_Value).pack(side=RIGHT)
			
		# ========== Button (Save / skip / Export h5 file) ============
		self.frame1_5 = ttk.LabelFrame(self.frame1, text=' Save / skip / Export h5 file ')
		self.frame1_5.pack(anchor='center')
			# ------- Save Button -------
		self.Save_btn = Button(self.frame1_5, text='Save', width=8,
					height=1, command = self.Save)
		self.Save_btn.pack(padx=15, pady=10, side=LEFT)
			# ------- Skip Button -------
		self.Skip_btn = Button(self.frame1_5, text='Skip', width=8,
					height=1, command = self.Skip)
		self.Skip_btn.pack(padx=15, pady=10, side=LEFT)
			# ------- Export Button -------
		self.Export_btn = Button(self.frame1_5, text='Export h5 file', width=10,
					height=1,command = self.Export)
		self.Export_btn.pack(padx=15, pady=10, side=LEFT)
	# --------------------------------------------------------
		# frame2
		self.frame2 = ttk.LabelFrame(self.tab1, text=' PCA Demo')
		self.frame2.grid(row=0,column=1, padx=8, pady=4)

		self.fig = Figure(figsize=(5, 5), dpi=100)

		self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame2)  # A tk.DrawingArea.
		self.canvas.draw()

		self.ax = self.fig.add_subplot(111, projection="3d")
		self.ax.set_zlabel('Z')  # 坐标轴
		self.ax.set_ylabel('Y')
		self.ax.set_xlabel('X')
		self.ax.set_xlim3d(-1, 1)
		self.ax.set_ylim3d(-1,1)
		self.ax.set_zlim3d(-1,1)
		
		self.toolbar = NavigationToolbar2Tk(self.canvas, self.frame2)
		self.toolbar.update()
		self.canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

	# --------------------------------------------------------
		# frame3
		self.frame3_font = tkFont.Font(family='Fixdsys', size=13, weight=tkFont.BOLD)
		self.frame3 = ttk.LabelFrame(self.tab1, text=' Parameter display ')
		self.frame3.grid(row=0,column=2, padx=8, pady=4)
		self.point_num_label = ttk.Label(self.frame3, text=" Point number: ",font=self.frame3_font).pack()
		self.num_var = StringVar()
		self.num_var.set('--------')
		self.point_num = ttk.Label(self.frame3, textvariable=self.num_var,font=self.frame3_font).pack()
		self.symmetric_label = ttk.Label(self.frame3, text="Symmetric or not: ",font=self.frame3_font).pack()
		self.symmetric_var = StringVar()
		self.symmetric_var.set('--------')
		self.symmetric_status = ttk.Label(self.frame3, textvariable=self.symmetric_var,font=self.frame3_font).pack()

		self.pc_set_number_label = ttk.Label(self.frame3, text="儲存的點雲數量: ",font=self.frame3_font).pack()
		self.pc_set_number_var = StringVar()
		self.pc_set_number_var.set('--------')
		self.pc_set_number_status = ttk.Label(self.frame3, textvariable=self.pc_set_number_var,font=self.frame3_font).pack()


# =================manu block===========================
		self.menu=Menu(self.window)
		self.window.config(menu=self.menu)
		self.submenu1=Menu(self.menu,tearoff=0)
		self.menu.add_cascade(label='file',menu=self.submenu1)
		self.submenu1.add_command(label='open',command=self.onOpenClick)
		self.submenu1.add_separator()
		self.submenu1.add_command(label='quit',command=self.quitApplication)

		self.submenu2=Menu(self.menu,tearoff=0)
		self.menu.add_cascade(label='edit',menu=self.submenu2)
		self.submenu2.add_command(label='clean all graph')
		# submenu2.add_separator()
		self.submenu2.add_command(label='redo')
# =====================manu block===============================

	# path selection
	def SelectPath(self):
		path_ = askdirectory()
		self.var_Path.set(str(path_))


	# sliderbar's value update
	def on_change_x(self, value):
		# update parameter                
		self.x_angle = self.x_rot_bar_value.get()
		r = R.from_euler('x',self.x_angle, degrees=True)
		# print(r.as_dcm())
		r3=np.dot(self.current_pca_axis.transpose(),r.as_dcm())
		self.pca_axis_after_rotating=r3.transpose()

		self.replot(self.current_partial_point_cloud,self.pca_axis_after_rotating,self.covarience,'PCA Rotate through x-axis')

	# sliderbar's value update
	def on_change_y(self, value):

		# update parameter                
		self.y_angle = self.y_rot_bar_value.get()
		r = R.from_euler('y',self.y_angle, degrees=True)
		# print(r.as_dcm())
		r3=np.dot(self.current_pca_axis.transpose(),r.as_dcm())
		self.pca_axis_after_rotating=r3.transpose()


		self.replot(self.current_partial_point_cloud,self.pca_axis_after_rotating,self.covarience,'PCA Rotate through y-axis')
		
	# sliderbar's value update
	def on_change_z(self, value):

		# update parameter                
		self.z_angle = self.z_rot_bar_value.get()
		r = R.from_euler('z',self.z_angle, degrees=True)
		# print(r.as_dcm())
		r3=np.dot(self.current_pca_axis.transpose(),r.as_dcm())
		self.pca_axis_after_rotating=r3.transpose()


		self.replot(self.current_partial_point_cloud,self.pca_axis_after_rotating,self.covarience,'PCA Rotate through z-axis')
		
	def savePCA(self):
		self.current_pca_axis=np.copy(self.pca_axis_after_rotating)
		self.x_rot_bar_value.set(0)
		self.y_rot_bar_value.set(0)
		self.z_rot_bar_value.set(0)
	# Start catching command
	def start_catching(self):
		if(len(self.POINT_CLOUD_SET_LIST)==0):
			tkinter.messagebox.showwarning("Warning","先載入點雲資料庫")
		self.x_rot_bar_value.set(0)
		self.y_rot_bar_value.set(0)
		self.z_rot_bar_value.set(0)
		# randomly pick one point cloud from the set
		print("start catching mode")
		idx=np.random.choice(len(self.POINT_CLOUD_SET_LIST),1 )
		self.current_point_cloud=self.POINT_CLOUD_SET_LIST[idx]
		self.current_point_cloud=np.squeeze(self.current_point_cloud,axis=0)
		print(self.current_point_cloud.shape)
		# 清除圖片
		self.ax.cla()
		self.ax.set_zlabel('Z')  # 坐标轴
		self.ax.set_ylabel('Y')
		self.ax.set_xlabel('X')
		self.ax.set_xlim3d(-1, 1)
		self.ax.set_ylim3d(-1,1)
		self.ax.set_zlim3d(-1,1)
		self.ax.set_title('Original point cloud')
		self.ax.scatter(self.current_point_cloud[:,0], self.current_point_cloud[:,1], self.current_point_cloud[:,2], c='b',s=1)
		self.canvas.draw()
		self.toolbar.update()

	#  Confirm point cloud mode
	def confirm_mode(self):
		self.ROTATING_MODE = self.Restore_mode_Value.get()
		if self.ROTATING_MODE == 1:
			self.current_point_cloud = self.rotate_point_cloud_through_x_negative90(self.current_point_cloud) #轉到z朝上
		# cal_pca(self.current_point_cloud,False,title="original point cloud")
		# normalized the random normal vector(x,y,z) 
		# 盡量讓normal vector可以是我眼睛看出去的方向，所以才規定這些參數
		rand_normal_vec_X=np.random.uniform(0, 1, 1)
		rand_normal_vec_Z=np.random.uniform(-1, -0.3, 1)
		rand_normal_vec_Y=np.random.uniform(-1, 1, 1)
		rand_normal_vec=np.hstack((rand_normal_vec_X,rand_normal_vec_Y,rand_normal_vec_Z))
		normalized_vec=rand_normal_vec/np.linalg.norm(rand_normal_vec)
		#print("Normal vector： %s" %(normalized_vec,))

		inner_outcome=np.inner(self.current_point_cloud[:],normalized_vec) # 對每個點內積
		self.current_partial_point_cloud=self.current_point_cloud[np.where(inner_outcome-self.CUTTING_THRESHOLD<0)] #挑出那些點小於零
		#print("取出來的partial point cloud數量： %d" %(self.current_partial_point_cloud.shape[0],))
		self.num_var.set(str(self.current_partial_point_cloud.shape[0]))
		if(self.current_partial_point_cloud.shape[0]>self.NUMBER_OF_POINTS):
			self.num_var.set('Okay')
		else:
			self.num_var.set('Need to recatch')
		# 資料前處理
		self.current_partial_point_cloud=normalize_point_cloud(self.current_partial_point_cloud)
		self.current_partial_point_cloud=furthest_point_sampling(self.current_partial_point_cloud,self.NUMBER_OF_POINTS)
		# perturbation
		if self.Rotating_perturbation_Value.get() == 1:
			self.current_partial_point_cloud=self.rotate_perturbation_point_cloud(self.current_partial_point_cloud)
		self.current_pca_axis,self.covarience=cal_pca_for_pose_data_generator(self.current_partial_point_cloud)
		if( self.covarience[0] - self.covarience[1] < self.SYMMETRIC_THRESHOLD):
			self.symmetric_var.set('Fit Three Fingers')
		else:
			self.symmetric_var.set('Fit Two Fingers')
			
		self.replot(self.current_partial_point_cloud,self.current_pca_axis,self.covarience,"Partial point cloud")
		self.pca_axis_after_rotating=np.copy(self.current_pca_axis)


	# Save button command
	def Save(self):
		# 防呆措施
		if(self.current_partial_point_cloud.size != 0):
			self.SAVE_PARTIAL_PCS_LIST.append(self.current_partial_point_cloud.tolist())
			self.MODE_LIST.append(int(self.Finger_selection_Value.get()))
			rotation_matrix=self.current_pca_axis.T
			quaternion= R.from_dcm(rotation_matrix).as_quat()
			self.QUATERNION_LABEL_LIST.append(quaternion)

			self.pc_set_number_var.set(str(len(self.MODE_LIST))+"組")
			# result parameter
			self.current_partial_point_cloud=np.asarray([])
			self.current_pca_axis=np.asarray([])
			self.x_rot_bar_value.set(0)
			self.y_rot_bar_value.set(0)
			self.z_rot_bar_value.set(0)
			self.ax.cla()
			self.canvas.draw()
			self.toolbar.update()
		else:
			tkinter.messagebox.showwarning("Warning","目前尚未有點雲")
	# Skip button command
	def Skip(self):
		self.current_partial_point_cloud=np.asarray([])
		self.start_catching()
	# Export button command
	def Export(self):
		if(not self.var_Path.get() or not self.h5_file_name_var.get()):
			tkinter.messagebox.showwarning("Warning","請輸入h5 file儲存路徑")
			return
		if(not self.SAVE_PARTIAL_PCS_LIST):
			tkinter.messagebox.showwarning("Warning","目前尚未儲存點雲")
			return
		h5_dir=self.var_Path.get()
		h5_name=self.h5_file_name_var.get()
		h5_file_path=h5_dir+"/"+h5_name

		self.save_h5_data_label_pose(h5_file_path,self.SAVE_PARTIAL_PCS_LIST,self.MODE_LIST,self.QUATERNION_LABEL_LIST)
		self.SAVE_PARTIAL_PCS_LIST=[]
		self.MODE_LIST=[]
		self.QUATERNION_LABEL_LIST=[]
		self.pc_set_number_var.set('--------')
	
	# --------------------

	def onOpenClick(self):
		print('open clicked!')
		print(self.POINT_CLOUD_SET_LIST.shape)

	def quitApplication(self):
		quit()
	def loadModelNet40(self):
		file_path_name=askopenfilenames()
		self.path_labelText.set(file_path_name[0])

		# ModelNet40 official train/test split
		TRAIN_FILES = provider.getDataFiles(file_path_name[0])
		# Shuffle train files
		train_file_idxs = np.arange(0, len(TRAIN_FILES))
		np.random.shuffle(train_file_idxs)
		for fn in range(len(TRAIN_FILES)):
			print("目前讀取的檔案是： %s" %(TRAIN_FILES[train_file_idxs[fn]],))
			current_data, current_label = provider.loadDataFile(TRAIN_FILES[train_file_idxs[fn]])
			current_data, current_label, _ = provider.shuffle_data(current_data, np.squeeze(current_label))
			if fn==0:
				point_cloud_collection=current_data
			else:
				point_cloud_collection=np.vstack((point_cloud_collection,current_data))
		self.POINT_CLOUD_SET_LIST=point_cloud_collection
		# Load model 進來後停止使用此button
		self.btn_loadModelNet40.configure(state='disabled')

	def rotate_point_cloud_through_x_negative90(self,point_cloud):
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

	def rotate_perturbation_point_cloud(self,pointcloud, angle_sigma=0.06, angle_clip=0.18):
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
	def replot(self,pointcloud,pca_axis,covarience,title):
		# 清除圖片
		self.ax.cla()
		self.ax.set_zlabel('Z')  # 坐标轴
		self.ax.set_ylabel('Y')
		self.ax.set_xlabel('X')
		self.ax.set_xlim3d(-1, 1)
		self.ax.set_ylim3d(-1,1)
		self.ax.set_zlim3d(-1,1)
		self.ax.set_title(title)
		self.ax.scatter(pointcloud[:,0], pointcloud[:,1], pointcloud[:,2], c='y',s=1)
		# centorid point and PCA axis
		xm,ym,zm=get_centroid_from_pc(pointcloud)
		self.ax.scatter(xm, ym, zm, c='r',s=10)
		discount = 1
		for length, vector in zip(covarience,pca_axis):
			self.ax.quiver(xm,ym,zm,vector[0],vector[1],vector[2], length=discount)
			discount/=3
		self.canvas.draw()
		self.toolbar.update()
	def save_h5_data_label_pose(self,h5_filename, data, label, pose, data_dtype='float32', label_dtype='uint8', pose_dtype='float32'):
		h5_fout = h5py.File(h5_filename)
		h5_fout.create_dataset(
				'data', data=data,
				compression='gzip', compression_opts=4,
				dtype=data_dtype)
		h5_fout.create_dataset(
				'pose', data=pose,
				compression='gzip', compression_opts=4,
				dtype=pose_dtype)
		h5_fout.create_dataset(
				'label', data=label,
				compression='gzip', compression_opts=1,
				dtype=label_dtype)
		h5_fout.close()

if __name__ == "__main__":
# Create a window and pass it to the Application object
# intialize the window in the first parameter.
	App=App(Tk(), "Python 照片處理系統")
	App.window.mainloop()