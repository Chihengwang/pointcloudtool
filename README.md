# pointcloudtool
Basic function for point cloud

### 需要的套件有

* python-opencv(option)
* open3d
* numpy
* pillow
* matplotlib
--------------------------------------------------
### 主要實現的功能:

> 2019/9/22實現的功能：
>>* 完成depth map & colorimg to point cloud 並且提供儲存
>>* down sample的兩種方法整合
>>* remove outlier兩種方法整合
>>* join map程式碼測試(已經可以完成3d reconstruction)

> 2019/9/23實現的功能:
>> * 可以使用mask 去產生partial point cloud並且顯示出來
>> * 另外下面測試的程式碼可以用downsizing 跟 removal去優化partial point cloud
>> * 新增計算中心 以及呈現中心點的matplot方法

> 2019/9/29修改及實現的功能:
>> * 修改down sample & remove outlier原本只能傳送open3d:pointcloud的限制
>>目前可以傳入numpy array &open3d:pointcloud
>> * 新增一個testfile test_pca.py 可以demo pca在pc改變時principal axes的狀況(圖片)
>> * 新增一個normalize的功能 平移到mean=0 同除一個最遠點的長度 可以將所有點scaling

> 2019/10/3修改及實現的功能:
>> * 將cal_pca 功能新增進去point_cloud_function.py裡面 
>> * 新增一個show result的folder來放置一些出圖的function 或script
>> 將test_pca.py改成display_pca.py

--------------------------------------------------

### TODOLIST:

* 將mask的條件也加入point cloud 的產生(V)
* select down size的random產生方法實作
* 整合進去restful api的程式碼裡面
* join map的方法整合進去point_cloud_function.py裡面
* test_pca 裡面的功能 整合進point_cloud_function.py裡面(V)


---------------------------------------------------
### 文件說明:
> 1. color 跟 depth folder裡面是從slambook載下來的資料集加上pose.txt可以做3d重建<br>
> 2. dataset folder裡面是realsense下來的資料 要搭配我預設的camera config去使用<br>
> 3. testfile folder是一些open3d裡面基本的操作 與 point cloud的操作<br>
> 4. join_map.py 實作3d重建,align_depth.py是pyrealsense的基本範例<br>
> 5. 主要就是import point_cloud_function.py就可以用裡面的所有function