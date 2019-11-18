# pointcloudtool
Basic function for point cloud

### 需要的套件有

* python-opencv(option)
* open3d
* numpy
* pillow
* matplotlib
* sklearn
* scipy
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

> 2019/10/4修改及實現的功能:
>> * 將join map功能新增進去point_cloud_function.py裡面 
>> * 新增座標轉換功能 quaterion to transformation matrix 的用法
>> * 測試了import 子目錄的module問題(要將util的搜尋層級加入sys.path裡面才能work)
>> 此外 import 子子目錄的py檔案也只要使用相對路徑 `from testfile.test import *` 即可使用
>> 在工作目錄下要使用`sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "child_dir")))`

> 2019/11/15實現的新功能及測試：
>> * rotate point cloud through z axis
>> * jitter point cloud with certain sig
>> * implementation of furthest point sampling which can let you sample
>> point cloud to certain numbers.

> 2019/11/18實現的新功能及測試：
>> * 測試join map without mask的情況下，圖會長得怎樣
>> * 另外也實作了join map with mask的function用以提供restful api裡面的exploration algorithm使用
>> * test_join_map_on_ra605.py file可以測試以上功能
--------------------------------------------------

### TODOLIST:

* 將mask的條件也加入point cloud 的產生(V)
* join map的方法整合進去point_cloud_function.py裡面(V)
* test_pca 裡面的功能 整合進point_cloud_function.py裡面(V)
* select down size的random產生方法實作(V)
* 將join map function加上mask 去實作(V)
* 整合進去restful api的程式碼裡面

---------------------------------------------------
### 文件說明:
> 1. color 跟 depth folder裡面是從slambook載下來的資料集加上pose.txt可以做3d重建<br>
> 2. dataset folder裡面是realsense下來的資料 要搭配預設的camera config去使用<br>
> 3. testfile folder是一些open3d裡面基本的操作 與 point cloud的操作<br>
> 4. join_map.py 實作3d重建,align_depth.py是pyrealsense的基本範例<br>
> 5. 主要就是import point_cloud_function.py就可以用裡面的所有function<br>
> 6. 提供dataset_for_cal_pos folder的資料集以及pose_ra605的三種相對應的姿態提供測試