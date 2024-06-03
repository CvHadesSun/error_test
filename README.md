## handle book
This is a python script to process raw data , compute mse error, and vis 3d pcd, etc.

### python env
```
trimesh
opencv
open3d
```


### dataset folder example

```
.
├── Cam-0
├── Cam-1
├── Cam-2
├── new_cam
```

### 1. process data

`process_data.py` 用来预处理数据，然后利用rgb & depth获取标定板3d点，用来后续计算误差。

### 2. compute mse error

`compute_mse.py` 使用上述步骤得到的3d点结合相机的外参计算误差并打印。


### 3. vis 3d pcd

`combine_pcd.py` 可以导出拼接的点云结果，每台相机的点云按照顺序导出（当前目录下），已经转换了坐标系。

