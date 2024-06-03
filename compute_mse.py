from tools import load_camera
import os
import numpy as np
import trimesh
import json


def same_frame(cam0,cam1,cam2):
    min_length_list = min(cam0, cam1, cam2, key=len)
    same_values=[]

    for i in range(len(min_length_list)):
        value = min_length_list[i]
        if value in cam0 and value in cam1 and value in cam2:
            same_values.append(value)
    return same_values


def export_pcd(out_path,data):
    point_cloud = trimesh.PointCloud(data)
    point_cloud.export(out_path)

def main(root_dir,vis_pcd=False):
    cam_dir = f"{root_dir}/new_cam"

    cam_files = os.listdir(cam_dir)

    cams={}

    extrs={}

    for item in cam_files:
        left,right,rot,t = load_camera(f"{cam_dir}/{item}")
        rot= np.array(rot).reshape(3,3)
        t = np.array(t).reshape(-1)
        T = np.eye(4)
        T[:3,:3] = rot
        T[:3,3] =t*1e-3

        for dat in [left,right]:

            if dat['Idx'] not in cams.keys():
                tmp={
                    'intr' :dat["Intrinsic"],
                    "dist": dat["Distortion"]
                }
            
                cams[dat['Idx']] = tmp
                

        id0 = int(left['Idx'])
        id1 = int(right["Idx"])
        extrs[f"{id0}_{id1}"] = T


    # trans_extrs={}

    # cam_num = 3
    # for i in range(cam_num-1):

    #     T = extrs[f"{i}_{i+1}"]
    #     if i ==0:
    #         trans_extrs[0] = T
    #     else:
    #         while 1:
    #             pre_id = i-1
    #             pre_T = trans_extrs[pre_id]
    #             pre_T = T @ pre_T 
    #             trans_extrs[pre_id] = pre_T
    #             if pre_id == 0:
    #                 break
    #         trans_extrs[i] = T
    # trans_extrs[cam_num-1] = np.eye(4)
    

    # process data.

    cam_names=["Cam-0","Cam-1","Cam-2"]

    data_dir="./tmp"

    cams_dats = {}

    for i in range(len(cam_names)):
        cams_dats[i] = os.listdir(f"{data_dir}/{cam_names[i]}")

    comm_name = same_frame(cams_dats[0],cams_dats[1],cams_dats[2])


    mannual_extr={
        0: np.eye(4),
        1: np.linalg.inv(extrs["0_1"]),
        2: np.linalg.inv(extrs["1_2"]@extrs["0_1"] )
    }

    if vis_pcd:
        out_cam_pts="cam_pts"
        os.makedirs(out_cam_dir,exist_ok=True)


    for item in comm_name:
        cam0_xyz = np.load(f"{data_dir}/{cam_names[0]}/{item}")
        cam1_xyz = np.load(f"{data_dir}/{cam_names[1]}/{item}")
        cam2_xyz = np.load(f"{data_dir}/{cam_names[2]}/{item}")

        xyz0 = mannual_extr[0][:3,:3] @ cam0_xyz.T +  mannual_extr[0][:3,3].reshape(-1,1)
        xyz1 = mannual_extr[1][:3,:3] @ cam1_xyz.T +  mannual_extr[1][:3,3].reshape(-1,1)
        xyz2 = mannual_extr[2][:3,:3] @ cam2_xyz.T +  mannual_extr[2][:3,3].reshape(-1,1)


        if vis_pcd:
            export_pcd('cam0.ply',xyz0.T)
            export_pcd('cam1.ply',xyz1.T)
            export_pcd('cam2.ply',xyz2.T)

            export_pcd('cam0_t.ply',cam0_xyz)
            export_pcd('cam1_t.ply',cam1_xyz)
            export_pcd('cam2_t.ply',cam2_xyz)

        diff_01 = (xyz0 - xyz1).mean(1).mean()
        diff_12 = (xyz1 - xyz2).mean(1).mean()
        diff_02 = (xyz0 - xyz2).mean(1).mean()

        mean_value = np.array([diff_01,diff_12,diff_02]).mean()
        print("the mean error is :",mean_value*1e+3,"mm")



main("/data1/aobi_dat/SavedImgs")