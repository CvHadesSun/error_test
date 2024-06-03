import open3d as o3d
import json
import numpy as np
from tools import split_data,load_camera,undist_rgbd,img2pcd
import os
import cv2


def main(root_dir):
    cam_dir = f"{root_dir}/new_cam"
    cam_files = os.listdir(cam_dir)
    cams={}
    extrs={}
    for item in cam_files:
        left,right,rot,t = load_camera(f"{cam_dir}/{item}")
        rot= np.array(rot).reshape(3,3)
        t = np.array(t).reshape(-1)
        T = np.eye(4)
        T[:3,:3] = rot #np.linalg.inv(rot)
        T[:3,3] = t*1e-3
        for dat in [left,right]:
            if dat['Idx'] not in cams.keys():
                tmp={
                    'intr' :dat["Intrinsic"],
                    "dist": dat["Distortion"]
                }
                cams[dat['Idx']] = tmp
        id0 = int(left['Idx'])
        id1 = int(right["Idx"])
        extrs[f"{id0}_{id1}"] = T# np.linalg.inv(T)
    trans_extrs={}
    cam_num = 3

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

    # get imgs

    out_dir = "pcd_results"
    os.makedirs(out_dir,exist_ok=True)

    out_img_dir = "undist_imgs"
    cam_dats=["Cam-0","Cam-1","Cam-2"]


    for item in cam_dats:
        out_cam = f"{out_img_dir}/{item}"
        os.makedirs(out_cam,exist_ok=True)

        imgs,depths = split_data(f"{root_dir}/{item}")
        cam_id = item.split('-')[-1]
        intr = np.array(cams[cam_id]["intr"]).reshape(3,3)
        dist = np.array(cams[cam_id]["dist"])

        for i in range(len(imgs)):
            img_path = f"{root_dir}/{item}/{imgs[i]}"
            depth_path = f"{root_dir}/{item}/{depths[i]}"

            img = cv2.imread(img_path)
            depth  =cv2.imread(depth_path,-1)
            undist_img,undist_depth = undist_rgbd(dist,intr,img,depth)
            # undist_img = img
            # undist_depth = depth

            cv2.imwrite(f"{out_cam}/rgb.png",undist_img)
            cv2.imwrite(f"{out_cam}/depth.png",undist_depth)



    ##

    mannual_extr={
        0: np.eye(4),
        1: extrs["0_1"],
        2: extrs["1_2"]@ extrs["0_1"] 
    }

    for item in cam_dats:
        cam_id = item.split('-')[-1]
        img_path = f"{out_img_dir}/{item}/rgb.png"
        depth_path = f"{out_img_dir}/{item}/depth.png"
        intr = np.array(cams[cam_id]['intr']).reshape(3,3)
        extr = mannual_extr[int(cam_id)]
        img2pcd(img_path,depth_path,intr,extr,f"{out_dir}/{cam_id}.ply")


main("/data1/aobi_dat/SavedImgs")
