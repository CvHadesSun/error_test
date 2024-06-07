import open3d as o3d
import json
import numpy as np
from tools import split_data,load_camera,undist_rgbd,img2pcd,load_multi_cams,align2key
import os
import cv2


def main(root_dir,ignore_list=["Cam-12","Cam-13","Cam-15"]):
    cam_dir = f"{root_dir}"
    cam_intr,cam_extr=load_multi_cams(cam_dir)
    data_dir = f"{root_dir}/images"


    out_dir = "pcd_results"
    os.makedirs(out_dir,exist_ok=True)

    out_img_dir = "undist_imgs"
    # cam_dats=["Cam-0","Cam-1","Cam-2"]

    cam_dats = os.listdir(data_dir)


    for item in cam_dats:
        if item in ignore_list:continue
        out_cam = f"{out_img_dir}/{item}"
        os.makedirs(out_cam,exist_ok=True)

        imgs,depths = split_data(f"{data_dir}/{item}")
        cam_id =  align2key(item)
        intr = np.array(cam_intr[cam_id]["intr"]).reshape(3,3)
        dist = np.array(cam_intr[cam_id]["dist"])

        for i in range(len(imgs)):
            img_path = f"{data_dir}/{item}/{imgs[i]}"
            depth_path = f"{data_dir}/{item}/{depths[i]}"

            img = cv2.imread(img_path)
            depth  =cv2.imread(depth_path,-1)
            # undist_img,undist_depth = undist_rgbd(dist,intr,img,depth)
            undist_img = img
            undist_depth = depth

            cv2.imwrite(f"{out_cam}/rgb.png",undist_img)
            cv2.imwrite(f"{out_cam}/depth.png",undist_depth)



    ##

    # mannual_extr={
    #     0: np.eye(4),
    #     1: extrs["0_1"],
    #     2: extrs["1_2"]@ extrs["0_1"] 
    # }

    for item in cam_dats:
        if item in ignore_list:continue
        cam_id =  align2key(item)
        img_path = f"{out_img_dir}/{item}/rgb.png"
        depth_path = f"{out_img_dir}/{item}/depth.png"
        intr = np.array(cam_intr[cam_id]['intr']).reshape(3,3)
        extr = cam_extr[cam_id]
        img2pcd(img_path,depth_path,intr,extr,f"{out_dir}/{cam_id}.ply")


main("/data1/aobi_dat/new_data")
