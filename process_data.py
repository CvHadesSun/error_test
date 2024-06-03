from tools import load_camera,undist_rgbd,split_data,img2cam,detect_corners
import os
import cv2
import numpy as np


# undist img
# detection img & get 3d points
# get camera extr and measure error.

def main(root_dir):
    cam_dir = f"{root_dir}/new_cam"

    cam_files = os.listdir(cam_dir)

    cams={}

    extrs={}

    for item in cam_files:
        left,right,rot,t = load_camera(f"{cam_dir}/{item}")

        for dat in [left,right]:

            if dat['Idx'] not in cams.keys():
                tmp={
                    'intr' :dat["Intrinsic"],
                    "dist": dat["Distortion"]
                }
                cams[dat['Idx']] = tmp


    # process data.

    cam_dats=["Cam-0","Cam-1","Cam-2"]

    out_dir="./tmp"
    
    for item in cam_dats:
        out_cam_dir = f"{out_dir}/{item}"
        os.makedirs(out_cam_dir,exist_ok=True)

        imgs,depths = split_data(f"{root_dir}/{item}")

        cam_id = item.split('-')[-1]

        intr = np.array(cams[cam_id]["intr"]).reshape(3,3)
        dist = np.array(cams[cam_id]["dist"])


        for i in range(len(imgs)):
            img_path = f"{root_dir}/{item}/{imgs[i]}"
            depth_path = f"{root_dir}/{item}/{depths[i]}"

            img = cv2.imread(img_path)
            depth  =cv2.imread(depth_path,-1)

            # undist_img,undist_depth = undist_rgbd(dist,intr,img,depth)
            undist_img = img
            undist_depth = depth
            corners = detect_corners(undist_img)

            if corners is not None:
                uvs = np.array(corners).reshape(-1,2).astype(np.float32)
                cam_xyz = img2cam(intr,uvs,undist_depth)

                np.save(f"{out_cam_dir}/{i}.npy",cam_xyz)



main("/data1/aobi_dat/SavedImgs")

