from tools import load_camera,undist_rgbd,split_data,img2cam,detect_corners,load_multi_cams,align2key
import os
import cv2
import numpy as np
from tools import get_common_frame,export_pcd,compute_error_2_cam
from itertools import combinations
import trimesh


# undist img
# detection img & get 3d points
# get camera extr and measure error.

def main(root_dir,ignore_list=["Cam-4","Cam-12","Cam-13","Cam-15"],vis_pcd=True):
    cam_dir = f"{root_dir}"
    cam_intr,cam_extr=load_multi_cams(cam_dir)
    data_dir = f"{root_dir}/images"

    # process data.
    cam_dats=os.listdir(data_dir)
    out_dir="./tmp"
    # print(cam_intr)
    
    for item in cam_dats:
        if item in ignore_list:continue

        out_cam_dir = f"{out_dir}/{item}"
        os.makedirs(out_cam_dir,exist_ok=True)

        imgs,depths = split_data(f"{data_dir}/{item}")

        cam_id = align2key(item)

        intr = np.array(cam_intr[cam_id]["intr"]).reshape(3,3)
        dist = np.array(cam_intr[cam_id]["dist"])


        for i in range(len(imgs)):
            img_path = f"{data_dir}/{item}/{imgs[i]}"
            depth_path = f"{data_dir}/{item}/{depths[i]}"

            img = cv2.imread(img_path)
            depth  =cv2.imread(depth_path,-1)

            undist_img,undist_depth = undist_rgbd(dist,intr,img,depth)
            # undist_img = img
            # undist_depth = depth
            corners = detect_corners(undist_img)

            if corners is not None:
                uvs = np.array(corners).reshape(-1,2).astype(np.float32)
                cam_xyz = img2cam(intr,uvs,undist_depth)
                np.save(f"{out_cam_dir}/{i}.npy",cam_xyz)


    # 

    same_frames = get_common_frame('tmp',ignore_list=ignore_list)
    if len(same_frames)<=0:
        print("no common frame in multi camera, please check.")
        return


    if vis_pcd:
        output_cam_pcd="./cam_pcds"
        os.makedirs(output_cam_pcd,exist_ok=True)

    for item in same_frames:
        cam_data={}
        cam_names=[]
        for cam in cam_dats:
            if cam in ignore_list:continue
            cam_id = align2key(cam)
            extr=np.linalg.inv(cam_extr[cam_id])
            if os.path.exists(f"{out_dir}/{cam}/{item}"):
                cam_xyz = np.load(f"{out_dir}/{cam}/{item}")
            else:
                continue
            xyz = extr[:3,:3] @ cam_xyz.T +  extr[:3,3].reshape(-1,1)
            cam_data[cam_id] = xyz
            cam_names.append(cam_id)
            if vis_pcd:
                export_pcd(f'{output_cam_pcd}/{cam}.ply',xyz.T)
        # 
        ll = combinations(cam_names,2)
        errors=[]
        for l in ll:
            cam0 = l[0]
            cam1 = l[1]
            error = compute_error_2_cam(cam_data[cam0],cam_data[cam1])
            errors.append(error)
            print(f"{cam0}-----{cam1} error:",error*1e+3,"mm")

        np_error = np.array(errors)
        print(f"--------------{item} mean error:",np_error.mean()*1e+3,"mm")

        break


    
main("/data1/aobi_dat/new_data")

