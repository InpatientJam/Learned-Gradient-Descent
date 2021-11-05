"""
Code for preprocessing 3DPW dataset.  Adatped from SPIN, the orginal snippet can be found at
https://github.com/nkolot/SPIN/blob/5c796852ca7ca7373e104e8489aa5864323fbf84/datasets/preprocess/pw3d.py
"""

import cv2
import os
import numpy as np
import pickle
import glob


def preprocess(dataset_path, split, out_file):
    # data we'll save
    output = {
        "imgnames": [],
        "genders": [],
        "smpl_pose": [],
        "smpl_shape": [],
        "poses2d": [],
        "poses3d": [],
    }

    split_dir = os.path.join(dataset_path, "sequenceFiles", split)
    print(split_dir)
    for filename in sorted(glob.glob(f"{split_dir}/*.pkl")):
        with open(filename, 'rb') as f:
            data = pickle.load(f, encoding='latin1')

        num_people = len(data["poses"])
        num_frames = len(data["poses"][0])

        genders = data['genders']
        smpl_shape = data["betas"]

        seq_name = str(data['sequence'])
        imgnames = np.array([
            f"imageFiles/{seq_name}/image_{i:05}.jpg"
            for i in range(num_frames)
        ])
        for i in range(num_people):
            valid_campose = data["campose_valid"][i].astype(bool)

            # Same as SPIN, we consider valid frames only
            valid_imgnames = imgnames[valid_campose]
            valid_smpl_pose = data["poses"][i][valid_campose]
            valid_smpl_shape = np.tile(smpl_shape[i][:10],
                                       (len(valid_smpl_pose), 1))
            valid_genders = np.tile(genders[i], len(valid_smpl_pose))
            valid_poses2d = data["poses2d"][i][valid_campose].transpose(0, 2, 1)
            valid_poses3d = data["jointPositions"][i][valid_campose].reshape(-1, 24, 3)

            # transform global poses
            extrinsics = data["cam_poses"][valid_campose][:, :3, :3]
            for j in range(len(extrinsics)):
                global_rot_mat = cv2.Rodrigues(valid_smpl_pose[j, :3])[0]
                valid_smpl_pose[j, :3] = cv2.Rodrigues(np.dot(extrinsics[j], global_rot_mat))[0].T[0]                      

            output["imgnames"].append(valid_imgnames)
            output["genders"].append(valid_genders)
            output["smpl_pose"].append(valid_smpl_pose.astype(np.float32))
            output["smpl_shape"].append(valid_smpl_shape.astype(np.float32))
            output["poses2d"].append(valid_poses2d.astype(np.float32))
            output["poses3d"].append(valid_poses3d.astype(np.float32))

    for k, v in output.items():
        output[k] = np.concatenate(v, axis=0)
        print(k, output[k].shape)
    np.savez(out_file, **output)


if __name__ == "__main__":
    preprocess("./data/3DPW/", "validation","./data/3DPW_valid.npz")
    preprocess("./data/3DPW/", "test", "./data/3DPW_test.npz")
