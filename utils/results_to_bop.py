import os
import glob
import json
import math
import argparse
import numpy as np
from plyfile import PlyData
from natsort import natsorted
import cv2 as cv
from matplotlib import pyplot as plt
import csv

_ZEROANGLES = 1e-10

def get_K(annotation):
    intrinsics = annotation['camera_data']['intrinsics']
    return np.array([
        [intrinsics['fx'], 0, intrinsics['cx']],
        [0, intrinsics['fy'], intrinsics['cy']],
        [0, 0, 1]
    ])

def quat2vec(q):
    r = np.zeros(3)
    mag = sum(qi ** 2 for qi in q)
    if abs(mag - 1.0) > 1e-5:
        q = [qi / math.sqrt(mag) for qi in q]

    th = math.acos(np.clip(q[0], -1.0, 1.0))
    s = math.sin(th)
    if abs(s) > _ZEROANGLES:
        th = 2.0 * th
        r = [qi * (th / s) for qi in q[1:]]
    return np.array(r)

def quat2wxyz(q_xyzw):
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])

def get_RT_matrix(rvec, tvec):
    R, _ = cv.Rodrigues(rvec)
    RT = np.eye(4)
    RT[:3, :3] = R
    RT[:3, 3] = tvec.flatten()
    return RT, R

def get_loc_q(obj):
    loc_mm = np.array(obj['location']) * 1000
    quat = quat2wxyz(np.array(obj['quaternion_xyzw']))
    return loc_mm, quat

def load3DModel(modelpath):
    ply_data = PlyData.read(modelpath)
    x = np.array(ply_data['vertex']['x'])
    y = np.array(ply_data['vertex']['y'])
    z = np.array(ply_data['vertex']['z'])
    return np.column_stack((x, y, z))

def visualize_pose(modelpoints, K, RT, idx, imgpath, outpath):
    projected = []
    for pt in modelpoints:
        p = K @ RT[:3, :] @ np.append(pt, 1).reshape(4, 1)
        p = p.flatten()
        projected.append(p[:2] / p[2])
    projected = np.array(projected)

    img = cv.imread(imgpath)
    if img is None:
        print(f"[WARN] Could not read image: {imgpath}")
        return
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    plt.figure()
    plt.imshow(img)
    plt.scatter(projected[:, 0], projected[:, 1], s=0.5, c='red', marker='x')
    fig_path = os.path.join(outpath, f"{os.path.splitext(os.path.basename(imgpath))[0]}_render_{idx}.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"[INFO] Saved visualization to {fig_path}")

def process_folder(input_folder, output_folder, modelpath=None, visualize=False):
    os.makedirs(output_folder, exist_ok=True)
    if visualize:
        os.makedirs(os.path.join(output_folder, 'figures'), exist_ok=True)

    files = natsorted(glob.glob(os.path.join(input_folder, "*.json")))
    cam_data = {}
    scene_gt = {}

    for filepath in files:
        with open(filepath, 'r') as f:
            annotation = json.load(f)

        img_id = str(int(os.path.splitext(os.path.basename(filepath))[0]))
        K = get_K(annotation)
        cam_data[img_id] = {"cam_K": list(K.flatten()), "depth_scale": 1.0}
        objects = annotation.get('objects', [])
        entries = []

        for idx, obj in enumerate(objects):
            obj_id = obj['class'].split('_')[1]
            tvec, quat = get_loc_q(obj)
            rvec = quat2vec(quat)
            RT, R = get_RT_matrix(rvec, tvec)

            entries.append({
                "cam_R_m2c": list(R.flatten()),
                "cam_t_m2c": list(tvec),
                "obj_id": obj_id
            })

            if visualize and modelpath:
                model_file = os.path.join(modelpath, f"obj_{int(obj_id):06d}.ply")
                if os.path.exists(model_file):
                    model_points = load3DModel(model_file)
                    img_path = os.path.splitext(filepath)[0] + ".png"
                    visualize_pose(model_points, K, RT, idx, img_path, os.path.join(output_folder, 'figures'))

        if entries:
            scene_gt[img_id] = entries

    with open(os.path.join(output_folder, "scene_camera.json"), 'w') as f:
        json.dump(cam_data, f, indent=4)
    with open(os.path.join(output_folder, "scene_gt.json"), 'w') as f:
        json.dump(scene_gt, f, indent=4)
    print(f"[INFO] Processed {input_folder}")

def write_bop_csv(output_root, csv_path):
    rows = []
    scenes = natsorted([
        os.path.join(output_root, d)
        for d in os.listdir(output_root)
        if os.path.isdir(os.path.join(output_root, d))
    ])

    for scene_path in scenes:
        scene_id = int(os.path.basename(scene_path))  # '000001' -> 1
        gt_file = os.path.join(scene_path, "scene_gt.json")
        if not os.path.exists(gt_file):
            continue

        with open(gt_file, 'r') as f:
            gt_data = json.load(f)

        for im_id_str, entries in gt_data.items():
            im_id = int(im_id_str)
            for entry in entries:
                row = [
                    scene_id,
                    im_id,
                    int(entry['obj_id']),
                    1.0,  # Score placeholder
                    *entry['cam_R_m2c'],
                    *entry['cam_t_m2c'],
                    -1.0  # Time placeholder
                ]
                rows.append(row)

    header = ['scene_id', 'im_id', 'obj_id', 'score'] + \
             [f'R{i}' for i in range(9)] + \
             [f't{i}' for i in range(3)] + ['time']

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"[INFO] BOP-compatible CSV saved to {csv_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert DOPE annotations to EPOS format with BOP evaluation support.")
    parser.add_argument('--root_folder', required=True, help="Root folder with subfolders (e.g. 000001, 000002)")
    parser.add_argument('--outf', required=True, help="Output folder to store EPOS annotations")
    parser.add_argument('--modelpath', help="Path to 3D models (PLY format) for visualization")
    parser.add_argument('--visualize', action='store_true', help="Visualize and save pose projections")
    args = parser.parse_args()

    subfolders = natsorted([
        os.path.join(args.root_folder, d)
        for d in os.listdir(args.root_folder)
        if os.path.isdir(os.path.join(args.root_folder, d))
    ])

    for folder in subfolders:
        folder_id = os.path.basename(folder)
        output_subdir = os.path.join(args.outf, folder_id)
        process_folder(folder, output_subdir, modelpath=args.modelpath, visualize=args.visualize)

    csv_output_path = os.path.join(args.outf, "bop_predictions.csv")
    write_bop_csv(args.outf, csv_output_path)

if __name__ == "__main__":
    main()
