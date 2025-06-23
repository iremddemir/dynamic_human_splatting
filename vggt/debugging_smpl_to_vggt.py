import sys
sys.path.append('..')

import argparse
import os
import copy

import torch
import joblib
import numpy as np
import open3d as o3d
import cv2
from tqdm import tqdm
from PIL import Image

from data_io import colmap_helper
from utils import ray_utils
from cameras import camera_pose
from geometry.basics import Translation, Rotation
from geometry import transformations, rotation


def read_4d_humans(images_dir, raw_smpl_path):
    vibe_estimates = {
        'joints3d': [],
        'joints2d_img_coord': [],
        'pose': [],
        'betas': [],
    }
    hmr_result = joblib.load(raw_smpl_path)
    for item in hmr_result.items():
        picture = item[1]
        vibe_estimates['joints3d'].append(np.array(picture['3d_joints'][0][0]))
        vibe_estimates['joints2d_img_coord'].append(np.array(picture['2d_joints'][0][0]))
        vibe_estimates['betas'].append(picture['smpl'][0]['betas'])

        body_pose = [rotation.rotation_matrix_to_angle_axis(torch.tensor(p)) for p in picture['smpl'][0]['body_pose']]
        vibe_estimates['pose'].append(body_pose)

    vibe_estimates['joints2d_img_coord'] = np.array(vibe_estimates['joints2d_img_coord'])
    vibe_estimates['joints3d'] = np.array(vibe_estimates['joints3d'])
    vibe_estimates['pose'] = np.array(vibe_estimates['pose'])
    return vibe_estimates


def scale_intrinsics(K, orig_res, new_res):
    sx = new_res[0] / orig_res[0]
    sy = new_res[1] / orig_res[1]
    K_scaled = K.copy()
    K_scaled[0, 0] *= sx  # fx
    K_scaled[0, 2] *= sx  # cx
    K_scaled[1, 1] *= sy  # fy
    K_scaled[1, 2] *= sy  # cy
    return K_scaled


def solve_translation(p3d, p2d, mvp):
    p3d = torch.from_numpy(p3d).float()
    p2d = torch.from_numpy(p2d).float()
    mvp = torch.from_numpy(mvp).float()
    translation = torch.zeros_like(p3d[0:1, 0:3], requires_grad=True)

    optim = torch.optim.Adam([{"params": translation, "lr": 1e-3}])
    for _ in range(1000):
        xyzw = torch.cat([p3d + translation, torch.ones_like(p3d[:, :1])], dim=1)
        cam_pts = torch.matmul(mvp, xyzw.T).T
        img_pts = cam_pts[:, :2] / cam_pts[:, 2:3]
        loss = torch.nn.functional.mse_loss(img_pts, p2d)
        optim.zero_grad()
        loss.backward()
        optim.step()
    return translation.detach().cpu().numpy()


def solve_scale(joints_world, cap, plane_model):
    cam_center = cap.cam_pose.camera_center_in_world
    a, b, c, d = plane_model
    scales = []
    for j in joints_world:
        right = -(a * cam_center[0] + b * cam_center[1] + c * cam_center[2] + d)
        coe = a * (j[0] - cam_center[0]) + b * (j[1] - cam_center[1]) + c * (j[2] - cam_center[2])
        s = right / coe
        if s > 0:
            scales.append(s)
    return min(scales)


def solve_transformation(j3d, j2d, plane_model, colmap_cap, smpl_cap):
    mvp = np.matmul(smpl_cap.intrinsic_matrix, smpl_cap.extrinsic_matrix)
    trans = solve_translation(j3d, j2d, mvp)
    smpl_cap.cam_pose.camera_center_in_world -= trans[0]
    joints_world = (ray_utils.to_homogeneous(j3d)
                    @ smpl_cap.cam_pose.world_to_camera.T
                    @ colmap_cap.cam_pose.camera_to_world.T)[:, :3]
    scale = solve_scale(joints_world, colmap_cap, plane_model)

    transf = smpl_cap.cam_pose.world_to_camera.T * scale
    transf[3, 3] = 1
    transf = transf @ colmap_cap.cam_pose.camera_to_world_3x4.T
    transl = transf[3]
    rot = smpl_cap.cam_pose.world_to_camera.T @ colmap_cap.cam_pose.camera_to_world.T
    rot = rot[:3, :3]
    return transf, scale, transl, rot


def make_smpl_opt(path, scales, transl, rotations):
    hmr_result = joblib.load(path)
    body_pose = []
    global_orient = []
    bbox = []
    betas = np.zeros((1, 10))
    for item, rot in zip(hmr_result.items(), rotations):
        pic = item[1]
        tmpbdps = rotation.rotation_matrix_to_angle_axis(pic['smpl'][0]['body_pose'].clone().detach()).reshape(69)
        body_pose.append(tmpbdps)
        tmpbbox = pic['bbox'][0]
        tmpbbox[2] += tmpbbox[0]
        tmpbbox[3] += tmpbbox[1]
        bbox.append(tmpbbox)
        tmpgo = pic['smpl'][0]['global_orient'][0]
        tmpgo = rot.T @ np.array(tmpgo)
        tmpgo = rotation.rotation_matrix_to_angle_axis(torch.tensor(tmpgo)).reshape(3)
        global_orient.append(tmpgo.numpy())
        betas += np.array(pic['smpl'][0]['betas'])

    betas = np.divide(betas, len(body_pose))
    betas = np.tile(betas, (len(body_pose), 1))
    np.savez('smpl_optimized_aligned_scale.npz',
             global_orient=global_orient, scale=scales,
             transl=transl, body_pose=body_pose, bbox=bbox, betas=betas)
    print('Saved: smpl_optimized_aligned_scale.npz')


def main(opt):
    scene = colmap_helper.ColmapAsciiReader.read_scene(opt.scene_dir, opt.images_dir, order='video')
    raw_smpl = read_4d_humans(opt.images_dir, opt.raw_smpl)

    assert len(raw_smpl['pose']) == len(scene.captures)

    original_res = (1920, 1080)  # Change if your original SMPL generation used a different size
    new_res = (opt.img_width, opt.img_height)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(scene.point_cloud[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(scene.point_cloud[:, 3:] / 255)
    plane_model, _ = pcd.segment_plane(0.02, 3, 1000)

    alignments = {}
    transls, rotations, scales = [], [], []

    for i, cap in tqdm(enumerate(scene.captures), total=len(scene.captures)):
        pts_3d = raw_smpl['joints3d'][i]
        pts_2d = raw_smpl['joints2d_img_coord'][i]

        # Scale camera intrinsics for resized images
        K_scaled = scale_intrinsics(cap.pinhole_cam.intrinsic_matrix, original_res, new_res)

        _, R_rod, t, _ = cv2.solvePnPRansac(
            pts_3d, pts_2d, K_scaled, np.zeros(4), flags=cv2.SOLVEPNP_EPNP)
        R, _ = cv2.Rodrigues(R_rod)
        quat = transformations.quaternion_from_matrix(R).astype(np.float32)
        t = t.astype(np.float32)[:, 0]

        smpl_cap = copy.deepcopy(cap)
        smpl_cap.cam_pose = camera_pose.CameraPose(Translation(t), Rotation(quat))
        smpl_cap.intrinsic_matrix = K_scaled

        transf, scale, transl, rot = solve_transformation(pts_3d, pts_2d, plane_model, cap, smpl_cap)
        scales.append(scale)
        transls.append(transl)
        rotations.append(rot)
        alignments[os.path.basename(cap.image_path)] = transf

    np.save(os.path.join(opt.scene_dir, '../alignments.npy'), alignments)
    print("Saved alignments to alignments.npy")
    make_smpl_opt(opt.raw_smpl, scales, transls, rotations)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_dir', type=str,  help='Path to COLMAP scene directory')
    parser.add_argument('--images_dir', type=str, help='Directory with resized images')
    parser.add_argument('--raw_smpl', type=str,help='Path to SMPL joblib file')
    parser.add_argument('--img_width', type=int, help='Resized image width')
    parser.add_argument('--img_height', type=int, help='Resized image height')
    opt = parser.parse_args()
    opt.scene_dir = "/local/home/idemir/Desktop/3dv_dynamichumansplatting/data/neuman/lab_0/colmap"
    opt.images_dir = "/local/home/idemir/Desktop/3dv_dynamichumansplatting/data/neuman/lab_0/images"
    opt.raw_smpl = "/local/home/idemir/Desktop/3dv_dynamichumansplatting/data/neuman/dataset/lab/smpl_pred/00000_png.npz"
    opt.img_width = 294
    opt.img_height = 518
    main(opt)
