import os
import argparse
import torch
import numpy as np
from pathlib import Path
from time import time
import sys
sys.path.append('.')

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
from icecream import ic
ic(torch.cuda.is_available())  # Check if CUDA is available
ic(torch.cuda.device_count())

# from mast3r.model import AsymmetricMASt3R
# from dust3r.image_pairs import make_pairs
# from dust3r.inference import inference

# from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from instantsplat.utils.sfm_utils import (save_intrinsics, save_extrinsic, save_points3D, save_time, save_images_and_masks,
                             init_filestructure, get_sorted_image_files, split_train_test, compute_co_vis_masks)
from instantsplat.utils.camera_utils import generate_interpolated_path

from vggt.models.vggt import VGGT
from vggt.visual_util import predictions_to_glb
# from vggt.utils.load_fn import load_and_preprocess_images
# from vggt.utils.pose_enc import pose_encoding_to_extri_intri
# from vggt.utils.geometry import unproject_depth_map_to_point_map

from vggt.vggt_to_colmap import (
    process_images,
    load_model,
)

def main(source_path, model_path, ckpt_path, device, batch_size, image_size, schedule, lr, niter, 
         min_conf_thr, llffhold, n_views, co_vis_dsp, depth_thre, conf_aware_ranking=False, focal_avg=False, infer_video=False):

    # ---------------- (1) Load model and images ----------------
    save_path, sparse_0_path, sparse_1_path = init_filestructure(Path(source_path), n_views)
    model, device = load_model(device)

    image_dir = Path(source_path) /'images'
    image_files, image_suffix = get_sorted_image_files(image_dir)

    # if infer_video:
    #     train_img_files = image_files
    #     test_img_files = []
    # else:
    #     train_img_files, test_img_files = split_train_test(image_files, llffhold, n_views, verbose=True)
    train_img_files = image_files
    test_img_files = []
    print(f"train_img_files: {train_img_files}, {len(train_img_files)}")
    print(f"test_img_files: {test_img_files}, {len(test_img_files)}")

    start_time = time()
    print(f">> Running VGGT inference...")
    predictions, image_names = process_images(image_dir, model, device)

    glb_scene = predictions_to_glb(predictions)
    #save glb scene
    glb_scene.export(os.path.join(model_path, "scene.glb"))

    # Extract scene information
    extrinsics_c2w = predictions["extrinsic"]  # (N, 3, 4)
    print(f"extrinsics_c2w shape: {extrinsics_c2w.shape}")

    N = extrinsics_c2w.shape[0]
    extrinsics_c2w_hom = np.eye(4)[None].repeat(N, axis=0)
    extrinsics_c2w_hom[:, :3, :4] = extrinsics_c2w
    print(f"extrinsics_c2w_hom shape: {extrinsics_c2w_hom.shape}")

    extrinsics_w2c = np.linalg.inv(extrinsics_c2w_hom)  # (N, 4, 4)
    print(f"extrinsics_w2c shape: {extrinsics_w2c.shape}")

    intrinsics = predictions["intrinsic"]  # (N, 3, 3)
    #print(f"intrinsics: {intrinsics}")
    print(f"intrinsics shape: {intrinsics.shape}")

    imgs = np.array(predictions["images"])
    print(f"imgs shape: {imgs.shape}")

    org_imgs_shape = imgs[0].shape[:2][::-1]  # (W, H)
    print(f"Original image shape (W, H): {org_imgs_shape}")

    pts3d = predictions["world_points_from_depth"]
    print(f"pts3d shape: {pts3d.shape}")

    depthmaps = predictions["depth"]
    print(f"depthmaps shape: {depthmaps.shape}")

    confs = predictions.get("depth_conf", np.ones_like(depthmaps[..., 0]))
    print(f"confs shape: {confs.shape}")

    print(f'>> Calculate the co-visibility mask...')

    if conf_aware_ranking:
        print(f'>> Confiden-aware Ranking...')
        avg_conf_scores = confs.mean(axis=(1, 2))
        sorted_conf_indices = np.argsort(avg_conf_scores)[::-1]
        sorted_conf_avg_conf_scores = avg_conf_scores[sorted_conf_indices]
        print("Sorted indices:", sorted_conf_indices)
        print("Sorted average confidence scores:", sorted_conf_avg_conf_scores)
    else:
        sorted_conf_indices = np.arange(n_views)
        print("Sorted indices:", sorted_conf_indices)

    if depth_thre > 0:
        overlapping_masks = compute_co_vis_masks(sorted_conf_indices, depthmaps, pts3d, intrinsics, extrinsics_w2c, imgs.shape, depth_threshold=depth_thre)
        overlapping_masks = ~overlapping_masks
    else:
        co_vis_dsp = False
        overlapping_masks = None

    end_time = time()
    Train_Time = end_time - start_time
    print(f"Time taken for {n_views} views: {Train_Time:.2f} seconds")
    save_time(model_path, '[1] coarse_init_TrainTime', Train_Time)

    # ---------------- (2) Interpolate training pose to get initial testing pose ----------------
    if not infer_video:
        n_train = len(train_img_files)
        n_test = len(test_img_files)

        if n_train < n_test:
            n_interp = (n_test // (n_train - 1)) + 1
            all_inter_pose = []
            for i in range(n_train - 1):
                tmp_inter_pose = generate_interpolated_path(poses=extrinsics_w2c[i:i + 2], n_interp=n_interp)
                all_inter_pose.append(tmp_inter_pose)
            all_inter_pose = np.concatenate(all_inter_pose, axis=0)
            all_inter_pose = np.concatenate([all_inter_pose, extrinsics_w2c[-1][:3, :].reshape(1, 3, 4)], axis=0)
            indices = np.linspace(0, all_inter_pose.shape[0] - 1, n_test, dtype=int)
            sampled_poses = all_inter_pose[indices]
            sampled_poses = np.array(sampled_poses).reshape(-1, 3, 4)
            assert sampled_poses.shape[0] == n_test
            inter_pose_list = []
            for p in sampled_poses:
                tmp_view = np.eye(4)
                tmp_view[:3, :3] = p[:3, :3]
                tmp_view[:3, 3] = p[:3, 3]
                inter_pose_list.append(tmp_view)
            pose_test_init = np.stack(inter_pose_list, 0)
        else:
            indices = np.linspace(0, extrinsics_w2c.shape[0] - 1, n_test, dtype=int)
            pose_test_init = extrinsics_w2c[indices]

        save_extrinsic(sparse_1_path, pose_test_init, test_img_files, image_suffix)
        test_focals = np.repeat(focals[0], n_test)
        save_intrinsics(sparse_1_path, test_focals, org_imgs_shape, imgs.shape, save_focals=False)

    # ---------------- (3) Save Training Outputs ----------------
    print(f'>> Saving results...')
    np.save(Path(model_path) / 'depthmaps.npy', depthmaps)
    np.save(Path(model_path) / 'intrinsics.npy', intrinsics)
    np.save(Path(model_path) / 'extrinsics.npy', extrinsics_c2w)

    end_time = time()
    save_time(model_path, '[1] init_geo', end_time - start_time)
    save_extrinsic(sparse_0_path, extrinsics_w2c, train_img_files, image_suffix)

    focals_x = intrinsics[:, 0, 0] 
    focals_y = intrinsics[:, 1, 1]
    save_intrinsics(sparse_0_path, focals_x, focals_y, org_imgs_shape, imgs.shape, save_focals=True)
    pts_num = save_points3D(sparse_0_path, imgs, pts3d, confs.reshape(pts3d.shape[0], -1), overlapping_masks, use_masks=co_vis_dsp, save_all_pts=True, save_txt_path=model_path, depth_threshold=depth_thre)
    save_images_and_masks(sparse_0_path, n_views, imgs, overlapping_masks, image_files, image_suffix)
    print(f'[INFO] VGGT Reconstruction is successfully converted to COLMAP files in: {str(sparse_0_path)}')
    print(f'[INFO] Number of points: {pts3d.reshape(-1, 3).shape[0]}')
    print(f'[INFO] Number of points after downsampling: {pts_num}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VGGT Initial Geometry for InstantSplat')
    #parser.add_argument('--source_path', '-s', type=str, required=True, help='Directory containing images')
    #parser.add_argument('--model_path', '-m', type=str, required=True, help='Directory to save the results')
    #parser.add_argument('--device', type=str, default='cuda', help='Device to use for inference')
    #parser.add_argument('--depth_thre', type=float, default=0.01, help='Depth threshold for filtering (unused)')
    #parser.add_argument('--co_vis_dsp', action="store_true", help='Whether to use co-visibility masking (currently disabled)')
    args = parser.parse_args()

    for i in ["20_all"]:
    #for i in range(0, 103):
        #i = str(i).zfill(5)
        source_path = f"./data/neuman/lab_{i}"
        model_path= f"./instantsplat/lab_{i}"
        device = "cuda"
        co_vis_dsp = True
        depth_thre = 0.01
        n_views=20
        llffhold = 8
        min_conf_thr = 5
        co_vis_dsp = False
        batch_size = 1
        image_size = 512
        schedule = "cosine"
        lr = 0.01
        niter = 300
        ckpt_path = ""

        #main(source_path, model_path, device, depth_thre, co_vis_dsp)
        #main(source_path, model_path, device, min_conf_thr, llffhold, n_views, depth_thre, infer_video=False)
        main(source_path, model_path, ckpt_path, device, batch_size, image_size, schedule, lr, niter, 
            min_conf_thr, llffhold, n_views, co_vis_dsp, depth_thre, conf_aware_ranking=False, focal_avg=False, infer_video=True)
        
        print(f"** lab-{i} is done ** ")

