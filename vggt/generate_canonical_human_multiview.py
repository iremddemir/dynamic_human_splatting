import sys

sys.path.append(".")

import os
import time
import random
import pickle
import json
import math
from datetime import datetime

import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms
from omegaconf import OmegaConf
from tqdm import tqdm
from lpips import LPIPS
import matplotlib.pyplot as plt

from vggt.models.our_vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.geometry import closed_form_inverse_se3
from instantsplat.scene.gaussian_model_w_act import GaussianModel_With_Act
from instantsplat.scene.cameras import Camera
from instantsplat.utils.general_utils import build_rotation
from hugs.trainer.gs_trainer import get_train_dataset
from hugs.cfg.config import cfg as default_cfg

from utils import *


SAVE_DIR = "folder_name"
WEIGHTS_PATH = "/work/courses/3dv/14/data/last_gs_head_feats_step.pth"  # adjust where the pretrained weights are located
EXP_NAME = f"exp_name"
neuman_dataset_path = f"/work/courses/3dv/14/projects/ml-hugs/data/neuman"
cfg_file = "output/human_scene/neuman/lab/hugs_trimlp/demo-dataset.seq=lab/2025-03-30_19-55-08/config_train.yaml"  # no need to change

timestamp = datetime.now().strftime("%d-%m_%H-%M")
folder_name = f"{EXP_NAME}-{timestamp}"
exp_folder = f"{SAVE_DIR}/{folder_name}"
os.makedirs(exp_folder, exist_ok=True)


device = "cuda"
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

random.seed(1)
to_tensor = transforms.ToTensor()

if __name__ == "__main__":

    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.gs_head_feats.load_state_dict(torch.load(WEIGHTS_PATH))
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    lpips = LPIPS(net="vgg", pretrained=True).to(device)
    for param in lpips.parameters():
        param.requires_grad = False

    cfg_file = OmegaConf.load(cfg_file)
    cfg = OmegaConf.merge(default_cfg, cfg_file)

    dataset = get_train_dataset(cfg)

    def get_data(idx):
        chosen_idx = dataset.train_split.index(idx)
        data = dataset[chosen_idx]
        return data

    image_ids = dataset.train_split
    image_names = [
        f"{neuman_dataset_path}/dataset/lab/images/{i:05}.png" for i in image_ids
    ]

    per_frame_canonical_human = []
    per_frame_scene_features = []

    def split_features_by_mask(
        features: dict,
        mask: torch.Tensor,
        N: int = 1,  # 2,
        scaling_modifier: float = 1.0,  # 0.8,
    ) -> dict:
        """
        Split each gaussian where mask[i]==1 into N new gaussians sampled
        from its own covariance, shrink their scales, and then remove the originals.

        Args:
            features: dict with keys
                "xyz"           : (P,3) float Tensor
                "features_dc"   : (P,1,F_dc) float Tensor
                "features_rest" : (P,1,F_rest) float Tensor
                "opacity"       : (P,1) float Tensor (logit space)
                "scaling"       : (P,3) float Tensor (log scale)
                "rotation"      : (P,4) float Tensor (quaternions, unnormalized)
            mask:  (P,) bool or {0,1} Tensor selecting which gaussians to split.
            N:     number of children per selected Gaussian.
            scaling_modifier: factor in denominator when shrinking child scales.

        Returns:
            new_features: same keys as input, with length
                        P - M + M*N  where M=mask.sum().
        """
        device = features["xyz"].device
        mask = mask.bool().to(device)
        P = features["xyz"].shape[0]
        M = int(mask.sum().item())
        if M == 0:
            return features.copy()

        # 1) extract parent params
        parent_xyz = features["xyz"][mask]  # (M,3)
        parent_fdc = features["features_dc"][mask]  # (M,1,Fdc)
        parent_frest = features["features_rest"][mask]  # (M,1,Fr)
        parent_op = features["opacity"][mask]  # (M,1)
        parent_scaling = torch.exp(
            features["scaling"][mask]
        )  # convert log-scale -> scale (M,3)
        parent_rot = features["rotation"][mask]  # (M,4)

        # 2) sample N children per parent
        stds = parent_scaling.repeat(N, 1)  # (N*M,3)
        samples = torch.normal(mean=torch.zeros_like(stds), std=stds)  # (N*M,3)
        rots_mat = build_rotation(parent_rot).repeat(N, 1, 1)  # (N*M,3,3)
        world_offsets = torch.bmm(rots_mat, samples.unsqueeze(-1)).squeeze(-1)
        child_xyz = world_offsets + parent_xyz.repeat(N, 1)  # (N*M,3)

        # 3) shrink child scales in log-space
        child_scaling = torch.log(parent_scaling.repeat(N, 1) / (scaling_modifier * N))

        # 4) replicate other attributes
        child_rot = parent_rot.repeat(N, 1)  # (N*M,4)
        child_fdc = parent_fdc.repeat(N, 1, 1)  # (N*M,1,Fdc)
        child_frest = parent_frest.repeat(N, 1, 1)  # (N*M,1,Fr)
        child_op = parent_op.repeat(N, 1)  # (N*M,1)

        # 5) keep only the unmasked originals
        keep = ~mask
        out_xyz = features["xyz"][keep]  # (P-M,3)
        out_fdc = features["features_dc"][keep]
        out_frest = features["features_rest"][keep]
        out_op = features["opacity"][keep]
        out_scaling = features["scaling"][keep]
        out_rotation = features["rotation"][keep]

        # 6) concatenate children
        new_features = {
            "xyz": torch.cat([out_xyz, child_xyz], dim=0),
            "features_dc": torch.cat([out_fdc, child_fdc], dim=0),
            "features_rest": torch.cat([out_frest, child_frest], dim=0),
            "opacity": torch.cat([out_op, child_op], dim=0),
            "scaling": torch.cat([out_scaling, child_scaling], dim=0),
            "rotation": torch.cat([out_rotation, child_rot], dim=0),
        }

        # 7) build the new mask: first (P-M) False, then (N*M) True
        keep_count = P - M
        child_count = N * M
        new_mask = torch.cat(
            [
                torch.zeros(keep_count, dtype=torch.bool, device=device),
                torch.ones(child_count, dtype=torch.bool, device=device),
            ],
            dim=0,
        )

        return new_features, new_mask

    for i in tqdm(range(0, len(image_ids), 1), "Processing images"):

        img_path = [image_names[i]]
        images = load_and_preprocess_images(img_path).to(device)[None]
        aggregated_tokens_list, ps_idx = model.aggregator(images)
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        fov_h = pose_enc[..., 7]
        fov_w = pose_enc[..., 8]

        feats, _ = model.gs_head_feats(aggregated_tokens_list, images, ps_idx)

        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)
        point_map = unproject_depth_map_to_point_map(
            depth_map.squeeze(0), extrinsic.squeeze(0), intrinsic.squeeze(0)
        )[None]
        point_map = torch.tensor(
            point_map, dtype=torch.float32, device=device
        )  # [1, B, H, W, 3]

        # Forward pass through gs_head
        scale = feats[:, :, :, :, 0:3]
        rot = feats[:, :, :, :, 3:7]
        sh = feats[:, :, :, :, 7:10]
        op = feats[:, :, :, :, 10:11]
        offset = feats[:, :, :, :, 11:14]

        print(i)

        b = 0

        image_idx = image_ids[i]

        cam_to_world_extrinsic = (
            closed_form_inverse_se3(extrinsic[0, b, :, :][None])[0]
            .detach()
            .cpu()
            .numpy()
        )
        R_cam_to_world = cam_to_world_extrinsic[:3, :3]
        t_cam_to_world = cam_to_world_extrinsic[:3, 3]
        T_w2c = -R_cam_to_world.T @ t_cam_to_world
        camera = Camera(
            colmap_id=1,
            R=R_cam_to_world,
            T=T_w2c,
            FoVx=fov_w[0, b],
            FoVy=fov_h[0, b],
            image=to_tensor(Image.open(img_path[b]).convert("RGB")),
            gt_alpha_mask=None,
            image_name=f"{image_idx:05}",
            uid=0,
        )
        camera_list = [camera]

        data = get_data(image_idx)
        if (
            to_tensor(Image.open(img_path[b]).convert("RGB")).cuda() - data["rgb"]
        ).abs().sum().item() > 0.0001:
            raise Exception("oops!")

        preprocessed_mask = preprocess_masks([data["mask"]]).view(-1)

        ##########################################

        gaussians = GaussianModel_With_Act(0)
        gaussians.init_RT_seq({1.0: camera_list})

        xyz = point_map[0, b].reshape(-1, 3) + offset[0, b, :, :, 0:3].view(-1, 3)
        features_scene_human = {
            "xyz": xyz,
            "features_dc": sh[0, b, :, :, 0:3].view(-1, 1, 3),
            "features_rest": torch.zeros(
                (sh[0, b, :, :, 0:3].detach().view(-1, 3, 1).shape[0], 15, 3),
                device=xyz.device,
            ),
            "opacity": op[0, b, :, :, 0].view(-1, 1),
            "scaling": scale[0, b, :, :, 0:3].view(-1, 3),
            "rotation": rot[0, b, :, :, 0:4].view(-1, 4),
        }

        features_scene_human, preprocessed_mask = split_features_by_mask(
            features_scene_human, preprocessed_mask.bool()
        )  # , N=3, scaling_modifier=0.5)
        render_and_save(
            gaussians,
            camera_list,
            features_scene_human,
            save_path=f"{exp_folder}/{image_idx}_01_splitted.png",
            bg="white",
        )

        features_human = {
            "xyz": features_scene_human["xyz"][preprocessed_mask.bool()],
            "features_dc": features_scene_human["features_dc"][
                preprocessed_mask.bool()
            ],
            "features_rest": features_scene_human["features_rest"][
                preprocessed_mask.bool()
            ],
            "opacity": features_scene_human["opacity"][preprocessed_mask.bool()],
            "scaling": features_scene_human["scaling"][preprocessed_mask.bool()],
            "rotation": features_scene_human["rotation"][preprocessed_mask.bool()],
        }

        features_scene = {
            "xyz": features_scene_human["xyz"][~preprocessed_mask.bool()],
            "features_dc": features_scene_human["features_dc"][
                ~preprocessed_mask.bool()
            ],
            "features_rest": features_scene_human["features_rest"][
                ~preprocessed_mask.bool()
            ],
            "opacity": features_scene_human["opacity"][~preprocessed_mask.bool()],
            "scaling": features_scene_human["scaling"][~preprocessed_mask.bool()],
            "rotation": features_scene_human["rotation"][~preprocessed_mask.bool()],
        }

        render_and_save(
            gaussians,
            camera_list,
            features_scene_human,
            save_path=f"{exp_folder}/{image_idx}_01_rendered_whole_scene.png",
            bg="white",
        )
        render_and_save(
            gaussians,
            camera_list,
            features_scene_human,
            save_path=f"{exp_folder}/{image_idx}_01_rendered_whole_scene_green.png",
            bg="green",
        )
        render_and_save(
            gaussians,
            camera_list,
            features_human,
            save_path=f"{exp_folder}/{image_idx}_01_human_gaussians_black.png",
            bg="black",
        )
        render_and_save(
            gaussians,
            camera_list,
            features_human,
            save_path=f"{exp_folder}/{image_idx}_01_human_gaussians_white.png",
            bg="white",
        )

        start = time.time()
        (
            smpl_locs,
            smpl_visibility,
            smpl_vertices,
            smpl_faces,
            smpl_cam_normals,
            downsampled_mask,
            zs,
            view_dir,
        ) = find_smpl_to_gaussian_correspondence(data)
        print(f"find_smpl_to_gaussian_correspondence: {(time.time() - start):.3f} secs")

        ### if smpl vertex is out of mask, make it invisible
        y_indices, x_indices = torch.where(preprocess_masks([data["mask"]])[0, 0] == 1)
        mask_pixels_locs = torch.stack((x_indices, y_indices), dim=1)  # shape (M, 2)

        eq = (downsampled_mask[0, 0] > 0)[smpl_locs[:, 1], smpl_locs[:, 0]]
        smpl_visibility = smpl_visibility & eq

        preprocessed_rgb = load_and_preprocess_images(
            [f"{exp_folder}/{image_idx}_01_rendered_whole_scene.png"]
        ).to(
            device
        )  # 69
        overlay_points_and_save(
            preprocessed_rgb,
            smpl_locs[~eq],
            f"{exp_folder}/{image_idx}_02_overlay_removed_smpl_vertices_by_mask.png",
        )
        overlay_points_and_save(
            preprocessed_rgb,
            mask_pixels_locs,
            f"{exp_folder}/{image_idx}_02_overlay_gt_human_mask.png",
        )

        c2w = torch.inverse(camera.world_view_transform).detach().to(device).T
        human_gaussians_pixel_locs, human_gaussians_vis = (
            find_gaussians_rendered_pixel_locs(
                features_human["xyz"],
                c2w,
                camera.FoVx,
                camera.FoVy,
                camera.image_height,
                camera.image_width,
                data["mask"],
            )
        )

        W = mask_pixels_locs[:, 1].max().item() + 1
        eq = torch.isin(
            human_gaussians_pixel_locs[:, 0] * W + human_gaussians_pixel_locs[:, 1],
            mask_pixels_locs[:, 0] * W + mask_pixels_locs[:, 1],
        )
        human_gaussians_vis = human_gaussians_vis & eq
        gauss_pixels = human_gaussians_pixel_locs

        new_features = {
            "xyz": features_human["xyz"][human_gaussians_vis],
            "features_dc": features_human["features_dc"][human_gaussians_vis],
            "features_rest": features_human["features_rest"][human_gaussians_vis],
            "opacity": features_human["opacity"][human_gaussians_vis],
            "scaling": features_human["scaling"][human_gaussians_vis],
            "rotation": features_human["rotation"][human_gaussians_vis],
        }

        render_and_save(
            gaussians,
            camera_list,
            new_features,
            save_path=f"{exp_folder}/{image_idx}_02_human_gaussians_masked_black.png",
            bg="black",
        )
        render_and_save(
            gaussians,
            camera_list,
            new_features,
            save_path=f"{exp_folder}/{image_idx}_02_human_gaussians_masked_white.png",
            bg="white",
        )

        ############################################################################

        start = time.time()
        matched_gaussian_indices = match_smpl_to_gaussians_fast_one2one(
            smpl_locs, gauss_pixels, smpl_visibility, threshold=5.0
        )
        print(f"match_smpl_to_gaussians_fast_one2one: {(time.time() - start):.3f} secs")

        valid_mask = matched_gaussian_indices != -1
        matches = matched_gaussian_indices[valid_mask]
        gauss_pixels = gauss_pixels[matches]

        masked_image = preprocessed_rgb

        masked_image = torch.where(
            preprocess_masks([data["mask"]]).bool(),
            masked_image,
            torch.ones_like(masked_image),
        )
        overlay_points_and_save(
            masked_image,
            smpl_locs[valid_mask],
            f"{exp_folder}/{image_idx}_03_overlay_matched_smpl_vertices.png",
        )

        ############################################################################

        """
            find offsets from the following algorithm:
                1. for each gaussian, take the closest smpl vertex
                2. find that smpl vertex's gaussian match (matched_gaussian)
                3. distance = current_gaussian - matched_gaussian
                4. bu 3D distance vektörünü scale, R ve T ile dönüştür
        """
        start = time.time()
        # Step 1: Filter only visible SMPL vertices
        visible_indices = smpl_visibility.nonzero(as_tuple=True)[
            0
        ]  # shape (V_visible,)
        visible_smpl_locs = smpl_locs[visible_indices]  # shape (V_visible, 2)
        # Step 2: Match visible SMPL vertices to closest gaussians
        matched_gaussian_mask = torch.zeros(
            human_gaussians_pixel_locs.shape[0],
            dtype=torch.bool,
            device=human_gaussians_pixel_locs.device,
        )
        matched_gaussian_mask[matches] = True

        unmatched_human_gaussian_locs = human_gaussians_pixel_locs[
            human_gaussians_vis & ~matched_gaussian_mask
        ]

        dists = torch.cdist(
            unmatched_human_gaussian_locs.float(), visible_smpl_locs.float()
        )  # (G, V_visible)
        min_dists, min_indices = dists.min(
            dim=1
        )  # for each unmatched Gaussian, get closest SMPL

        # Step 3: Threshold for valid matches
        threshold = 20.0  # pixels
        valid_matches = min_dists < threshold

        # Map to full SMPL vertex indices
        closest_smpl_indices = visible_indices[min_indices]  # shape (G,)

        # Optional: only keep valid ones under threshold
        matched_smpl_indices = torch.full(
            (unmatched_human_gaussian_locs.shape[0],),
            -1,
            dtype=torch.long,
            device=smpl_locs.device,
        )
        matched_smpl_indices[valid_matches] = closest_smpl_indices[valid_matches]

        print(f"unmatched gaussian calculation: {(time.time() - start):.3f} secs")

        unmatched_human_gaussian_features = {
            "xyz": features_human["xyz"][human_gaussians_vis & ~matched_gaussian_mask],
            "features_dc": features_human["features_dc"][
                human_gaussians_vis & ~matched_gaussian_mask
            ],
            "features_rest": features_human["features_rest"][
                human_gaussians_vis & ~matched_gaussian_mask
            ],
            "opacity": features_human["opacity"][
                human_gaussians_vis & ~matched_gaussian_mask
            ],
            "scaling": features_human["scaling"][
                human_gaussians_vis & ~matched_gaussian_mask
            ],
            "rotation": features_human["rotation"][
                human_gaussians_vis & ~matched_gaussian_mask
            ],
        }
        render_and_save(
            gaussians,
            camera_list,
            unmatched_human_gaussian_features,
            save_path=f"{exp_folder}/{image_idx}_03_unmatched_human_gaussians.png",
            bg="white",
        )

        overlay_points_and_save(
            preprocessed_rgb,
            unmatched_human_gaussian_locs,
            f"{exp_folder}/{image_idx}_03_overlay_unmatched_gaussians.png",
        )

        matched_human_gaussian_features = {
            "xyz": features_human["xyz"][human_gaussians_vis & matched_gaussian_mask],
            "features_dc": features_human["features_dc"][
                human_gaussians_vis & matched_gaussian_mask
            ],
            "features_rest": features_human["features_rest"][
                human_gaussians_vis & matched_gaussian_mask
            ],
            "opacity": features_human["opacity"][
                human_gaussians_vis & matched_gaussian_mask
            ],
            "scaling": features_human["scaling"][
                human_gaussians_vis & matched_gaussian_mask
            ],
            "rotation": features_human["rotation"][
                human_gaussians_vis & matched_gaussian_mask
            ],
        }
        render_and_save(
            gaussians,
            camera_list,
            matched_human_gaussian_features,
            save_path=f"{exp_folder}/{image_idx}_03_matched_human_gaussians.png",
            bg="white",
        )

        ############################################################################

        """
            smpl_scale, smpl_R, and smpl_transl is from SMPL world to VGGT world
        """
        import gc

        torch.cuda.empty_cache()
        gc.collect()
        start = time.time()
        results = get_deformed_human_using_image_correspondences(
            features_human["xyz"][matches],
            features_human["scaling"][matches],
            features_human["rotation"][matches],
            smpl_vertices,
            gauss_pixels,
            smpl_locs,
            valid_mask,
            data["betas"],
            data["body_pose"],
            data["global_orient"],
        )
        print(
            f"get_deformed_human_using_image_correspondences: {(time.time() - start):.3f} secs"
        )

        (
            deformed_smpl_at_canonical,
            posed_smpl_at_canonical,
            knn_idx,
            smpl_scale,
            smpl_R,
            smpl_transl,
            smpl_global_orient,
            canonical_human_scales,
            canonical_human_rotation,
            lbs_T,
            canonical_normals,
        ) = results

        if (
            torch.isnan(canonical_human_rotation).any()
            or torch.isinf(canonical_human_rotation).any()
        ):
            raise ValueError("⚠️ Canonical rotation contains NaNs or Infs!")

        ############################################################################

        # convert_offsets_in_vggt_to_smpl_world(offsets, smpl_scale, smpl_R, lbs_T, valid_smpl_indices)

        """
            add those offset_canonical to valid_smpl_indices. First convert those offsets in VGGT world to SMPL world
        """
        canonical_human_features_at_smpl = {
            "xyz": deformed_smpl_at_canonical[0][valid_mask],
            "features_dc": features_human["features_dc"][matches],
            "features_rest": features_human["features_rest"][matches],
            "opacity": features_human["opacity"][matches],
            "scaling": canonical_human_scales,
            "rotation": canonical_human_rotation,
        }
        # render_smpl_gaussians(canonical_human_features_at_smpl, f"{exp_folder}/{image_idx}_03_canonical_human_black.png", "black")
        # render_smpl_gaussians(canonical_human_features_at_smpl, f"{exp_folder}/{image_idx}_03_canonical_human_white.png", "white")
        render_smpl_gaussians_gif(
            canonical_human_features_at_smpl,
            f"{exp_folder}/{image_idx}_03_canonical_human_white.gif",
        )

        deformed_xyz, human_smpl_scales, deformed_gs_rotq, lbs_T = (
            get_deform_from_T_to_pose(
                deformed_smpl_at_canonical[0],
                canonical_human_scales,
                canonical_human_rotation,
                valid_mask,
                data["betas"],
                data["body_pose"],
                smpl_global_orient,
                smpl_scale,
                smpl_transl,
            )
        )

        #######
        new_features = {
            "xyz": deformed_xyz[0][valid_mask],
            "features_dc": features_human["features_dc"][matches],
            "features_rest": features_human["features_rest"][matches],
            "opacity": features_human["opacity"][matches],
            "scaling": human_smpl_scales,
            "rotation": deformed_gs_rotq,
        }
        render_and_save(
            gaussians,
            camera_list,
            new_features,
            save_path=f"{exp_folder}/{image_idx}_05_canon2pose_human.png",
            bg="white",
        )

        merged_features = {
            "xyz": torch.cat([features_scene["xyz"], new_features["xyz"]], dim=0),
            "features_dc": torch.cat(
                [features_scene["features_dc"], new_features["features_dc"]], dim=0
            ),
            "features_rest": torch.cat(
                [features_scene["features_rest"], new_features["features_rest"]], dim=0
            ),
            "opacity": torch.cat(
                [features_scene["opacity"], new_features["opacity"]], dim=0
            ),
            "scaling": torch.cat(
                [features_scene["scaling"], new_features["scaling"]], dim=0
            ),
            "rotation": torch.cat(
                [features_scene["rotation"], new_features["rotation"]], dim=0
            ),
        }
        render_and_save(
            gaussians,
            camera_list,
            merged_features,
            save_path=f"{exp_folder}/{image_idx}_05_canon2pose_human_with_scene_white.png",
            bg="white",
        )
        render_and_save(
            gaussians,
            camera_list,
            merged_features,
            save_path=f"{exp_folder}/{image_idx}_05_canon2pose_human_with_scene_green.png",
            bg="green",
        )
        #######

        ##### OFFSET CALCULATION #####
        # required attributes from above:
        #   • smpl_locs             – 2D SMPL‐vertex pixel coordinates
        #   • smpl_visibility       – bool mask of visible SMPL vertices
        #   • data["mask"]          – ground‐truth human segmentation mask
        #   • camera                – current Camera instance
        #   • exp_folder            – path to save all rendered outputs
        #   • smpl_vertices         – SMPL mesh vertices
        #   • deformed_smpl_at_canonical – SMPL verts warped into canonical space
        #   • canonical_human_scales    – per‐vertex scales in canonical space
        #   • canonical_human_rotation  – per‐vertex rotations in canonical space
        #   • smpl_scale, smpl_transl, smpl_global_orient – SMPL→VGGT world transform params
        #   • valid_mask            – mask of gaussians originally matched to SMPL verts
        #   • matches               – indices of those matched gaussians
        #   • features_human        – dict of human‐only gaussian features
        #   • data["betas"], data["body_pose"] – SMPL shape & pose parameters

        # Important: not included in older versions of this code (must be returned from utils functions):
        #   • smpl_cam_normals      – SMPL normals in camera space
        #   • canonical_normals     – SMPL normals in canonical space
        #   • zs                    – per‐vertex depth values
        #   • view_dir              – per‐vertex view‐direction vectors
        #   • smpl_faces            – SMPL mesh faces

        def compute_and_visualize_rim_offsets(
            smpl_pix: torch.Tensor,
            smpl_normals_cam: torch.Tensor,
            smpl_valid_mask: torch.Tensor,
            human_mask: torch.BoolTensor,
            camera: Camera,
            depths: torch.Tensor,
            view_dir: torch.Tensor,
            rendered_img_path: str,
            normals_threshold: float = 0.3,
            offset_threshold: float = 8.0,
            upsample: int = 4,
            save_path: str = None,
        ):
            """
            1) Select "rim" SMPL vertices whose camera-space normal makes
            n·[0,0,1] < normals_threshold.
            2) Compute the 1-px erosion of human_mask to get the binary rim mask.
            3) For each rim-vertex, find the closest rim-pixel (L1 distance).
            4) Upsample the rendered image by `upsample`.
            5) Overlay arrows from each vertex→matched rim pixel and show/save.

            Args:
                smpl_pix:           (N,2) tensor of 2D integer pixel coords [x,y].
                smpl_normals_cam:   (N,3) tensor of normals in camera coords.
                human_mask:         (H,W) bool tensor (True inside human).
                rendered_img_path:  path to the PNG you rendered for this frame.
                normals_threshold:  cutoff on n·view_dir (view_dir=[0,0,1]).
                upsample:           integer factor to enlarge the background image.
                save_path:          if given, path to write out the overlay PNG.

            Returns:
                offsets_2d: (M,2) numpy array of (Δx,Δy) for each rim vertex.
            """
            orig_indices = torch.arange(
                smpl_normals_cam.shape[0], device=smpl_normals_cam.device
            )
            smpl_normals_cam = smpl_normals_cam[smpl_valid_mask]
            smpl_pix = smpl_pix[smpl_valid_mask]
            smpl_idx = orig_indices[smpl_valid_mask]
            view_dir = view_dir[smpl_valid_mask]

            base, _ = os.path.splitext(save_path)
            plotA_path = f"{base}_smpl_rimverts.png"
            plotC_path = save_path

            # --- 1) Select rim‐vertices by normals ---
            # view_dir = torch.tensor([0,0,-1.], device=smpl_normals_cam.device)
            view_dir = view_dir / view_dir.norm(
                dim=1, keepdim=True
            )  # Not completely sure about this here
            dots = (smpl_normals_cam * view_dir).sum(dim=1)  # (N,)
            rim_mask_v = dots < normals_threshold  # (N,)
            smpl_rim_idx = smpl_idx[rim_mask_v]  # (M,)
            smpl_xy = smpl_pix.cpu().numpy().astype(int)  # (N,2)
            rim_xy = smpl_xy[rim_mask_v.cpu().numpy()]  # (M,2)

            # --- 2) Mask rim extraction ---
            mask_np = human_mask.cpu().numpy().astype(np.uint8) * 255
            kernel = np.ones((3, 3), np.uint8)
            eroded = cv2.erode(mask_np, kernel, iterations=1)
            rim_mask = (mask_np == 255) & (eroded == 0)
            ys, xs = np.nonzero(rim_mask)
            rim_pixels = np.stack([xs, ys], axis=1)

            # --- DEBUG PLOT ---
            # determine canvas size from mask
            H, W = human_mask.shape
            canvas = np.stack([mask_np] * 3, axis=2)
            # draw all verts in blue (BGR)
            for x, y in smpl_xy:
                if 0 <= x < W and 0 <= y < H:
                    canvas[y, x] = (0, 0, 255)  # blue
            for x, y in rim_xy:
                if 0 <= x < W and 0 <= y < H:
                    canvas[y, x] = (255, 0, 0)  # red
            # dots_np = dots.cpu().numpy()
            # colors_rgb = cm.get_cmap('magma')(plt.Normalize(vmin=dots_np.min(), vmax=dots_np.max())(dots_np))[:, :3]
            # colors_rgb = (colors_rgb * 255).astype(np.uint8)
            # for (x, y), col in zip(smpl_xy, colors_rgb):
            #     if 0 <= x < W and 0 <= y < H:
            #         canvas[y, x] = tuple(int(c) for c in col)
            for x, y in rim_pixels:
                canvas[y, x] = (0, 255, 0)  # green
            Image.fromarray(canvas).save(plotA_path)
            # ------------------

            # --- 3) Ray‐march along normals to find first boundary pixel ---
            # this is actually an approximation of the offset, should be fine for now though. Exact calculation via Jacobians possible I think
            offsets_2d = []
            for (vx, vy), normal in zip(rim_xy, smpl_normals_cam[rim_mask_v]):
                nx, ny, nz = normal.cpu().numpy()
                # project to 2D pixel‐direction
                dir2d = -np.array([nx / nz, ny / nz], dtype=np.float32)
                # normalize step to unit‐pixel
                step = dir2d / np.linalg.norm(dir2d)
                # march until hit rim
                for t in range(0, max(H, W)):
                    xs_f = vx + step[0] * t
                    ys_f = vy + step[1] * t
                    xi, yi = int(round(xs_f)), int(round(ys_f))
                    if (xi == 0 or xi == W or yi == 0 or yi == H) or rim_mask[yi, xi]:
                        offsets_2d.append([xi - vx, yi - vy])
                        break

            offsets_2d = torch.tensor(offsets_2d, dtype=torch.float32, device=device)

            # Filter the offsets based on the thresholds for each body part
            with open("vggt/smpl_vert_segmentation_subdivided2.json", "r") as f:
                part_segm = json.load(f)
            vertex_to_part = {
                idx: part for part, idxs in part_segm.items() for idx in idxs
            }
            part_thresholds = {
                "head": offset_threshold * 0.5,
                "neck": offset_threshold * 0.5,
                "rightHand": offset_threshold * 0.2,
                "rightHandIndex1": offset_threshold * 0.1,
                "leftHand": offset_threshold * 0.2,
                "leftHandIndex1": offset_threshold * 0.1,
                "rightToeBase": offset_threshold * 0.5,
                "leftToeBase": offset_threshold * 0.5,
            }
            thresholds = torch.tensor(
                [
                    part_thresholds.get(vertex_to_part[int(idx)], offset_threshold)
                    for idx in smpl_rim_idx
                ],
                device=device,
            )
            keep = torch.norm(offsets_2d, dim=1) <= thresholds
            offsets_2d = offsets_2d[keep]

            # Calculate the 3D offsets based on the camera parameters
            fx = camera.image_width / (2 * torch.tan(camera.FoVx / 2))
            fy = camera.image_height / (2 * torch.tan(camera.FoVy / 2))
            cx = (camera.image_width - 1) / 2.0
            cy = (camera.image_height - 1) / 2.0

            u, v = smpl_pix[rim_mask_v][keep].unbind(1)  # (K,2)
            Z = depths[smpl_valid_mask][rim_mask_v][keep]
            n_x, n_y, n_z = smpl_normals_cam[rim_mask_v][keep].unbind(1)

            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy

            denom_Z2 = Z * Z + 1e-12

            s_x = fx * (n_x * Z - X * n_z) / denom_Z2  # (K,)
            s_y = fy * (n_y * Z - Y * n_z) / denom_Z2  # (K,)

            denom = s_x * s_x + s_y * s_y + 1e-12  # (K,)
            normals_offset_scales_3d = (
                s_x * offsets_2d[:, 0] + s_y * offsets_2d[:, 1]
            ) / denom

            valid_mask = torch.zeros_like(smpl_valid_mask)
            valid_mask[smpl_rim_idx[keep]] = True

            # --- 4) VISUALZATION ---
            t0 = time.time()
            img = Image.open(rendered_img_path)
            img = Image.fromarray(
                np.stack([mask_np] * 3, axis=2).astype("uint8"), mode="RGB"
            )
            W0, H0 = img.size
            ratios = np.array([W0 / 518, H0 / 294])
            img_up = img.resize((W0 * upsample, H0 * upsample), Image.BILINEAR)
            keep_np = keep.cpu().numpy()
            all_verts_up = smpl_xy * ratios * upsample  # (N,2)
            rim_verts_up = rim_xy * ratios * upsample  # (M,2)
            rim_pixels_up = rim_pixels * ratios * upsample  # (P,2)
            fig, ax = plt.subplots(
                figsize=(W0 * upsample / 100, H0 * upsample / 100), dpi=100
            )
            ax.imshow(img_up)
            ax.axis("off")
            ax.scatter(rim_pixels_up[:, 0], rim_pixels_up[:, 1], s=1, c="green")
            ax.scatter(all_verts_up[:, 0], all_verts_up[:, 1], s=1, c="blue")
            ax.scatter(
                rim_verts_up[:, 0][keep_np], rim_verts_up[:, 1][keep_np], s=1, c="red"
            )
            ax.scatter(
                rim_verts_up[:, 0][~keep_np],
                rim_verts_up[:, 1][~keep_np],
                s=1,
                c="orange",
            )
            V_up = rim_xy * ratios * upsample
            O_up = offsets_2d.cpu().numpy() * ratios * upsample
            U, V = V_up[:, 0], V_up[:, 1]
            dU, dV = O_up[:, 0], O_up[:, 1]
            ax.quiver(
                U[keep_np],
                V[keep_np],
                dU,
                dV,
                angles="xy",
                scale_units="xy",
                scale=1,
                color="red",
                width=0.0005,
                headwidth=2,
                headlength=2,
            )

            fig.savefig(plotC_path, bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            t1 = time.time()
            print(f"    offset visualization: {t1 - t0:.3f} secs")
            # ---------------------

            fig2, ax2 = plt.subplots(
                figsize=(W0 * upsample / 100, H0 * upsample / 100), dpi=100
            )
            ax2.imshow(img_up)
            ax2.axis("off")

            # smpl_idx is the original‐vertex index for each row of smpl_xy
            parts_list = [vertex_to_part.get(int(i), "unknown") for i in smpl_idx]
            unique_parts = sorted(set(parts_list))
            cmap = plt.get_cmap("tab20", len(unique_parts))
            color_map = {p: cmap(j) for j, p in enumerate(unique_parts)}
            part_colors = [color_map[p] for p in parts_list]

            # scatter all visible verts colored by part
            ax2.scatter(all_verts_up[:, 0], all_verts_up[:, 1], c=part_colors, s=2)

            save_path2 = f"{base}_smpl_by_part_2d.png"
            fig2.savefig(save_path2, bbox_inches="tight", pad_inches=0)
            plt.close(fig2)

            return normals_offset_scales_3d, valid_mask

        start = time.time()
        normals_offset_scales_3d, found_valid_offset = (
            compute_and_visualize_rim_offsets(
                smpl_pix=smpl_locs,
                smpl_normals_cam=smpl_cam_normals,  # must be in camera coords
                smpl_valid_mask=smpl_visibility,
                human_mask=preprocess_masks([data["mask"]]).squeeze(0).squeeze(0),
                camera=camera,
                depths=zs,
                view_dir=view_dir,
                rendered_img_path=f"{exp_folder}/{image_idx}_01_rendered_whole_scene.png",
                save_path=f"{exp_folder}/{image_idx}_99_rim_offsets.png",
            )
        )
        print(f"compute_and_visualize_rim_offsets: {(time.time() - start):.3f} secs")
        normals_offset_scales = torch.zeros(smpl_cam_normals.shape[0], device=device)
        normals_offset_scales[found_valid_offset] = normals_offset_scales_3d

        start = time.time()
        interpolated_offsets = interpolate_laplacian(
            verts=smpl_vertices,
            faces=smpl_faces,
            offsets=normals_offset_scales,
            mask=found_valid_offset,
            num_iters=200,
            lam=0.5,
        )
        print(f"interpolate_laplacian: {(time.time() - start):.3f} secs")

        # Right now I assume recalculating the normals in the canonical space, but just transforming them should also be possible
        # normals_world = smpl_cam_normals @ torch.tensor(camera.R.T, device=device)  # (V, 3)
        # normals_canonical = torch.matmul(normals_world, smpl_R.T)

        # extra coloring just for visualization and debugging
        rgb_green = torch.tensor(
            [57 / 255, 1.0, 20 / 255], device=device
        )  # [0.2235,1,0.0784]
        dc_green = rgb_green / (1.0 / (2 * math.sqrt(math.pi)))
        features_dc_filtered = features_human["features_dc"][matches]
        features_dc_filtered[found_valid_offset[valid_mask]] = dc_green.unsqueeze(
            0
        ).unsqueeze(0)
        rgb_red = torch.tensor([1.0, 0.0, 0.0], device=device)  # [0.2235,1,0.0784]
        dc_red = rgb_red / (1.0 / (2 * math.sqrt(math.pi)))
        features_dc_filtered2 = features_human["features_dc"][matches]
        features_dc_filtered2[found_valid_offset[valid_mask]] = dc_red.unsqueeze(
            0
        ).unsqueeze(0)

        canonical_human_features_at_smpl = {
            "xyz": deformed_smpl_at_canonical[0][valid_mask]
            + canonical_normals[valid_mask]
            * normals_offset_scales.unsqueeze(1)[valid_mask]
            * smpl_scale,
            "features_dc": features_dc_filtered,
            "features_rest": features_human["features_rest"][matches],
            "opacity": features_human["opacity"][matches],
            "scaling": canonical_human_scales,
            "rotation": canonical_human_rotation,
        }
        render_smpl_gaussians_gif(
            canonical_human_features_at_smpl,
            f"{exp_folder}/{image_idx}_99_canonical_human_with_matched_offsets_highlighted_white.gif",
        )

        canonical_human_features_at_smpl = {
            "xyz": deformed_smpl_at_canonical[0][valid_mask]
            + canonical_normals[valid_mask]
            * normals_offset_scales.unsqueeze(1)[valid_mask]
            * smpl_scale,
            "features_dc": features_human["features_dc"][matches],
            "features_rest": features_human["features_rest"][matches],
            "opacity": features_human["opacity"][matches],
            "scaling": canonical_human_scales,
            "rotation": canonical_human_rotation,
        }
        render_smpl_gaussians_gif(
            canonical_human_features_at_smpl,
            f"{exp_folder}/{image_idx}_99_canonical_human_with_matched_offsets_white.gif",
        )

        canonical_human_features_at_smpl = {
            "xyz": deformed_smpl_at_canonical[0][valid_mask]
            + canonical_normals[valid_mask]
            * interpolated_offsets.unsqueeze(1)[valid_mask]
            * smpl_scale,
            "features_dc": features_human["features_dc"][matches],
            "features_rest": features_human["features_rest"][matches],
            "opacity": features_human["opacity"][matches],
            "scaling": canonical_human_scales,
            "rotation": canonical_human_rotation,
        }
        render_smpl_gaussians_gif(
            canonical_human_features_at_smpl,
            f"{exp_folder}/{image_idx}_99_canonical_human_with_matched_offsets_interpolated_white.gif",
        )

        canonical_human_features_at_smpl_wo = {
            "xyz": deformed_smpl_at_canonical[0][valid_mask],
            "features_dc": features_dc_filtered2,
            "features_rest": features_human["features_rest"][matches],
            "opacity": features_human["opacity"][matches],
            "scaling": canonical_human_scales,
            "rotation": canonical_human_rotation,
        }
        render_smpl_gaussians_gif(
            canonical_human_features_at_smpl_wo,
            f"{exp_folder}/{image_idx}_99_canonical_human_without_matched_offsets_white.gif",
        )

        deformed_xyz, human_smpl_scales, deformed_gs_rotq, lbs_T = (
            get_deform_from_T_to_pose(
                deformed_smpl_at_canonical[0]
                + canonical_normals
                * interpolated_offsets.unsqueeze(1)
                * smpl_scale,  # not sure if better transpose individually and then add. Or is it same result?
                canonical_human_scales,
                canonical_human_rotation,
                valid_mask,
                data["betas"],
                data["body_pose"],
                smpl_global_orient,
                smpl_scale,
                smpl_transl,
            )
        )
        new_features = {
            "xyz": deformed_xyz[0][valid_mask],
            "features_dc": features_human["features_dc"][matches],
            "features_rest": features_human["features_rest"][matches],
            "opacity": features_human["opacity"][matches],
            "scaling": human_smpl_scales,
            "rotation": deformed_gs_rotq,
        }
        merged_features = {
            "xyz": torch.cat([features_scene["xyz"], new_features["xyz"]], dim=0),
            "features_dc": torch.cat(
                [features_scene["features_dc"], new_features["features_dc"]], dim=0
            ),
            "features_rest": torch.cat(
                [features_scene["features_rest"], new_features["features_rest"]], dim=0
            ),
            "opacity": torch.cat(
                [features_scene["opacity"], new_features["opacity"]], dim=0
            ),
            "scaling": torch.cat(
                [features_scene["scaling"], new_features["scaling"]], dim=0
            ),
            "rotation": torch.cat(
                [features_scene["rotation"], new_features["rotation"]], dim=0
            ),
        }
        render_and_save(
            gaussians,
            camera_list,
            merged_features,
            save_path=f"{exp_folder}/{image_idx}_99_canon2pose_human_with_scene_green.png",
            bg="green",
        )

        canonical_human_features = {
            "i": i,
            "image_idx": image_idx,
            "smpl_scale": smpl_scale.cpu(),
            "smpl_transl": smpl_transl.cpu(),
            "smpl_global_orient": smpl_global_orient.cpu(),
            "camera": camera.cpu(),
            "smpl_vertices": smpl_vertices.cpu(),
            "smpl_faces": smpl_faces.cpu(),
            "xyz": deformed_smpl_at_canonical[0].cpu(),
            "features_dc": features_human["features_dc"][matches].cpu(),
            "features_rest": features_human["features_rest"][matches].cpu(),
            "opacity": features_human["opacity"][matches].cpu(),
            "scaling": canonical_human_scales.cpu(),
            "rotation": canonical_human_rotation.cpu(),
            "normals": canonical_normals.cpu(),
            "offset_scales": normals_offset_scales.cpu(),
            "offset_scales_interpolated": interpolated_offsets.cpu(),
            "found_valid_offset": found_valid_offset.cpu(),
            "visibility_mask": valid_mask.cpu(),
        }

        per_frame_canonical_human.append(canonical_human_features)
        per_frame_scene_features.append(
            {
                name: feature.cpu()
                for name, feature in features_scene.items()
                if type(feature) == torch.Tensor
            }
        )

    print(
        f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! SAVED AFTER: {image_idx} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    )
    with open(f"{exp_folder}/per_frame_canonical_human.pkl", "wb") as f:
        pickle.dump(per_frame_canonical_human, f)
    with open(f"{exp_folder}/per_frame_scene_features.pkl", "wb") as f:
        pickle.dump(per_frame_scene_features, f)
