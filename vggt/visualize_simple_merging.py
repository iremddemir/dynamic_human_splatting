import sys

sys.path.append(".")

import torch
from vggt.models.our_vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from instantsplat.scene import GaussianModel
from instantsplat.scene.gaussian_model_w_act import GaussianModel_With_Act
from instantsplat.scene.cameras import Camera
from instantsplat.gaussian_renderer import render
from tqdm import tqdm
from hugs.trainer.gs_trainer import get_train_dataset
from omegaconf import OmegaConf
from hugs.cfg.config import cfg as default_cfg
import torchvision
from instantsplat.arguments import PipelineParams, ArgumentParser
import random

import rerun as rr


import torch
import pickle

device = "cuda"
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


import os
import torch
import numpy as np
import open3d as o3d
import imageio
from typing import List, Dict
from pytorch3d.transforms import so3_exp_map
import pickle


gaussian_dir = "./gaussians"
out_dir = "./videos"
"""
Gaussians we have: 
        canonical_human_features = {
            "i":                    i,
            "image_idx":            image_idx,
            "smpl_scale":           smpl_scale,
            "smpl_transl":          smpl_transl,
            "smpl_global_orient":   smpl_global_orient,
            "camera":               camera,
             
            "xyz":             deformed_smpl_at_canonical[0].to("cuda"),
            "features_dc":     features_human["features_dc"][matches].to("cuda"),
            "features_rest":   features_human["features_rest"][matches].to("cuda"),
            "opacity":         features_human["opacity"][matches].to("cuda"),
            "gs_rotq" :        deformed_gs_rotq,
            "scaling":         canonical_human_scales.to("cuda"),
            "rotation":        canonical_human_rotation.to("cuda"),
            "visibility_mask": valid_mask
        }

"""

import torch


def render_gaussians(gaussians, camera_list):
    viewpoint_cam = camera_list[0]

    parser = ArgumentParser(description="Training script parameters")
    pipe = PipelineParams(parser)
    args = parser.parse_args(sys.argv[1:])
    pipe = pipe.extract(args)

    pose = gaussians.get_RT(viewpoint_cam.uid)

    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)

    render_pkg = render(viewpoint_cam, gaussians, pipe, bg, camera_pose=pose)

    return render_pkg


def set_gaussian_model_features(model: GaussianModel, features: dict):
    """
    Sets feature tensors on a GaussianModel instance without wrapping in nn.Parameter,
    preserving gradient history.

    Args:
        model (GaussianModel): The target GaussianModel.
        features (dict): Dictionary with keys and shapes:
            - 'xyz': (P, 3)
            - 'features_dc': (P, 3, 1)
            - 'features_rest': (P, 3, (sh_degree+1)^2 - 1)
            - 'opacity': (P, 1)
            - 'scaling': (P, 3)
            - 'rotation': (P, 4)
    """
    model._xyz = features["xyz"]
    model._features_dc = features["features_dc"]
    model._features_rest = features["features_rest"]
    model._opacity = features["opacity"]
    model._scaling = features["scaling"]
    model._rotation = features["rotation"]

    model.max_radii2D = torch.zeros(
        (features["xyz"].shape[0]), device=features["xyz"].device
    )


import copy
import torch.nn.functional as F


import copy
import torch.nn.functional as F

os.makedirs(out_dir, exist_ok=True)

PKL_PATH = (
    "/local/home/idemir/Desktop/3dv_dynamichumansplatting/per_frame_canonical_human.pkl"
)
with open(PKL_PATH, "rb") as f:
    all_gaussians = pickle.load(f)

# for i, g in enumerate(all_gaussians):
#     print("features_rest shape:", g["features_dc"].shape)
#     print("sample:", g["features_dc"][0])
#     gaussians = GaussianModel_With_Act(0)
#     image_idx = g["image_idx"]

#     camera = torch.load(f"/local/home/idemir/Desktop/3dv_dynamichumansplatting/instantsplat/lab_{image_idx:05}/viewpoint_cam_0.pth")
#     camera_list = [camera]
#     gaussians.init_RT_seq({1.0: camera_list})

#     g["xyz"] = g["xyz"][g["visibility_mask"]]* g["smpl_scale"] + g["smpl_transl"]
#     render_scale = g["scaling"] + g["smpl_scale"] .log()
#     new_features = {
#             "xyz":            g["xyz"],
#             "features_dc":     g["features_dc"],
#             "features_rest":   g["features_rest"],
#             "opacity":         g["opacity"] ,
#             "scaling":         render_scale,
#             "rotation":        g["gs_rotq"]
#         }
#     set_gaussian_model_features(gaussians, new_features)
#     render_pkg = render_gaussians(gaussians, camera_list)

#     image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
#     if 100 % 100 == 0:
#         #continue
#         torchvision.utils.save_image(image, f"{out_dir}/{image_idx}_canonical_human_black_MERGED.png")

xyzs = []
features_dcs = []
features_rests = []
opacities = []
scalings = []
rotations = []

image_idxs = []
smpl_scales = []
smpl_transls = []
smpl_global_orients = []
cameras = []
gs_rotqs = []
visibility_masks = []
all_gaussians = all_gaussians[0:3]
for i, g in enumerate(all_gaussians):
    """"     canonical_human_features = {
                "i":                    i,
                "image_idx":            image_idx,
                "smpl_scale":           smpl_scale,
                "smpl_transl":          smpl_transl,
                "smpl_global_orient":   smpl_global_orient,
                "camera":               camera,
                
                "xyz":             deformed_smpl_at_canonical[0].to("cuda"),
                "features_dc":     features_human["features_dc"][matches].to("cuda"),
                "features_rest":   features_human["features_rest"][matches].to("cuda"),
                "opacity":         features_human["opacity"][matches].to("cuda"),
                "gs_rotq" :        deformed_gs_rotq,
                "scaling":         canonical_human_scales.to("cuda"),
                "rotation":        canonical_human_rotation.to("cuda"),
                "visibility_mask": valid_mask
            }
    """
    image_idx = g["image_idx"]
    smpl_scale = g["smpl_scale"]
    smpl_transl = g["smpl_transl"]
    smpl_global_orient = g["smpl_global_orient"]
    camera = g["camera"]
    gs_rotq = g["gs_rotq"]

    image_idxs.append(image_idx)
    smpl_scales.append(smpl_scale)
    smpl_transls.append(smpl_transl)
    smpl_global_orients.append(smpl_global_orient)
    cameras.append(camera)
    gs_rotqs.append(gs_rotq)
    visibility_masks.append(g["visibility_mask"])
    xyzs.append(g["xyz"])

    full_features_dc = []
    full_features_rest = []
    full_opacity = []
    full_scaling = []
    full_rotation = []
    local_idx = 0
    for i, v in enumerate(g["visibility_mask"]):
        if v:
            full_features_dc.append(g["features_dc"][local_idx])
            full_features_rest.append(g["features_rest"][local_idx])
            full_opacity.append(g["opacity"][local_idx])
            full_scaling.append(g["scaling"][local_idx])
            full_rotation.append(g["rotation"][local_idx])
            local_idx += 1
        else:
            full_features_dc.append(torch.full_like(g["features_dc"][0], float("nan")))
            full_features_rest.append(
                torch.full_like(g["features_rest"][0], float("nan"))
            )
            full_opacity.append(torch.full_like(g["opacity"][0], float("nan")))
            full_scaling.append(torch.full_like(g["scaling"][0], float("nan")))
            full_rotation.append(torch.full_like(g["rotation"][0], float("nan")))

    full_features_dc = torch.stack(full_features_dc).to("cuda")
    full_features_rest = torch.stack(full_features_rest).to("cuda")
    full_opacity = torch.stack(full_opacity).to("cuda")
    full_scaling = torch.stack(full_scaling).to("cuda")
    full_rotation = torch.stack(full_rotation).to("cuda")

    features_dcs.append(full_features_dc)
    features_rests.append(full_features_rest)
    opacities.append(full_opacity)
    scalings.append(full_scaling)
    rotations.append(full_rotation)


# Stack: shape [num_frames, 110210, C]
stacked_features_dc = torch.stack(features_dcs, dim=0)  # [T, 110210, 3]
print("stacked_features_dc shape:", stacked_features_dc.shape)
merged_features_dc = torch.nanmean(stacked_features_dc, dim=0)  # [110210, 3]
print("merged_features_dc shape:", merged_features_dc.shape)

stacked_features_rest = torch.stack(features_rests, dim=0)  # [T, 110210, 15, 3]
print("stacked_features_rest shape:", stacked_features_rest.shape)
merged_features_rest = torch.nanmean(stacked_features_rest, dim=0)  # [110210, 15, 3]
print("merged_features_rest shape:", merged_features_rest.shape)

stacked_opacity = torch.stack(opacities, dim=0)  # [T, 110210, 1]
print("stacked_opacity shape:", stacked_opacity.shape)
merged_opacity = torch.nanmean(stacked_opacity, dim=0)
print("merged_opacity shape:", merged_opacity.shape)

stacked_scaling = torch.stack(scalings, dim=0)  # [T, 110210, 3]
print("stacked_scaling shape:", stacked_scaling.shape)
merged_scaling = torch.nanmean(stacked_scaling, dim=0)
print("merged_scaling shape:", merged_scaling.shape)

stacked_rotation = torch.stack(rotations, dim=0)  # [T, 110210, 4]
print("stacked_rotation shape:", stacked_rotation.shape)
merged_rotation = torch.nanmean(stacked_rotation, dim=0)
print("merged_rotation shape:", merged_rotation.shape)

stacked_opacity = torch.stack(opacities, dim=0)  # [T, 110210, 1]
print("stacked_opacity shape:", stacked_opacity.shape)
merged_opacity = torch.nanmean(stacked_opacity, dim=0)
print("merged_opacity shape:", merged_opacity.shape)

merged_xyz = xyzs[0]

visibility_masks_tensor = torch.stack(visibility_masks)  # [N_frames, 110210]
merged_visibility_mask = visibility_masks_tensor.any(dim=0)  # [110210]

merged_gaussian = {
    "i": i,
    "image_idxs": image_idxs,
    "smpl_scales": smpl_scales,
    "smpl_transls": smpl_transls,
    "smpl_global_orients": smpl_global_orients,
    "cameras": cameras,
    "xyz": merged_xyz.to("cuda"),
    "features_dc": merged_features_dc.to("cuda"),
    "features_rest": merged_features_rest.to("cuda"),
    "opacities": merged_opacity.to("cuda"),
    "gs_rotqs": gs_rotqs,
    "scaling": merged_scaling.to("cuda"),
    "rotation": merged_rotation.to("cuda"),
    "visibility_masks": merged_visibility_mask.to("cuda"),
}

for i, image_idx in enumerate(image_idxs):
    gaussians = GaussianModel_With_Act(0)

    camera = torch.load(
        f"/local/home/idemir/Desktop/3dv_dynamichumansplatting/instantsplat/lab_{image_idx:05}/viewpoint_cam_0.pth"
    )
    camera_list = [camera]
    gaussians.init_RT_seq({1.0: camera_list})

    xyz = (
        merged_gaussian["xyz"][merged_gaussian["visibility_masks"]]
        * merged_gaussian["smpl_scales"][i]
        + merged_gaussian["smpl_transls"][i]
    )
    render_scale = (
        merged_gaussian["scaling"][merged_gaussian["visibility_masks"]]
        + merged_gaussian["smpl_scales"][i].log()
    )
    new_features = {
        "xyz": xyz,
        "features_dc": merged_gaussian["features_dc"][
            merged_gaussian["visibility_masks"]
        ],
        "features_rest": merged_gaussian["features_rest"][
            merged_gaussian["visibility_masks"]
        ],
        "opacity": merged_gaussian["opacities"][merged_gaussian["visibility_masks"]],
        "scaling": render_scale,
        "rotation": merged_gaussian["gs_rotqs"][i],
    }
    set_gaussian_model_features(gaussians, new_features)
    render_pkg = render_gaussians(gaussians, camera_list)

    image, viewspace_point_tensor, visibility_filter, radii = (
        render_pkg["render"],
        render_pkg["viewspace_points"],
        render_pkg["visibility_filter"],
        render_pkg["radii"],
    )
    if 100 % 100 == 0:
        # continue
        torchvision.utils.save_image(
            image, f"{out_dir}/{image_idx}_canonical_human_black_MERGED_v2.png"
        )
