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

    # print(f"Gaussian model features set. Num points: {features['xyz'].shape[0]}")


path = (
    "/local/home/idemir/Desktop/3dv_dynamichumansplatting/per_frame_canonical_human.pkl"
)

with open(path, "rb") as f:
    canonical_human = pickle.load(f)

print(f"Top-level type: {type(canonical_human)}")
print(f"List length: {len(canonical_human)}")

# Print info about first item
first_item = canonical_human[0]
print(f"Type of first item: {type(first_item)}")

if isinstance(first_item, dict):
    print("Keys in first item:", first_item.keys())
    for k, v in first_item.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}, device={v.device}")
        else:
            print(f"  {k}: type={type(v)}")
elif isinstance(first_item, torch.Tensor):
    print(
        f"First item is tensor: shape={first_item.shape}, dtype={first_item.dtype}, device={first_item.device}"
    )
else:
    print("First item details:", first_item)


import os
import pickle
import torch
import torchvision
from tqdm import tqdm

PKL_PATH = (
    "/local/home/idemir/Desktop/3dv_dynamichumansplatting/per_frame_canonical_human.pkl"
)
CAMERA_PATH = "/local/home/idemir/Desktop/3dv_dynamichumansplatting/instantsplat/lab_00000/viewpoint_cam_0.pth"
SAVE_DIR = "./_irem/merging_experiments"
os.makedirs(SAVE_DIR, exist_ok=True)

with open(PKL_PATH, "rb") as f:
    all_gaussians = pickle.load(f)
print(f"Loaded {len(all_gaussians)} canonical humans.")
print("Sample Gaussian structure:")
for k, v in all_gaussians[0].items():
    print(f"  {k}: type={type(v)}, shape={getattr(v, 'shape', None)}")

# find number of tru in visibility mask
print(all_gaussians[0]["visibility_mask"].sum())
for k, v in all_gaussians[1].items():
    print(f"  {k}: type={type(v)}, shape={getattr(v, 'shape', None)}")
print(all_gaussians[1]["visibility_mask"].sum())
camera = torch.load(CAMERA_PATH)


import torch
from typing import List, Dict
from pytorch3d.transforms import so3_exp_map


def average_quaternions(quats: torch.Tensor) -> torch.Tensor:
    """Average quaternion rotations."""
    A = torch.zeros(4, 4, device=quats.device)
    for q in quats:
        A += torch.outer(q, q)
    eigvals, eigvecs = torch.linalg.eigh(A)
    return eigvecs[:, -1]


def merge_canonical_gaussians(gaussians_list: List[Dict]) -> Dict:
    # Accumulate all Gaussian-level attributes
    xyz_list = [g["xyz"] for g in gaussians_list]
    scaling_list = [g["scaling"] for g in gaussians_list]
    rotation_list = [g["rotation"] for g in gaussians_list]
    opacity_list = [g["opacity"] for g in gaussians_list]
    features_dc_list = [g["features_dc"] for g in gaussians_list]
    features_rest_list = [g["features_rest"] for g in gaussians_list]

    # Simply keep SMPL-related info as lists
    smpl_scales = [g["smpl_scale"] for g in gaussians_list]
    smpl_transls = [g["smpl_transl"] for g in gaussians_list]
    smpl_global_orients = [g["smpl_global_orient"] for g in gaussians_list]
    visibility_masks = [g["visibility_mask"] for g in gaussians_list]
    gs_rotq_list = [g["gs_rotq"] for g in gaussians_list]

    # Merge by averaging
    xyz_merged = torch.stack(xyz_list).mean(dim=0)
    scaling_merged = torch.stack(scaling_list).mean(dim=0)
    opacity_merged = torch.stack(opacity_list).mean(dim=0)
    features_dc_merged = torch.stack(features_dc_list).mean(dim=0)
    features_rest_merged = torch.stack(features_rest_list).mean(dim=0)

    # Quaternion averaging per Gaussian
    rotations_stacked = torch.stack(rotation_list)  # [N, 25566, 4]
    rotation_merged = torch.stack(
        [
            average_quaternions(rotations_stacked[:, i])
            for i in range(rotations_stacked.shape[1])
        ],
        dim=0,
    )

    # Final merged canonical
    return {
        "xyz": xyz_merged,
        "scaling": scaling_merged,
        "rotation": rotation_merged,
        "opacity": opacity_merged,
        "features_dc": features_dc_merged,
        "features_rest": features_rest_merged,
        "smpl_scales": smpl_scales,
        "smpl_transls": smpl_transls,
        "smpl_global_orients": smpl_global_orients,
        "visibility_masks": visibility_masks,
        "gs_rotqs": gs_rotq_list,
    }


merged = merge_canonical_gaussians(all_gaussians)


print("merged:", merged.keys())


# exit(0)

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


##################################################

print()


##################################################


rr.init("corresponding_finder_2", recording_id="v0.1")
rr.connect_tcp("0.0.0.0:9876")

device = "cuda"
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


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

    # print(f"Gaussian model features set. Num points: {features['xyz'].shape[0]}")


model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)


for param in model.parameters():
    param.requires_grad = False

for param in model.gs_head_feats.parameters():
    param.requires_grad = True
for param in model.aggregator.parameters():
    param.requires_grad = False

model.gs_head_feats.train()

cfg_file = "/local/home/idemir/Desktop/3dv_dynamichumansplatting/output/human_scene/neuman/lab/hugs_trimlp/demo-dataset.seq=lab/2025-03-30_19-55-08/config_train.yaml"
cfg_file = OmegaConf.load(cfg_file)
cfg = OmegaConf.merge(default_cfg, cfg_file)


dataset = get_train_dataset(cfg)


def get_data(idx):
    data = dataset[idx]
    return data


def render_gaussians(gaussians, camera_list, bg="black"):
    viewpoint_cam = camera_list[0]

    parser = ArgumentParser(description="Training script parameters")
    pipe = PipelineParams(parser)
    args = parser.parse_args(sys.argv[1:])
    pipe = pipe.extract(args)

    pose = gaussians.get_RT(viewpoint_cam.uid)

    if bg == "black":
        bg = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)
    elif bg == "white":
        bg = torch.tensor([1, 1, 1], dtype=torch.float32, device=device)

    render_pkg = render(viewpoint_cam, gaussians, pipe, bg, camera_pose=pose)

    return render_pkg


# point_map, point_conf = model.point_head(aggregated_tokens_list, images, ps_idx)

# image_ids = [0, 1, 2, 3]
image_ids = dataset.train_split
image_names = [
    f"/local/home/idemir/Desktop/3dv_dynamichumansplatting/data/neuman/dataset/lab/images/{i:05}.png"
    for i in image_ids
]


print("optimizer init")

from hugs.losses.utils import ssim
from lpips import LPIPS

lpips = LPIPS(net="vgg", pretrained=True).to(device)
for param in lpips.parameters():
    param.requires_grad = False

import matplotlib.pyplot as plt

l1_losses = []
ssim_losses = []
lpips_losses = []

l1_human_losses = []
ssim_human_losses = []
lpips_human_losses = []

steps = []
lambda_1 = 0.8
lambda_2 = 0.2
lambda_3 = 1.0
iterations = 10000
# exp_name = f"CANONICAL_HUMAN_SINGLE_HEAD_W_OFFSET_{len(image_ids)}_IMAGE"
# exp_name = f"MERGED_HUMAN_SINGLE_HEAD_W_OFFSET_{len(image_ids)}_IMAGE"
exp_name = f"AVERAGING_CANONICAL_HUMAN"

import os
from datetime import datetime

timestamp = datetime.now().strftime("%d-%m_%H-%M")
# folder_name = f"{timestamp}_{exp_name}"
folder_name = f"{exp_name}-{timestamp}"
os.makedirs(f"./_irem/{folder_name}", exist_ok=True)
print("STARTING TRAINING")


#########################################################################################################################
from utils import get_deformed_gs_xyz
from utils import preprocess_masks, preprocess_masks_and_transforms
from utils import smpl_lbsweight_top_k_2d_image_plane
from utils import get_deformed_human_using_image_correspondences
from utils import find_smpl_to_gaussian_correspondence
from utils import *


import pickle

model.gs_head_feats.load_state_dict(
    torch.load(
        "/local/home/idemir/Desktop/3dv_dynamichumansplatting/_irem/1_GS_TRAINING_SINGLE_HEAD_W_OFFSET_82_IMAGE_15000_0.8_0.2_1.0-WHITE_BG-12-05_12-10/gs_head_feats.pth"
    )
)


for param in model.parameters():
    param.requires_grad = False
model.eval()


per_frame_canonical_human = []

for step in tqdm(range(iterations), desc="Training", total=iterations):
    step = 100
    for i in range(0, len(image_ids), 2):
        print(i)

        image_idx = image_ids[i]

        img_path = image_names[i]
        images = load_and_preprocess_images([img_path]).to(device)[None]
        aggregated_tokens_list, ps_idx = model.aggregator(images)
        data = dataset[i]

        smpl_locs, smpl_visibility, smpl_vertices = (
            find_smpl_to_gaussian_correspondence(data)
        )

        canonical_human_rotation = merged["rotation"].to("cuda")
        valid_mask = merged["visibility_mask"][i].to("cuda")
        canonical_human_scales = merged["scaling"].to("cuda")
        smpl_scale = merged["smpl_scales"][i].to("cuda")
        smpl_global_orient = merged["smpl_global_orients"][i].to("cuda")
        smpl_transl = merged["smpl_transls"][i].to("cuda")
        features_dc = merged["features_dc"].to("cuda")
        features_rest = merged["features_rest"].to("cuda")
        opacity = merged["opacity"].to("cuda")

        #######
        raw_norm = canonical_human_rotation.norm(dim=-1, keepdim=True)  # (..., 1)
        normed_rots = torch.nn.functional.normalize(
            canonical_human_rotation, dim=-1
        )  # (..., 4)
        gs_rotmat = quaternion_to_matrix(normed_rots)  # (..., 3, 3)
        deformed_gs_rotq = merged["gs_rotqs"][i].to("cuda")
        render_scale = canonical_human_scales + smpl_scale.log()
        new_features = {
            "xyz": merged["xyz"] * smpl_scale + smpl_transl,
            "features_dc": features_dc,
            "features_rest": features_rest,
            "opacity": opacity,
            "scaling": render_scale,
            "rotation": deformed_gs_rotq,
        }

        camera = torch.load(
            f"/local/home/idemir/Desktop/3dv_dynamichumansplatting/instantsplat/lab_{image_idx:05}/viewpoint_cam_0.pth"
        )
        camera_list = [camera]

        gaussians = GaussianModel_With_Act(0)
        gaussians.init_RT_seq({1.0: camera_list})

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
                image,
                f"./_irem/{folder_name}/{image_idx}_canonical_human_black_MERGED.png",
            )

        set_gaussian_model_features(gaussians, new_features)
        render_pkg = render_gaussians(gaussians, camera_list, bg="white")
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )
        if 100 % 100 == 0:
            # continue
            torchvision.utils.save_image(
                image,
                f"./_irem/{folder_name}/{image_idx}_canonical_human_white_MERGED.png",
            )
        #######

        deformed_xyz, human_smpl_scales, deformed_gs_rotq, _ = (
            get_deform_from_T_to_pose(
                merged["xyz"],
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
            "features_dc": features_dc,
            "features_rest": features_rest,
            "opacity": opacity,
            "scaling": human_smpl_scales,
            "rotation": deformed_gs_rotq,
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
                image, f"./_irem/{folder_name}/{image_idx}_canon2pose_human_MERGED.png"
            )

    break
