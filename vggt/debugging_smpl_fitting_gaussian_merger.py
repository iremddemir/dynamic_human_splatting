import sys
sys.path.append('.')

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

    model.max_radii2D = torch.zeros((features["xyz"].shape[0]), device=features["xyz"].device)


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

# image_ids = [0, 1, 2, 3]
image_ids = dataset.train_split[:26]
image_names = [f"/local/home/idemir/Desktop/3dv_dynamichumansplatting/data/neuman/dataset/lab/images/{i:05}.png" for i in image_ids]


print("optimizer init")

from hugs.losses.utils import ssim
from lpips import LPIPS
lpips = LPIPS(net="vgg", pretrained=True).to(device)
for param in lpips.parameters(): param.requires_grad=False

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
exp_name = f"CANONICAL_HUMAN_SINGLE_HEAD_W_OFFSET_{len(image_ids)}_IMAGE_{iterations}_{lambda_1}_{lambda_2}_{lambda_3}"

import os
from datetime import datetime
timestamp = datetime.now().strftime("%d-%m_%H-%M")
# folder_name = f"{timestamp}_{exp_name}"
folder_name = f"{exp_name}"
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
# model.gs_head_feats.load_state_dict(torch.load("/local/home/idemir/Desktop/3dv_dynamichumansplatting/_irem/SINGLE_HEAD_W_OFFSET_4_IMAGE_10000_0.8_0.2_1.0/gs_head_feats.pth"))
# model.gs_head_feats.load_state_dict(torch.load("/local/home/idemir/Desktop/3dv_dynamichumansplatting/_irem/10_MAY_SINGLE_HEAD_W_OFFSET_4_IMAGE_10000_0.8_0.2_1.0/gs_head_feats.pth"))
# model.gs_head_feats.load_state_dict(torch.load("/local/home/idemir/Desktop/3dv_dynamichumansplatting/_irem/10_MAY_SINGLE_HEAD_W_OFFSET_32_IMAGE_10000_0.8_0.2_1.0/gs_head_feats.pth"))
model.gs_head_feats.load_state_dict(torch.load("/local/home/idemir/Desktop/3dv_dynamichumansplatting/_irem/10_MAY_I_FIXED_BUG_SINGLE_HEAD_W_OFFSET_26_IMAGE_10000_0.8_0.2_1.0/gs_head_feats_middle.pth"))


for param in model.parameters():
    param.requires_grad = False
model.eval()


for step in tqdm(range(iterations), desc="Training", total=iterations):

    # GET THE DATA POINT:
    # i = random.randint(0, images_list.shape[1] - 1)
    
    i = random.randint(0, len(image_ids) - 1)
    image_idx = image_ids[i]


    # images = images_list[i]
    # aggregated_tokens_list = tokens_list[i]
    # ps_idx = ps_idx_list[i]
    # data = data_list[i]


    img_path = image_names[i]
    images = load_and_preprocess_images([img_path]).to(device)[None]
    aggregated_tokens_list, ps_idx = model.aggregator(images)
    data = dataset[i]

    
    feats, _ = model.gs_head_feats(aggregated_tokens_list, images, ps_idx)

    # Forward pass through gs_head
    scale   = feats[:,:,:,:,0:3]
    rot     = feats[:,:,:,:,3:7]
    sh      = feats[:,:,:,:,7:10]
    op      = feats[:,:,:,:,10:11]
    offset  = feats[:,:,:,:,11:14]
    # offset, _ = model.gs_head_xyz_offset(aggregated_tokens_list, images, ps_idx)
    # op, _ = model.gs_head_opacity(aggregated_tokens_list, images, ps_idx)
    # rot, _ = model.gs_head_rotation(aggregated_tokens_list, images, ps_idx)
    # scale, _ = model.gs_head_scale(aggregated_tokens_list, images, ps_idx)
    # sh, _ = model.gs_head_sh(aggregated_tokens_list, images, ps_idx)

    point_map, _ = model.point_head(aggregated_tokens_list, images, ps_idx)
    xyz = point_map[0, 0].reshape(-1, 3)
    xyz += offset[0, 0, :, :, 0:3].view(-1, 3)
    
    # camera = torch.load("camera_0.pth")
    # camera = torch.load("viewpoint_cam.pth")
    # camera = torch.load(f"/local/home/idemir/Desktop/3dv_dynamichumansplatting/instantsplat/lab_{i}/viewpoint_cam.pth") 
    camera = torch.load(f"/local/home/idemir/Desktop/3dv_dynamichumansplatting/instantsplat/lab_{image_idx:05}/viewpoint_cam_0.pth") 
    # camera = torch.load("viewpoint_cam.pth") 
    camera_list = [camera]


    gaussians = GaussianModel_With_Act(0)
    gaussians.init_RT_seq({1.0: camera_list})

    
    features_scene_human = {
        "xyz":              xyz,
        "features_dc":      sh[0, 0, :, :, 0:3].view(-1, 1, 3),
        "features_rest": torch.zeros((sh[0, 0, :, :, 0:3].detach().view(-1, 3, 1).shape[0], 15, 3), device=xyz.device),
        "opacity":          op[0, 0, :, :, 0].view(-1, 1),
        "scaling":          scale[0, 0, :, :, 0:3].view(-1, 3),
        "rotation":         rot[0, 0, :, :, 0:4].view(-1, 4),
    }
    set_gaussian_model_features(gaussians, features_scene_human)

    # render_pkg = render_gaussians(gaussians_from_other, camera_list)
    render_pkg = render_gaussians(gaussians, camera_list)
    image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
    if step % 100 == 0:
        #continue
        torchvision.utils.save_image(image, f"./_irem/{image_idx}_scene.png")

    # preprocessed_mask = preprocess_masks([data["mask"]]).view(-1)


    mask_tensor = torchvision.io.read_image(
        "/local/home/idemir/Desktop/3dv_dynamichumansplatting/data/neuman/dataset/lab/4d_humans/sam_segmentations/mask_0000.png",
        torchvision.io.ImageReadMode.GRAY
    ).gt(0).to(device=device, dtype=torch.float32).squeeze(0)
    preprocessed_mask = preprocess_masks([mask_tensor]).view(-1)


    features_human = {
        "xyz":             features_scene_human["xyz"][preprocessed_mask.bool()],
        "features_dc":     features_scene_human["features_dc"][preprocessed_mask.bool()],
        "features_rest":   features_scene_human["features_rest"][preprocessed_mask.bool()],
        "opacity":         features_scene_human["opacity"][preprocessed_mask.bool()] ,
        "scaling":         features_scene_human["scaling"][preprocessed_mask.bool()],
        "rotation":        features_scene_human["rotation"][preprocessed_mask.bool()]
    }

    features_scene = {
        "xyz":             features_scene_human["xyz"][~preprocessed_mask.bool()],
        "features_dc":     features_scene_human["features_dc"][~preprocessed_mask.bool()],
        "features_rest":   features_scene_human["features_rest"][~preprocessed_mask.bool()],
        "opacity":         features_scene_human["opacity"][~preprocessed_mask.bool()] ,
        "scaling":         features_scene_human["scaling"][~preprocessed_mask.bool()],
        "rotation":        features_scene_human["rotation"][~preprocessed_mask.bool()]
    }

    smpl_locs, smpl_visibility, smpl_vertices = find_smpl_to_gaussian_correspondence(data)
    

    c2w = torch.eye(4, device=device)
    c2w[0:3, 0:3] = torch.tensor(camera.R)
    c2w[0:3, 3] = torch.tensor(camera.T)
    human_gaussians_pixel_locs, human_gaussians_vis = find_gaussians_rendered_pixel_locs(features_human["xyz"], c2w, camera.FoVx, camera.FoVy, camera.image_height, camera.image_width, data["mask"])

    gauss_pixels = human_gaussians_pixel_locs


    # Step 1: Filter only visible SMPL vertices
    visible_indices = smpl_visibility.nonzero(as_tuple=True)[0]  # shape (V_visible,)
    visible_smpl_locs = smpl_locs[visible_indices]               # shape (V_visible, 2)
    # Step 2: Match visible SMPL vertices to closest gaussians
    dists = torch.cdist(visible_smpl_locs.float(), gauss_pixels.float(), p=2)  # (V_visible, G)
    min_dists, min_indices = dists.min(dim=1)  # (V_visible,)
    # Step 3: Threshold for valid matches
    threshold = 5.0  # pixels
    valid_matches = min_dists < threshold
    # Step 4: Build full-size mapping for all V SMPL vertices (init with -1s)
    matched_gaussian_indices = torch.full((smpl_locs.shape[0],), -1, dtype=torch.long, device=smpl_locs.device)
    # Fill in only visible and valid ones
    visible_valid_indices = visible_indices[valid_matches]
    matched_gaussian_indices[visible_valid_indices] = min_indices[valid_matches]
    
    '''
    usage:

    for v_idx, g_idx in enumerate(matched_gaussian_indices):
        if g_idx != -1:
            gaussian = features_human["xyz"][g_idx]
            smpl_vertex = smpl_locs[v_idx]
    '''

    valid_mask = matched_gaussian_indices != -1
    matches = matched_gaussian_indices[valid_mask]
    gauss_pixels = gauss_pixels[matches]



    set_gaussian_model_features(gaussians, features_scene_human)
    render_pkg = render_gaussians(gaussians, camera_list)
    image, _, _, _ = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
    torchvision.utils.save_image(image, f"./_irem/{folder_name}/render_step_{step}_cam_{image_idx}_exp_{exp_name}.png")

    preprocessed_rgb = load_and_preprocess_images([f"./_irem/{folder_name}/render_step_{step}_cam_{image_idx}_exp_{exp_name}.png"]).to(device)
    # preprocessed_rgb = load_and_preprocess_images([f"/local/home/idemir/Desktop/3dv_dynamichumansplatting/_irem/SINGLE_HEAD_W_OFFSET_4_IMAGE_10000_0.8_0.2_1.0/render_step_9500_cam_0_exp_SINGLE_HEAD_W_OFFSET_4_IMAGE_10000_0.8_0.2_1.0.png"]).to(device)
    # preprocessed_rgb = image
    rgb = preprocessed_rgb.squeeze(0).clone()  # (3,H,W)
    pts = smpl_locs[valid_mask]               # (M,2)
    if pts.numel() > 0:
        xs = pts[:, 0]       # x coords
        ys = pts[:, 1]       # y coords
        rgb[:, ys, xs] = 0.5
    torchvision.utils.save_image(rgb, f"./_irem/{image_idx}_overlay_matched_smpl_vertices.png")
    
    deformed_smpl_at_canonical, posed_smpl_at_canonical, knn_idx, smpl_scale, smpl_R, smpl_transl, smpl_global_orient, canonical_human_scales, canonical_human_rotation, lbs_T = get_deformed_human_using_image_correspondences(features_human["xyz"][matches], features_human["scaling"][matches], features_human["rotation"][matches], smpl_vertices, gauss_pixels, smpl_locs, valid_mask, data["betas"], data["body_pose"], data["global_orient"])


    #######
    gs_rotmat = quaternion_to_matrix(canonical_human_rotation)
    deformed_gs_rotmat = torch.inverse(lbs_T[valid_mask, :3, :3]) @ gs_rotmat
    deformed_gs_rotq = matrix_to_quaternion(deformed_gs_rotmat)
    render_scale = canonical_human_scales + smpl_scale.log()
    new_features = {
        "xyz":             deformed_smpl_at_canonical[0][valid_mask] * smpl_scale + smpl_transl,
        "features_dc":     features_human["features_dc"][matches],
        "features_rest":   features_human["features_rest"][matches],
        "opacity":         features_human["opacity"][matches] ,
        "scaling":         render_scale,
        "rotation":        deformed_gs_rotq
    }
    set_gaussian_model_features(gaussians, new_features)
    render_pkg = render_gaussians(gaussians, camera_list)
    image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
    if step % 100 == 0:
        #continue
        torchvision.utils.save_image(image, f"./_irem/{image_idx}_canonical_human.png")
    #######
    #######
    new_features = {
        "xyz":             features_human["xyz"][matches],
        "features_dc":     features_human["features_dc"][matches],
        "features_rest":   features_human["features_rest"][matches],
        "opacity":         features_human["opacity"][matches] ,
        "scaling":         features_human["scaling"][matches],
        "rotation":        features_human["rotation"][matches]
    }
    set_gaussian_model_features(gaussians, new_features)
    render_pkg = render_gaussians(gaussians, camera_list)
    image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
    if step % 100 == 0:
        #continue
        torchvision.utils.save_image(image, f"./_irem/{image_idx}_matched_gaussians.png")
    #######

    deformed_xyz, human_smpl_scales, deformed_gs_rotq, _ = get_deform_from_T_to_pose(deformed_smpl_at_canonical[0], canonical_human_scales, canonical_human_rotation, valid_mask, data["betas"], data["body_pose"], smpl_global_orient, smpl_scale, smpl_transl)
    
    #######
    new_features = {
        "xyz":             deformed_xyz[0][valid_mask],
        "features_dc":     features_human["features_dc"][matches],
        "features_rest":   features_human["features_rest"][matches],
        "opacity":         features_human["opacity"][matches] ,
        "scaling":         human_smpl_scales,
        "rotation":        deformed_gs_rotq
    }
    set_gaussian_model_features(gaussians, new_features)
    render_pkg = render_gaussians(gaussians, camera_list)
    image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
    if step % 100 == 0:
        #continue
        torchvision.utils.save_image(image, f"./_irem/{i}_canon2pose_human.png")
    #######

    #######
    merged_features = {
        "xyz":           torch.cat([features_scene["xyz"], new_features["xyz"]], dim=0),
        "features_dc":   torch.cat([features_scene["features_dc"], new_features["features_dc"]], dim=0),
        "features_rest": torch.cat([features_scene["features_rest"], new_features["features_rest"]], dim=0),
        "opacity":       torch.cat([features_scene["opacity"], new_features["opacity"]], dim=0),
        "scaling":       torch.cat([features_scene["scaling"], new_features["scaling"]], dim=0),
        "rotation":      torch.cat([features_scene["rotation"], new_features["rotation"]], dim=0)
    }
    set_gaussian_model_features(gaussians, merged_features)
    render_pkg = render_gaussians(gaussians, camera_list)
    image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
    if step % 100 == 0:
        #continue
        torchvision.utils.save_image(image, f"./_irem/{i}_canon2pose_human_with_scene.png")
    #######

    break


