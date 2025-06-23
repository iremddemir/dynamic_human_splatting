import sys

sys.path.append(".")

import torch
from vggt.models.our_vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.geometry import closed_form_inverse_se3
from instantsplat.scene import GaussianModel
from instantsplat.scene.gaussian_model_w_act import GaussianModel_With_Act
from instantsplat.scene.cameras import Camera
from instantsplat.gaussian_renderer import render
from tqdm import tqdm
from hugs.trainer.gs_trainer import (
    get_train_dataset,
    get_train_dataset_ordered,
    get_val_dataset,
)
from omegaconf import OmegaConf
from hugs.cfg.config import cfg as default_cfg
import torchvision
from instantsplat.arguments import PipelineParams, ArgumentParser
import random
from PIL import Image
from hugs.losses.utils import ssim
from hugs.utils.image import psnr
import matplotlib.pyplot as plt
import pickle
import rerun as rr

from utils import *


rr.init("corresponding_finder_2", recording_id="v0.1")
rr.connect_tcp("0.0.0.0:9876")


device = "cuda"
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

for param in model.parameters():
    param.requires_grad = False

for param in model.gs_head_feats.parameters():
    param.requires_grad = True
for param in model.aggregator.parameters():
    param.requires_grad = False

model.gs_head_feats.train()


from lpips import LPIPS

lpips = LPIPS(net="vgg", pretrained=True).to(device)
for param in lpips.parameters():
    param.requires_grad = False


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


def render_gaussians(gaussians, camera_list, bg="white"):
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
    elif bg == "green":
        bg = torch.tensor([0, 1, 0], dtype=torch.float32, device=device)

    render_pkg = render(viewpoint_cam, gaussians, pipe, bg, camera_pose=pose)

    return render_pkg


def render_and_save(
    gaussians, camera_list, features: dict, save_path: str, bg: str = "white"
):
    set_gaussian_model_features(gaussians, features)
    render_pkg = render_gaussians(gaussians, camera_list, bg=bg)
    torchvision.utils.save_image(render_pkg["render"], save_path)


def overlay_points_and_save(rgb_image, pts, save_path):
    rgb = rgb_image.squeeze(0).clone()

    if pts.numel() > 0:
        height, width = rgb.shape[-2], rgb.shape[-1]

        xs = pts[:, 0].long()
        ys = pts[:, 1].long()

        rgb[:, ys, xs] = 0.5

    torchvision.utils.save_image(rgb, save_path)


def render_smpl_gaussians(features, save_path, bg):
    R = np.array(
        [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
    )  # Camera looking straight at origin
    T = np.array([0.0, -0.35, 2.0])  # Camera 3 units away from origin (along Z)
    FoVx = np.radians(60)  # Horizontal FoV in radians
    FoVy = np.radians(60)  # Vertical FoV in radians
    gt_alpha_mask = None  # or a binary tensor of shape (1, H, W)

    # Instantiate Camera
    cam = Camera(
        colmap_id=1,
        R=R,
        T=T,
        FoVx=FoVx,
        FoVy=FoVy,
        image=torch.ones(3, 500, 500).to(device),
        gt_alpha_mask=gt_alpha_mask,
        image_name="smpl_render_cam",
        uid=0,
        trans=np.array([0.0, 0.0, 0.0]),  # for SMPL world
        scale=1.0,
    )

    gaussians = GaussianModel_With_Act(0)
    gaussians.init_RT_seq({1.0: [cam]})
    set_gaussian_model_features(gaussians, features)
    render_pkg = render_gaussians(gaussians, [cam], bg=bg)
    torchvision.utils.save_image(render_pkg["render"], save_path)


def render_smpl_gaussians_gif(features, save_path="smpl_360.gif"):
    from hugs.datasets.utils import get_rotating_camera

    camera_params = get_rotating_camera(
        dist=5.0,
        img_size=512,
        nframes=36,
        device="cuda",
        angle_limit=2 * torch.pi,
    )

    frames = []

    for cam_param in camera_params:
        # === Extract R and T from cam_ext ===
        cam_ext = cam_param["cam_ext"].T.cpu()  # (4x4 matrix)
        R = cam_ext[:3, :3].numpy()  # 3x3 rotation matrix
        T = cam_ext[:3, 3].numpy()  # 3D translation vector

        T[1] = -0.35

        # === Other parameters ===
        FoVx = cam_param["fovx"]
        FoVy = cam_param["fovy"]
        image = torch.ones(
            3, cam_param["image_height"], cam_param["image_width"]
        )  # dummy image
        gt_alpha_mask = None  # or you can load real mask
        image_name = "from_cam_param.png"
        uid = 0
        colmap_id = 0

        # === Instantiate Camera ===
        cam = Camera(
            colmap_id=colmap_id,
            R=R,
            T=T,
            FoVx=FoVx,
            FoVy=FoVy,
            image=image,
            gt_alpha_mask=gt_alpha_mask,
            image_name=image_name,
            uid=uid,
        )

        gaussians = GaussianModel_With_Act(0)
        gaussians.init_RT_seq({1.0: [cam]})
        set_gaussian_model_features(gaussians, features)
        render_pkg = render_gaussians(gaussians, [cam], bg="white")
        # Convert rendered tensor to PIL Image
        image_tensor = render_pkg["render"].detach().cpu().clamp(0, 1)
        image_pil = torchvision.transforms.functional.to_pil_image(image_tensor)
        frames.append(image_pil)

    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        duration=100,  # milliseconds per frame
        loop=0,  # loop forever
    )


cfg_file = "/local/home/idemir/Desktop/3dv_dynamichumansplatting/output/human_scene/neuman/lab/hugs_trimlp/demo-dataset.seq=lab/2025-03-30_19-55-08/config_train.yaml"
cfg_file = OmegaConf.load(cfg_file)
cfg = OmegaConf.merge(default_cfg, cfg_file)


# dataset = get_train_dataset(cfg)
dataset = get_val_dataset(cfg)


# dataset = get_train_dataset_ordered(cfg)
def get_data(idx):
    chosen_idx = dataset.val_split.index(idx)
    data = dataset[chosen_idx]
    return data


# image_ids = dataset.train_split[:20]
image_ids = dataset.val_split
image_names = [
    f"/local/home/idemir/Desktop/3dv_dynamichumansplatting/data/neuman/dataset/lab/images/{i:05}.png"
    for i in image_ids
]


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
exp_name = f"7_NOVEL_POSE_SYNTHESIS"

import os
from datetime import datetime

timestamp = datetime.now().strftime("%d-%m_%H-%M")
# folder_name = f"{timestamp}_{exp_name}"
folder_name = f"{exp_name}-{timestamp}"
os.makedirs(f"./_irem/{folder_name}", exist_ok=True)


#########################################################################################################################


per_frame_canonical_human = []


random.seed(1)


model.gs_head_feats.load_state_dict(torch.load("last_gs_head_feats_step.pth"))
for param in model.parameters():
    param.requires_grad = False
model.eval()


n_of_smpl_vertices = smpl().vertices.shape[1]


import pickle

with open("FIRST_GAUSSIANS_learned_overfitted.pkl", "rb") as f:
    global_merged_human = pickle.load(f)
global_merged_human_valid_mask = ~torch.isnan(global_merged_human["scaling"]).all(dim=1)

with open("global_scene_test.pkl", "rb") as f:
    global_scene = pickle.load(f)


from torchvision import transforms as TF

to_tensor = TF.ToTensor()


img_path = image_names[::1]
image_ids = image_ids[::1]
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


ssim_losses = []
lpips_losses = []
psnr_losses = []


from hugs.utils.rotations import matrix_to_axis_angle


for i in range(0, len(image_ids), 1):
    # img_path = [image_names[i]]
    # images = load_and_preprocess_images(img_path).to(device)[None]
    # aggregated_tokens_list, ps_idx = model.aggregator(images)
    # pose_enc = model.camera_head(aggregated_tokens_list)[-1]
    # extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
    # fov_h = pose_enc[..., 7]
    # fov_w = pose_enc[..., 8]

    # feats, _ = model.gs_head_feats(aggregated_tokens_list, images, ps_idx)

    # depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)
    # point_map = unproject_depth_map_to_point_map(depth_map.squeeze(0),
    #                                             extrinsic.squeeze(0),
    #                                             intrinsic.squeeze(0))[None]
    # point_map = torch.tensor(point_map, dtype=torch.float32, device=device)  # [1, B, H, W, 3]

    # # Forward pass through gs_head
    # scale   = feats[:,:,:,:,0:3]
    # rot     = feats[:,:,:,:,3:7]
    # sh      = feats[:,:,:,:,7:10]
    # op      = feats[:,:,:,:,10:11]
    # offset  = feats[:,:,:,:,11:14]

    print(i)

    b = i
    # b = 0

    image_idx = image_ids[i]

    cam_to_world_extrinsic = (
        closed_form_inverse_se3(extrinsic[0, b, :, :][None])[0].detach().cpu().numpy()
    )
    R_cam_to_world = cam_to_world_extrinsic[:3, :3]
    t_cam_to_world = cam_to_world_extrinsic[:3, 3]
    T_w2c = -R_cam_to_world.T @ t_cam_to_world
    camera = Camera(
        colmap_id=1,
        R=R_cam_to_world,
        T=T_w2c,  # Camera 3 units away from origin (along Z)
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

    features_human = {
        "xyz": features_scene_human["xyz"][preprocessed_mask.bool()],
        "features_dc": features_scene_human["features_dc"][preprocessed_mask.bool()],
        "features_rest": features_scene_human["features_rest"][
            preprocessed_mask.bool()
        ],
        "opacity": features_scene_human["opacity"][preprocessed_mask.bool()],
        "scaling": features_scene_human["scaling"][preprocessed_mask.bool()],
        "rotation": features_scene_human["rotation"][preprocessed_mask.bool()],
    }

    # features_scene = {
    #     "xyz":             features_scene_human["xyz"][~preprocessed_mask.bool()],
    #     "features_dc":     features_scene_human["features_dc"][~preprocessed_mask.bool()],
    #     "features_rest":   features_scene_human["features_rest"][~preprocessed_mask.bool()],
    #     "opacity":         features_scene_human["opacity"][~preprocessed_mask.bool()] ,
    #     "scaling":         features_scene_human["scaling"][~preprocessed_mask.bool()],
    #     "rotation":        features_scene_human["rotation"][~preprocessed_mask.bool()]
    # }

    features_scene = global_scene

    render_and_save(
        gaussians,
        camera_list,
        features_scene_human,
        save_path=f"./_irem/{folder_name}/{image_idx}_rendered_whole_scene.png",
        bg="white",
    )
    render_and_save(
        gaussians,
        camera_list,
        features_scene,
        save_path=f"./_irem/{folder_name}/{image_idx}_rendered_only_scene.png",
        bg="white",
    )
    # render_and_save(gaussians, camera_list, features_scene_human, save_path=f"./_irem/{folder_name}/{image_idx}_rendered_whole_scene_green.png", bg='green')
    # render_and_save(gaussians, camera_list, features_human, save_path=f"./_irem/{folder_name}/{image_idx}_human_gaussians_black.png", bg='black')
    # render_and_save(gaussians, camera_list, features_human, save_path=f"./_irem/{folder_name}/{image_idx}_human_gaussians_white.png", bg='white')

    # continue
    ############################################################################

    smpl_locs, smpl_visibility, smpl_vertices = find_smpl_to_gaussian_correspondence(
        data
    )

    ### if smpl vertex is out of mask, make it invisible
    # y_indices, x_indices = torch.where(preprocessed_mask.view((294, 518)) == 1)
    y_indices, x_indices = torch.where(preprocess_masks([data["mask"]])[0, 0] == 1)
    mask_pixels_locs = torch.stack((x_indices, y_indices), dim=1)  # shape (M, 2)

    eq = (
        (smpl_locs[:, None, :] == mask_pixels_locs[None, :, :]).all(dim=2).any(dim=1)
    )  # shape (V, M, 2)
    smpl_visibility = smpl_visibility & eq

    # preprocessed_rgb = load_and_preprocess_images([f"./_irem/{folder_name}/{image_idx}_rendered_whole_scene.png"]).to(device)
    # overlay_points_and_save(preprocessed_rgb, smpl_locs[~eq], f"./_irem/{folder_name}/{image_idx}_overlay_removed_smpl_vertices_by_mask.png")
    # overlay_points_and_save(preprocessed_rgb, mask_pixels_locs, f"./_irem/{folder_name}/{image_idx}_overlay_gt_human_mask.png")

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

    eq = (
        (human_gaussians_pixel_locs[:, None, :] == mask_pixels_locs[None, :, :])
        .all(dim=2)
        .any(dim=1)
    )  # shape (V, M, 2)
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

    # render_and_save(gaussians, camera_list, new_features, save_path=f"./_irem/{folder_name}/{image_idx}_human_gaussians_masked_black.png", bg='black')
    # render_and_save(gaussians, camera_list, new_features, save_path=f"./_irem/{folder_name}/{image_idx}_human_gaussians_masked_white.png", bg='white')

    ############################################################################

    """
    usage:

    for v_idx, g_idx in enumerate(matched_gaussian_indices):
        if g_idx != -1:
            gaussian = features_human["xyz"][g_idx]
            smpl_vertex = smpl_locs[v_idx]
    """

    # # # # # Step 1: Filter only visible SMPL vertices
    # # # # visible_indices = smpl_visibility.nonzero(as_tuple=True)[0]  # shape (V_visible,)
    # # # # visible_smpl_locs = smpl_locs[visible_indices]               # shape (V_visible, 2)
    # # # # # Step 2: Match visible SMPL vertices to closest gaussians
    # # # # dists = torch.cdist(visible_smpl_locs.float(), gauss_pixels.float(), p=2)  # (V_visible, G)
    # # # # min_dists, min_indices = dists.min(dim=1)  # (V_visible,)
    # # # # # Step 3: Threshold for valid matches
    # # # # threshold = 5.0  # pixels
    # # # # valid_matches = min_dists < threshold
    # # # # # Step 4: Build full-size mapping for all V SMPL vertices (init with -1s)
    # # # # matched_gaussian_indices = torch.full((smpl_locs.shape[0],), -1, dtype=torch.long, device=smpl_locs.device)
    # # # # # Fill in only visible and valid ones
    # # # # visible_valid_indices = visible_indices[valid_matches]
    # # # # matched_gaussian_indices[visible_valid_indices] = min_indices[valid_matches]

    # # # # valid_mask = matched_gaussian_indices != -1
    # # # # matches = matched_gaussian_indices[valid_mask]
    # # # # gauss_pixels = gauss_pixels[matches]

    """
        tersten bak, her bir gaussianı en yakın smpl vertexine ata ve threshold düşük olsun 3 mesela
    
        there are three ways to match for now, its sensitive
    """

    matched_gaussian_indices = match_smpl_to_gaussians_batched(
        smpl_locs, gauss_pixels, smpl_visibility, threshold=5.0, batch_size=40000
    )
    # matched_gaussian_indices = match_smpl_to_gaussians_vectorized(
    #     smpl_locs, gauss_pixels, smpl_visibility
    # )
    valid_mask = matched_gaussian_indices != -1
    matches = matched_gaussian_indices[valid_mask]
    gauss_pixels = gauss_pixels[matches]

    # preprocessed_rgb = load_and_preprocess_images([f"./_irem/{folder_name}/{image_idx}_rendered_whole_scene.png"]).to(device)
    # overlay_points_and_save(preprocessed_rgb, smpl_locs[valid_mask] , f"./_irem/{folder_name}/{image_idx}_overlay_matched_smpl_vertices.png")

    ############################################################################

    """
        find offsets from the following algorithm:
            1. for each gaussian, take the closest smpl vertex
            2. find that smpl vertex's gaussian match (matched_gaussian)
            3. distance = current_gaussian - matched_gaussian
            4. bu 3D distance vektörünü scale, R ve T ile dönüştür
    """

    # Step 1: Filter only visible SMPL vertices
    visible_indices = smpl_visibility.nonzero(as_tuple=True)[0]  # shape (V_visible,)
    visible_smpl_locs = smpl_locs[visible_indices]  # shape (V_visible, 2)
    # Step 2: Match visible SMPL vertices to closest gaussians
    matched_gaussian_mask = torch.zeros(
        human_gaussians_pixel_locs.shape[0],
        dtype=torch.bool,
        device=human_gaussians_pixel_locs.device,
    )
    matched_gaussian_mask[matches] = True

    # unmatched_human_gaussian_locs = human_gaussians_pixel_locs[human_gaussians_vis & ~matched_gaussian_mask]
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
    # render_and_save(gaussians, camera_list, unmatched_human_gaussian_features, save_path=f"./_irem/{folder_name}/{image_idx}_unmatched_human_gaussians.png", bg='white')
    # preprocessed_rgb = load_and_preprocess_images([f"./_irem/{folder_name}/{image_idx}_rendered_whole_scene.png"]).to(device)
    # overlay_points_and_save(preprocessed_rgb, unmatched_human_gaussian_locs , f"./_irem/{folder_name}/{image_idx}_overlay_unmatched_gaussians.png")

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
    # render_and_save(gaussians, camera_list, matched_human_gaussian_features, save_path=f"./_irem/{folder_name}/{image_idx}_matched_human_gaussians.png", bg='white')

    # for i, _matched_smpl_index in enumerate(matched_smpl_indices):
    #     if _matched_smpl_index == -1:
    #         continue
    #     _matched_gaussian_index = matched_gaussian_indices[_matched_smpl_index]
    #     _matched_gaussian_xyz = features_human["xyz"][_matched_gaussian_index]
    #     _current_gaussian_xyz = unmatched_human_gaussian_features["xyz"][i]

    #     offset = _matched_gaussian_xyz - _current_gaussian_xyz

    # Step 1: Filter valid unmatched Gaussians (those with a valid SMPL match)
    valid_unmatched_mask = matched_smpl_indices != -1
    valid_unmatched_indices = valid_unmatched_mask.nonzero(as_tuple=True)[0]

    # Step 2: Get matched SMPL vertex indices for each valid unmatched Gaussian
    valid_smpl_indices = matched_smpl_indices[
        valid_unmatched_indices
    ]  # shape (M_valid,)

    # Step 3: Use matched_gaussian_indices to get which Gaussian was originally matched to each SMPL
    matched_gaussian_indices_for_smpl = matched_gaussian_indices[
        valid_smpl_indices
    ]  # (M_valid,)

    # Step 4: Get 3D coordinates
    matched_gaussian_xyz = features_human["xyz"][
        matched_gaussian_indices_for_smpl
    ]  # (M_valid, 3)
    current_gaussian_xyz = unmatched_human_gaussian_features["xyz"][
        valid_unmatched_indices
    ]  # (M_valid, 3)

    # Step 5: Compute offsets
    # offsets = matched_gaussian_xyz - current_gaussian_xyz  # (M_valid, 3)
    # offsets = - matched_gaussian_xyz + current_gaussian_xyz  # (M_valid, 3)
    offsets = current_gaussian_xyz - matched_gaussian_xyz  # (M_valid, 3)

    ############################################################################

    """
        smpl_scale, smpl_R, and smpl_transl is from SMPL world to VGGT world
    """

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
    # render_smpl_gaussians(canonical_human_features_at_smpl, f"./_irem/{folder_name}/{image_idx}_canonical_human_black.png", "black")
    # render_smpl_gaussians(canonical_human_features_at_smpl, f"./_irem/{folder_name}/{image_idx}_canonical_human_white.png", "white")
    # render_smpl_gaussians_gif(canonical_human_features_at_smpl, f"./_irem/{folder_name}/{image_idx}_canonical_human_white.gif")

    offsets_canonical = torch.matmul(
        offsets / smpl_scale, smpl_R.T
    )  # no need for transl, because the offset is relative
    deformed_gs_rotq = transform_rotations_through_lbs(
        unmatched_human_gaussian_features["rotation"], lbs_T[valid_smpl_indices, :3, :3]
    )
    offsets_gaussians_canonical = {
        "xyz": deformed_smpl_at_canonical[0][valid_smpl_indices] + offsets_canonical,
        "features_dc": unmatched_human_gaussian_features["features_dc"],
        "features_rest": unmatched_human_gaussian_features["features_rest"],
        "opacity": unmatched_human_gaussian_features["opacity"],
        "scaling": unmatched_human_gaussian_features["scaling"] - smpl_scale.log(),
        "rotation": deformed_gs_rotq,
    }
    # render_smpl_gaussians(offsets_gaussians_canonical, f"./_irem/{folder_name}/{image_idx}_canonical_human_only_offsets_black.png", "black")
    # render_smpl_gaussians(offsets_gaussians_canonical, f"./_irem/{folder_name}/{image_idx}_canonical_human_only_offsets_white.png", "white")
    # render_smpl_gaussians_gif(offsets_gaussians_canonical, f"./_irem/{folder_name}/{image_idx}_canonical_human_only_offsets_white.gif")

    canonical_human_with_offset = {
        "xyz": torch.cat(
            [
                canonical_human_features_at_smpl["xyz"],
                offsets_gaussians_canonical["xyz"],
            ],
            dim=0,
        ),
        "features_dc": torch.cat(
            [
                canonical_human_features_at_smpl["features_dc"],
                offsets_gaussians_canonical["features_dc"],
            ],
            dim=0,
        ),
        "features_rest": torch.cat(
            [
                canonical_human_features_at_smpl["features_rest"],
                offsets_gaussians_canonical["features_rest"],
            ],
            dim=0,
        ),
        "opacity": torch.cat(
            [
                canonical_human_features_at_smpl["opacity"],
                offsets_gaussians_canonical["opacity"],
            ],
            dim=0,
        ),
        "scaling": torch.cat(
            [
                canonical_human_features_at_smpl["scaling"],
                offsets_gaussians_canonical["scaling"],
            ],
            dim=0,
        ),
        "rotation": torch.cat(
            [
                canonical_human_features_at_smpl["rotation"],
                offsets_gaussians_canonical["rotation"],
            ],
            dim=0,
        ),
    }
    # render_smpl_gaussians(canonical_human_with_offset, f"./_irem/{folder_name}/{image_idx}_canonical_human_with_offsets_black.png", "black")
    # render_smpl_gaussians(canonical_human_with_offset, f"./_irem/{folder_name}/{image_idx}_canonical_human_with_offsets_white.png", "white")
    # render_smpl_gaussians_gif(canonical_human_with_offset, f"./_irem/{folder_name}/{image_idx}_canonical_human_with_offsets_white.gif")

    # rr.set_time_seconds("frame", 0)
    # rr.log(f"world/scene", rr.Points3D(positions=features_scene["xyz"].detach().cpu().numpy(), colors=features_scene["features_dc"].squeeze(1).detach().cpu().numpy()))
    # rr.log(f"world/canonical_human_with_offset", rr.Points3D(positions=offsets_gaussians_canonical["xyz"].detach().cpu().numpy()))

    # # # # # # # # break

    ###### transform offsets to VGGT world

    offsets_gaussians_canonical_in_vggt_world = {
        "xyz": offsets_gaussians_canonical["xyz"] * smpl_scale + smpl_transl,
        "features_dc": offsets_gaussians_canonical["features_dc"],
        "features_rest": offsets_gaussians_canonical["features_rest"],
        "opacity": offsets_gaussians_canonical["opacity"],
        "scaling": offsets_gaussians_canonical["scaling"] + smpl_scale.log(),
        "rotation": offsets_gaussians_canonical["rotation"],
    }
    # render_and_save(gaussians, camera_list, offsets_gaussians_canonical_in_vggt_world, save_path=f"./_irem/{folder_name}/{image_idx}_canonical_offset_in_vggt_world_white.png", bg='white')

    # # # # smpl_model = smpl
    # # # # smpl_output = smpl(
    # # # #     betas=data["betas"].unsqueeze(0),
    # # # #     body_pose=data["body_pose"].unsqueeze(0),
    # # # #     global_orient=smpl_global_orient.unsqueeze(0),
    # # # #     disable_posedirs=False,
    # # # #     return_full_pose=True
    # # # # )
    # # # # verts = smpl_output.vertices[0]               # (V,3)
    # # # # faces = smpl_model.faces_tensor.to(verts.device)  # (F,3)
    # # # # face_verts = verts[faces]                     # (F,3,3)
    # # # # e1 = face_verts[:,1] - face_verts[:,0]        # (F,3)
    # # # # e2 = face_verts[:,2] - face_verts[:,0]        # (F,3)
    # # # # face_normals = torch.cross(e1, e2, dim=1)     # (F,3)
    # # # # face_normals = face_normals / face_normals.norm(dim=1, keepdim=True)  # unit-length
    # # # # vertex_normals = torch.zeros_like(verts)      # (V,3)
    # # # # idx = faces.view(-1)                          # (F*3)
    # # # # norms = face_normals.unsqueeze(1).repeat(1,3,1).view(-1,3)  # (F*3,3)
    # # # # vertex_normals = vertex_normals.index_add(0, idx, norms)
    # # # # vertex_normals = vertex_normals / vertex_normals.norm(dim=1, keepdim=True)
    # # # # gauss_offsets = vertex_normals * 2.0                   # (M,3)
    # # # # deformed_xyz, human_smpl_scales, deformed_gs_rotq, lbs_T = get_deform_from_T_to_pose(
    # # # #     deformed_smpl_at_canonical[0] + gauss_offsets,
    # # # #     canonical_human_scales,
    # # # #     canonical_human_rotation,
    # # # #     valid_mask,
    # # # #     data["betas"], data["body_pose"], smpl_global_orient, smpl_scale, smpl_transl
    # # # # )
    # # # # #######
    # # # # new_features = {
    # # # #     "xyz":             deformed_xyz[0][valid_mask],
    # # # #     "features_dc":     features_human["features_dc"][matches],
    # # # #     "features_rest":   features_human["features_rest"][matches],
    # # # #     "opacity":         features_human["opacity"][matches],
    # # # #     "scaling":         human_smpl_scales,
    # # # #     "rotation":        deformed_gs_rotq
    # # # # }
    # # # # render_and_save(gaussians, camera_list, new_features, save_path=f"./_irem/{folder_name}/{image_idx}_canon2pose_human.png", bg='white')

    # novel_pose = matrix_to_axis_angle(
    #     torch.tensor([[[[ 9.6346e-01, -1.5875e-01, -2.1574e-01],
    #       [ 2.2399e-01,  3.5793e-02,  9.7393e-01],
    #       [-1.4689e-01, -9.8667e-01,  7.0044e-02]],

    #      [[ 9.7650e-01,  2.1531e-01, -8.8753e-03],
    #       [-2.1396e-01,  9.7364e-01,  7.8998e-02],
    #       [ 2.5651e-02, -7.5243e-02,  9.9684e-01]],

    #      [[ 9.7680e-01,  1.8435e-01,  1.0903e-01],
    #       [-1.0325e-01,  8.5134e-01, -5.1436e-01],
    #       [-1.8764e-01,  4.9117e-01,  8.5062e-01]],

    #      [[ 9.3551e-01,  3.4248e-01, -8.6700e-02],
    #       [-7.5509e-02, -4.5903e-02, -9.9609e-01],
    #       [-3.4512e-01,  9.3840e-01, -1.7082e-02]],

    #      [[ 9.1845e-01,  2.8407e-01,  2.7525e-01],
    #       [ 2.3087e-01,  1.8003e-01, -9.5618e-01],
    #       [-3.2118e-01,  9.4175e-01,  9.9763e-02]],

    #      [[ 9.9936e-01,  1.7771e-02, -3.1023e-02],
    #       [-1.4590e-02,  9.9489e-01,  9.9909e-02],
    #       [ 3.2640e-02, -9.9393e-02,  9.9451e-01]],

    #      [[ 9.6838e-01,  1.4283e-01,  2.0456e-01],
    #       [-1.6607e-01,  9.8090e-01,  1.0127e-01],
    #       [-1.8619e-01, -1.3204e-01,  9.7360e-01]],

    #      [[ 9.9304e-01, -5.0036e-02, -1.0666e-01],
    #       [ 7.6798e-02,  9.6147e-01,  2.6397e-01],
    #       [ 8.9338e-02, -2.7033e-01,  9.5861e-01]],

    #      [[ 9.9973e-01, -1.7909e-02,  1.4645e-02],
    #       [ 1.9478e-02,  9.9316e-01, -1.1513e-01],
    #       [-1.2483e-02,  1.1538e-01,  9.9324e-01]],

    #      [[ 8.9698e-01, -4.3841e-01,  5.6772e-02],
    #       [ 3.9901e-01,  8.5818e-01,  3.2297e-01],
    #       [-1.9032e-01, -2.6705e-01,  9.4470e-01]],

    #      [[ 9.6664e-01,  2.3492e-01, -1.0208e-01],
    #       [-1.7506e-01,  8.9684e-01,  4.0624e-01],
    #       [ 1.8698e-01, -3.7482e-01,  9.0805e-01]],

    #      [[ 9.9313e-01,  1.0079e-01,  5.9479e-02],
    #       [-1.0423e-01,  9.9287e-01,  5.7817e-02],
    #       [-5.3227e-02, -6.3619e-02,  9.9655e-01]],

    #      [[ 9.6664e-01,  7.6863e-02,  2.4434e-01],
    #       [-1.1027e-01,  9.8587e-01,  1.2613e-01],
    #       [-2.3119e-01, -1.4887e-01,  9.6145e-01]],

    #      [[ 9.5959e-01, -1.7155e-01, -2.2304e-01],
    #       [ 1.9706e-01,  9.7553e-01,  9.7502e-02],
    #       [ 2.0086e-01, -1.3752e-01,  9.6992e-01]],

    #      [[ 9.9798e-01,  5.6458e-02,  2.9220e-02],
    #       [-5.4506e-02,  9.9648e-01, -6.3760e-02],
    #       [-3.2716e-02,  6.2039e-02,  9.9754e-01]],

    #      [[ 6.7489e-01,  7.3792e-01, -5.2311e-04],
    #       [-7.2199e-01,  6.6047e-01,  2.0620e-01],
    #       [ 1.5250e-01, -1.3878e-01,  9.7851e-01]],

    #      [[ 7.4488e-01, -6.5108e-01,  1.4578e-01],
    #       [ 6.1037e-01,  7.5321e-01,  2.4518e-01],
    #       [-2.6944e-01, -9.3645e-02,  9.5845e-01]],

    #      [[-5.7708e-01, -2.1265e-01, -7.8852e-01],
    #       [ 6.0514e-01,  5.3703e-01, -5.8770e-01],
    #       [ 5.4843e-01, -8.1632e-01, -1.8123e-01]],

    #      [[-3.5511e-01,  2.2365e-01,  9.0768e-01],
    #       [-6.3629e-01,  6.5351e-01, -4.0996e-01],
    #       [-6.8486e-01, -7.2312e-01, -8.9762e-02]],

    #      [[ 9.4111e-01, -3.3768e-01, -1.6775e-02],
    #       [ 3.3555e-01,  9.2679e-01,  1.6873e-01],
    #       [-4.1430e-02, -1.6442e-01,  9.8552e-01]],

    #      [[ 9.3789e-01,  3.4690e-01, -5.2201e-03],
    #       [-3.4251e-01,  9.2821e-01,  1.4530e-01],
    #       [ 5.5249e-02, -1.3448e-01,  9.8937e-01]],

    #      [[ 9.6229e-01,  2.5160e-01, -1.0342e-01],
    #       [-2.4957e-01,  9.6782e-01,  3.2374e-02],
    #       [ 1.0824e-01, -5.3435e-03,  9.9411e-01]],

    #      [[ 9.6433e-01, -2.6084e-01,  4.4997e-02],
    #       [ 2.5567e-01,  9.6191e-01,  9.6743e-02],
    #       [-6.8517e-02, -8.1789e-02,  9.9429e-01]]]])
    # ).to(device)
    # novel_pose = matrix_to_axis_angle(
    #     torch.tensor(
    #         [[[[ 0.5621, -0.7613,  0.3234],
    #       [ 0.7451,  0.6358,  0.2017],
    #       [-0.3591,  0.1276,  0.9245]],

    #      [[ 0.7837,  0.6209, -0.0142],
    #       [-0.5402,  0.6927,  0.4779],
    #       [ 0.3066, -0.3668,  0.8783]],

    #      [[ 0.9827,  0.0428, -0.1804],
    #       [-0.1008,  0.9400, -0.3260],
    #       [ 0.1556,  0.3386,  0.9280]],

    #      [[ 0.9010, -0.0892, -0.4245],
    #       [-0.0470,  0.9528, -0.2998],
    #       [ 0.4312,  0.2901,  0.8543]],

    #      [[ 0.9167,  0.1360,  0.3757],
    #       [-0.1629,  0.9858,  0.0404],
    #       [-0.3649, -0.0982,  0.9259]],

    #      [[ 0.9565,  0.2866,  0.0535],
    #       [-0.2868,  0.8921,  0.3492],
    #       [ 0.0523, -0.3493,  0.9355]],

    #      [[ 0.8007,  0.5740,  0.1713],
    #       [-0.5281,  0.8114, -0.2503],
    #       [-0.2827,  0.1100,  0.9529]],

    #      [[ 0.8477, -0.4617, -0.2612],
    #       [ 0.4321,  0.8866, -0.1649],
    #       [ 0.3077,  0.0270,  0.9511]],

    #      [[ 0.9838,  0.1752, -0.0381],
    #       [-0.1656,  0.9696,  0.1801],
    #       [ 0.0685, -0.1709,  0.9829]],

    #      [[ 0.8235, -0.5457, -0.1549],
    #       [ 0.5669,  0.7815,  0.2607],
    #       [-0.0212, -0.3025,  0.9529]],

    #      [[ 0.7840,  0.6158,  0.0778],
    #       [-0.6204,  0.7815,  0.0660],
    #       [-0.0202, -0.1000,  0.9948]],

    #      [[ 0.9247,  0.3780, -0.0462],
    #       [-0.2972,  0.7922,  0.5329],
    #       [ 0.2380, -0.4791,  0.8449]],

    #      [[ 0.9681, -0.0842,  0.2360],
    #       [ 0.0630,  0.9934,  0.0961],
    #       [-0.2425, -0.0782,  0.9670]],

    #      [[ 0.8875,  0.4590, -0.0411],
    #       [-0.4392,  0.8695,  0.2261],
    #       [ 0.1395, -0.1826,  0.9732]],

    #      [[ 0.9973,  0.0529,  0.0511],
    #       [-0.0673,  0.9366,  0.3440],
    #       [-0.0296, -0.3465,  0.9376]],

    #      [[ 0.9700,  0.0704,  0.2325],
    #       [-0.1285,  0.9609,  0.2452],
    #       [-0.2062, -0.2678,  0.9412]],

    #      [[ 0.8528,  0.4300,  0.2963],
    #       [-0.5150,  0.7864,  0.3411],
    #       [-0.0863, -0.4435,  0.8921]],

    #      [[ 0.9084, -0.0924, -0.4078],
    #       [ 0.2471,  0.9053,  0.3454],
    #       [ 0.3373, -0.4145,  0.8452]],

    #      [[ 0.7128, -0.0637,  0.6984],
    #       [-0.2362,  0.9159,  0.3246],
    #       [-0.6604, -0.3964,  0.6378]],

    #      [[ 0.9935,  0.0831, -0.0772],
    #       [-0.0718,  0.9877,  0.1385],
    #       [ 0.0877, -0.1321,  0.9873]],

    #      [[ 0.9878,  0.0524,  0.1463],
    #       [-0.0338,  0.9913, -0.1268],
    #       [-0.1517,  0.1203,  0.9811]],

    #      [[ 0.9995, -0.0265,  0.0154],
    #       [ 0.0258,  0.9988,  0.0419],
    #       [-0.0165, -0.0415,  0.9990]],

    #      [[ 0.9989, -0.0349,  0.0317],
    #       [ 0.0345,  0.9993,  0.0148],
    #       [-0.0322, -0.0136,  0.9994]]]]
    #     )
    # ).to(device)

    novel_pose = matrix_to_axis_angle(
        torch.tensor(
            [
                [
                    [
                        [7.0445e-01, -6.7872e-01, 2.0755e-01],
                        [3.2050e-01, 5.6512e-01, 7.6021e-01],
                        [-6.3326e-01, -4.6901e-01, 6.1563e-01],
                    ],
                    [
                        [7.1024e-01, 6.5662e-01, -2.5378e-01],
                        [-3.1784e-01, 6.2079e-01, 7.1666e-01],
                        [6.2812e-01, -4.2834e-01, 6.4961e-01],
                    ],
                    [
                        [9.9994e-01, 1.0410e-02, 3.8924e-03],
                        [-4.7218e-03, 7.1496e-01, -6.9915e-01],
                        [-1.0061e-02, 6.9909e-01, 7.1496e-01],
                    ],
                    [
                        [7.3691e-01, 6.4022e-01, -2.1699e-01],
                        [3.1765e-01, -6.1130e-01, -7.2486e-01],
                        [-5.9671e-01, 4.6522e-01, -6.5383e-01],
                    ],
                    [
                        [7.7849e-01, -6.0400e-01, 1.7070e-01],
                        [-2.2511e-01, -5.2256e-01, -8.2235e-01],
                        [5.8590e-01, 6.0177e-01, -5.4277e-01],
                    ],
                    [
                        [9.9936e-01, 2.0739e-02, 2.9057e-02],
                        [-1.7878e-02, 9.9527e-01, -9.5466e-02],
                        [-3.0900e-02, 9.4886e-02, 9.9501e-01],
                    ],
                    [
                        [9.4024e-01, -1.2714e-01, 3.1590e-01],
                        [4.4542e-02, 9.6563e-01, 2.5608e-01],
                        [-3.3760e-01, -2.2670e-01, 9.1358e-01],
                    ],
                    [
                        [9.5037e-01, 1.4720e-01, -2.7409e-01],
                        [-6.6243e-02, 9.5653e-01, 2.8401e-01],
                        [3.0398e-01, -2.5175e-01, 9.1881e-01],
                    ],
                    [
                        [9.9997e-01, -6.7080e-04, 7.3098e-03],
                        [6.9462e-04, 9.9999e-01, -3.2561e-03],
                        [-7.3075e-03, 3.2611e-03, 9.9997e-01],
                    ],
                    [
                        [9.8261e-01, -3.8094e-02, 1.8171e-01],
                        [-2.8237e-02, 9.3668e-01, 3.4905e-01],
                        [-1.8350e-01, -3.4812e-01, 9.1932e-01],
                    ],
                    [
                        [9.7594e-01, 6.8619e-02, -2.0697e-01],
                        [5.4955e-03, 9.4115e-01, 3.3794e-01],
                        [2.1798e-01, -3.3095e-01, 9.1813e-01],
                    ],
                    [
                        [9.9938e-01, 1.9222e-02, -2.9469e-02],
                        [-2.3807e-02, 9.8615e-01, -1.6413e-01],
                        [2.5906e-02, 1.6473e-01, 9.8600e-01],
                    ],
                    [
                        [8.5860e-01, 5.1000e-01, -5.1934e-02],
                        [-5.1222e-01, 8.5758e-01, -4.6735e-02],
                        [2.0702e-02, 6.6729e-02, 9.9756e-01],
                    ],
                    [
                        [8.7403e-01, -4.7235e-01, 1.1380e-01],
                        [4.8112e-01, 8.7408e-01, -6.7124e-02],
                        [-6.7765e-02, 1.1342e-01, 9.9123e-01],
                    ],
                    [
                        [9.9974e-01, -1.6977e-02, 1.5168e-02],
                        [1.7053e-02, 9.9984e-01, -4.8890e-03],
                        [-1.5083e-02, 5.1464e-03, 9.9987e-01],
                    ],
                    [
                        [5.9719e-01, 8.0168e-01, 2.6015e-02],
                        [-7.7894e-01, 5.7189e-01, 2.5729e-01],
                        [1.9139e-01, -1.7391e-01, 9.6598e-01],
                    ],
                    [
                        [5.7158e-01, -8.2033e-01, -1.9145e-02],
                        [8.0702e-01, 5.5778e-01, 1.9389e-01],
                        [-1.4837e-01, -1.2627e-01, 9.8084e-01],
                    ],
                    [
                        [4.3684e-01, -5.1695e-01, -7.3616e-01],
                        [6.1983e-01, 7.6607e-01, -1.7014e-01],
                        [6.5191e-01, -3.8197e-01, 6.5507e-01],
                    ],
                    [
                        [5.0582e-01, 4.6047e-01, 7.2946e-01],
                        [-5.0412e-01, 8.4398e-01, -1.8319e-01],
                        [-7.0000e-01, -2.7508e-01, 6.5904e-01],
                    ],
                    [
                        [9.6313e-01, -1.5567e-01, 2.1942e-01],
                        [2.3808e-01, 8.7297e-01, -4.2571e-01],
                        [-1.2528e-01, 4.6226e-01, 8.7785e-01],
                    ],
                    [
                        [9.7014e-01, 1.2909e-01, -2.0534e-01],
                        [-2.0783e-01, 8.7888e-01, -4.2938e-01],
                        [1.2505e-01, 4.5924e-01, 8.7947e-01],
                    ],
                    [
                        [9.5777e-01, 2.3075e-01, -1.7156e-01],
                        [-1.9261e-01, 9.5787e-01, 2.1305e-01],
                        [2.1350e-01, -1.7101e-01, 9.6186e-01],
                    ],
                    [
                        [9.5461e-01, -2.4600e-01, 1.6794e-01],
                        [1.9588e-01, 9.4324e-01, 2.6820e-01],
                        [-2.2438e-01, -2.2313e-01, 9.4861e-01],
                    ],
                ]
            ]
        )
    ).to(device)

    data["body_pose"] = novel_pose.view(-1)

    deformed_xyz, human_smpl_scales, deformed_gs_rotq, lbs_T = (
        get_deform_from_T_to_pose(
            global_merged_human["xyz"],
            global_merged_human["scaling"][global_merged_human_valid_mask],
            global_merged_human["rotation"][global_merged_human_valid_mask],
            global_merged_human_valid_mask,
            data["betas"],
            data["body_pose"],
            smpl_global_orient,
            smpl_scale,
            smpl_transl,
        )
    )

    # camera = Camera(
    #     colmap_id=1,
    #     R=R_cam_to_world @ R_cam_to_world,
    #     T=T_w2c,  # Camera 3 units away from origin (along Z)
    #     FoVx=fov_w[0, b],
    #     FoVy=fov_h[0, b],
    #     image=to_tensor(Image.open(img_path[b]).convert('RGB')),
    #     gt_alpha_mask=None,
    #     image_name=f'{image_idx:05}',
    #     uid=0
    # )
    # camera_list = [camera]

    # deformed_xyz, human_smpl_scales, deformed_gs_rotq, lbs_T = get_deform_from_T_to_pose(
    #     deformed_smpl_at_canonical[0],
    #     canonical_human_scales,
    #     canonical_human_rotation,
    #     valid_mask,
    #     data["betas"], data["body_pose"], smpl_global_orient, smpl_scale, smpl_transl
    # )

    #######
    new_features = {
        "xyz": deformed_xyz[0][global_merged_human_valid_mask],
        "features_dc": global_merged_human["features_dc"][
            global_merged_human_valid_mask
        ],
        "features_rest": global_merged_human["features_rest"][
            global_merged_human_valid_mask
        ],
        "opacity": global_merged_human["opacity"][global_merged_human_valid_mask],
        "scaling": human_smpl_scales,
        "rotation": deformed_gs_rotq,
    }
    render_and_save(
        gaussians,
        camera_list,
        new_features,
        save_path=f"./_irem/{folder_name}/{image_idx}_canon2pose_human.png",
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
        save_path=f"./_irem/{folder_name}/{image_idx}_canon2pose_human_with_scene_white.png",
        bg="white",
    )
    render_and_save(
        gaussians,
        camera_list,
        merged_features,
        save_path=f"./_irem/{folder_name}/{image_idx}_canon2pose_human_with_scene_green.png",
        bg="green",
    )

    break
