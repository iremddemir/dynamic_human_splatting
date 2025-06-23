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
# images_list, tokens_list, ps_idx_list, data_list, camera_list = [], [], [], [], []

# for img_path, i in zip(image_names, image_ids):
#     print(f"Processing image {i+1}/{len(image_names)}: {img_path}")

#     image = load_and_preprocess_images([img_path]).to(device)
#     with torch.no_grad():
#         with torch.cuda.amp.autocast(dtype=dtype):
#             image = image[None]  # add batch dimension
#             images_list.append(image)
#             aggregated_tokens_list, ps_idx = model.aggregator(image)
#             tokens_list.append(aggregated_tokens_list)
#             ps_idx_list.append(ps_idx)
#         data = dataset[i]
#     data_list.append(data)


# images_list, tokens_list, ps_idx_list, data_list, camera_list = [], [], [], [], []

# for img_path, i in zip(image_names, image_ids):
#     print(f"Processing image {i+1}/{len(image_names)}: {img_path}")
#     data = dataset[i]
#     data_list.append(data)

# images_list = load_and_preprocess_images(image_names).to(device)
# with torch.no_grad():
#     with torch.cuda.amp.autocast(dtype=dtype):
#         images_list = images_list[None]  # add batch dimension
#         tokens_list, ps_idx_list = model.aggregator(images_list)

# # ####################################################################################################

# final_conv_layer = model.gs_head_feats.scratch.output_conv2[-1]

# splits_and_inits = [
#     (3, 0.00003, -7.0),  # Scales
#     (4, 1.0, 0.0),  # Rotations
#     (3, 1.0, 0.0),  # Spherical Harmonics
#     (1, 1.0, 0.0),  # Opacity
#     (3, 0.00001, 0.0),  # 3D mean offsets
# ]

# start = 0
# for out_channel, std, bias_val in splits_and_inits:
#     end = start + out_channel

#     # Xavier init for weight slice
#     torch.nn.init.xavier_uniform_(
#         final_conv_layer.weight[start:end],
#         gain=std
#     )

#     # Constant init for bias slice
#     torch.nn.init.constant_(
#         final_conv_layer.bias[start:end],
#         bias_val
#     )

#     start = end


# print("FINAL LAYER INITS")

# optimizer = torch.optim.AdamW(
#     [
#         {"params": model.gs_head_feats.parameters(), "lr": 0.00005}
#     ],
#     lr=2e-4,
#     betas=(0.9, 0.999),
#     eps=1e-15,
#     weight_decay=0.0
# )


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
exp_name = f"2_2_DUMPING_CANON_HUMAN"

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


# model.gs_head_feats.load_state_dict(torch.load("/local/home/idemir/Desktop/3dv_dynamichumansplatting/_irem/SINGLE_HEAD_W_OFFSET_4_IMAGE_10000_0.8_0.2_1.0/gs_head_feats.pth"))
# model.gs_head_feats.load_state_dict(torch.load("/local/home/idemir/Desktop/3dv_dynamichumansplatting/_irem/10_MAY_SINGLE_HEAD_W_OFFSET_4_IMAGE_10000_0.8_0.2_1.0/gs_head_feats.pth"))
# model.gs_head_feats.load_state_dict(torch.load("/local/home/idemir/Desktop/3dv_dynamichumansplatting/_irem/10_MAY_SINGLE_HEAD_W_OFFSET_32_IMAGE_10000_0.8_0.2_1.0/gs_head_feats.pth"))
# model.gs_head_feats.load_state_dict(torch.load("/local/home/idemir/Desktop/3dv_dynamichumansplatting/_irem/son_10_MAY_I_FIXED_BUG_SINGLE_HEAD_W_OFFSET_26_IMAGE_10000_0.8_0.2_1.0/gs_head_feats.pth"))


# model.gs_head_feats.load_state_dict(torch.load("/local/home/idemir/Desktop/3dv_dynamichumansplatting/_irem/12-05_09-54-I_FIXED_BUG_SINGLE_HEAD_W_OFFSET_82_IMAGE_15000_0.8_0.2_1.0/gs_head_feats.pth"))
model.gs_head_feats.load_state_dict(
    torch.load(
        "/local/home/idemir/Desktop/3dv_dynamichumansplatting/_irem/1_GS_TRAINING_SINGLE_HEAD_W_OFFSET_82_IMAGE_15000_0.8_0.2_1.0-WHITE_BG-12-05_12-10/gs_head_feats.pth"
    )
)


for param in model.parameters():
    param.requires_grad = False
model.eval()


# n_of_smpl_vertices = smpl().vertices.shape[1]
# global_merged_human = {
#     "xyz":             torch.zeros((n_of_smpl_vertices, 3), device="cpu"),
#     "features_dc":     torch.zeros((n_of_smpl_vertices, 1, 3), device="cpu"),
#     "features_rest":   torch.zeros((n_of_smpl_vertices, 15, 3), device="cpu"),
#     "opacity":         torch.zeros((n_of_smpl_vertices, 1), device="cpu"),
#     "scaling":         torch.zeros((n_of_smpl_vertices, 3), device="cpu"),
#     "rotation":        torch.zeros((n_of_smpl_vertices, 4), device="cpu")
# }
# # all_valid_masks = torch.zeros((n_of_smpl_vertices,))
# all_valid_masks = torch.ones((n_of_smpl_vertices,), dtype=torch.bool, device="cuda")


per_frame_canonical_human = []


for step in tqdm(range(iterations), desc="Training", total=iterations):
    # GET THE DATA POINT:
    # i = random.randint(0, images_list.shape[1] - 1)

    # i = random.randint(0, len(image_ids) - 1)

    # for i in range(0, len(image_ids), 2):
    for i in range(0, len(image_ids), 2):
        print(i)

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
        scale = feats[:, :, :, :, 0:3]
        rot = feats[:, :, :, :, 3:7]
        sh = feats[:, :, :, :, 7:10]
        op = feats[:, :, :, :, 10:11]
        offset = feats[:, :, :, :, 11:14]
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
        camera = torch.load(
            f"/local/home/idemir/Desktop/3dv_dynamichumansplatting/instantsplat/lab_{image_idx:05}/viewpoint_cam_0.pth"
        )
        # camera = torch.load("viewpoint_cam.pth")
        camera_list = [camera]

        gaussians = GaussianModel_With_Act(0)
        gaussians.init_RT_seq({1.0: camera_list})

        features_scene_human = {
            "xyz": xyz,
            "features_dc": sh[0, 0, :, :, 0:3].view(-1, 1, 3),
            "features_rest": torch.zeros(
                (sh[0, 0, :, :, 0:3].detach().view(-1, 3, 1).shape[0], 15, 3),
                device=xyz.device,
            ),
            "opacity": op[0, 0, :, :, 0].view(-1, 1),
            "scaling": scale[0, 0, :, :, 0:3].view(-1, 3),
            "rotation": rot[0, 0, :, :, 0:4].view(-1, 4),
        }
        set_gaussian_model_features(gaussians, features_scene_human)
        render_pkg = render_gaussians(gaussians, camera_list)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )
        if step % 100 == 0:
            torchvision.utils.save_image(
                image, f"./_irem/{folder_name}/{image_idx}_whole_scene.png"
            )

        # with open(f"./_irem/__scenes/features_scene_human_{image_idx}.pkl", "wb") as f:
        #     pickle.dump(features_scene_human, f)
        # continue

        preprocessed_mask = preprocess_masks([data["mask"]]).view(-1)

        # mask_tensor = torchvision.io.read_image(
        #     "/local/home/idemir/Desktop/3dv_dynamichumansplatting/data/neuman/dataset/lab/4d_humans/sam_segmentations/mask_0000.png",
        #     torchvision.io.ImageReadMode.GRAY
        # ).gt(0).to(device=device, dtype=torch.float32).squeeze(0)
        # preprocessed_mask = preprocess_masks([mask_tensor]).view(-1)

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

        # rr.set_time_seconds("frame", 0)
        # rr.log(f"world/xyz", rr.Arrows3D(vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]))
        # rr.log(f"world/scene{image_idx}", rr.Points3D(positions=features_scene["xyz"].detach().cpu().numpy(), colors=features_scene["features_dc"].squeeze(1).detach().cpu().numpy()))

        set_gaussian_model_features(gaussians, features_human)
        render_pkg = render_gaussians(gaussians, camera_list, bg="black")
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )
        torchvision.utils.save_image(
            image, f"./_irem/{folder_name}/{image_idx}_human_gaussians_black.png"
        )

        set_gaussian_model_features(gaussians, features_human)
        render_pkg = render_gaussians(gaussians, camera_list, bg="white")
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )
        torchvision.utils.save_image(
            image, f"./_irem/{folder_name}/{image_idx}_human_gaussians_white.png"
        )

        ###################
        smpl_locs, smpl_visibility, smpl_vertices = (
            find_smpl_to_gaussian_correspondence(data)
        )

        c2w = torch.eye(4, device=device)
        c2w[0:3, 0:3] = torch.tensor(camera.R)
        c2w[0:3, 3] = torch.tensor(camera.T)
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

        # y_indices, x_indices = torch.where(preprocessed_mask.view((294, 518)) == 1)
        # mask_pixels_locs = torch.stack((x_indices, y_indices), dim=1)
        # eq = (human_gaussians_pixel_locs[:, None, :] == mask_pixels_locs[None, :, :])  # shape [N, M, 2]
        # matches = eq.all(dim=2).any(dim=1)  # [N]
        # gauss_pixels = human_gaussians_pixel_locs[matches]

        gauss_pixels = human_gaussians_pixel_locs

        # Step 1: Filter only visible SMPL vertices
        visible_indices = smpl_visibility.nonzero(as_tuple=True)[
            0
        ]  # shape (V_visible,)
        visible_smpl_locs = smpl_locs[visible_indices]  # shape (V_visible, 2)
        # Step 2: Match visible SMPL vertices to closest gaussians
        dists = torch.cdist(
            visible_smpl_locs.float(), gauss_pixels.float(), p=2
        )  # (V_visible, G)
        min_dists, min_indices = dists.min(dim=1)  # (V_visible,)
        # Step 3: Threshold for valid matches
        threshold = 5.0  # pixels
        valid_matches = min_dists < threshold
        # Step 4: Build full-size mapping for all V SMPL vertices (init with -1s)
        matched_gaussian_indices = torch.full(
            (smpl_locs.shape[0],), -1, dtype=torch.long, device=smpl_locs.device
        )
        # Fill in only visible and valid ones
        visible_valid_indices = visible_indices[valid_matches]
        matched_gaussian_indices[visible_valid_indices] = min_indices[valid_matches]

        """
        usage:

        for v_idx, g_idx in enumerate(matched_gaussian_indices):
            if g_idx != -1:
                gaussian = features_human["xyz"][g_idx]
                smpl_vertex = smpl_locs[v_idx]
        """

        valid_mask = matched_gaussian_indices != -1
        matches = matched_gaussian_indices[valid_mask]
        gauss_pixels = gauss_pixels[matches]

        preprocessed_rgb = load_and_preprocess_images(
            [f"./_irem/{folder_name}/{image_idx}_whole_scene.png"]
        ).to(device)
        # preprocessed_rgb = load_and_preprocess_images([f"/local/home/idemir/Desktop/3dv_dynamichumansplatting/_irem/SINGLE_HEAD_W_OFFSET_4_IMAGE_10000_0.8_0.2_1.0/render_step_9500_cam_0_exp_SINGLE_HEAD_W_OFFSET_4_IMAGE_10000_0.8_0.2_1.0.png"]).to(device)
        # preprocessed_rgb = image
        rgb = preprocessed_rgb.squeeze(0).clone()  # (3,H,W)
        pts = smpl_locs[valid_mask]  # (M,2)
        if pts.numel() > 0:
            xs = pts[:, 0]  # x coords
            ys = pts[:, 1]  # y coords
            rgb[:, ys, xs] = 0.5
        torchvision.utils.save_image(
            rgb, f"./_irem/{folder_name}/{image_idx}_overlay_matched_smpl_vertices.png"
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
        ) = get_deformed_human_using_image_correspondences(
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
        if (
            (
                torch.isnan(canonical_human_rotation)
                | torch.isinf(canonical_human_rotation)
            )
            .any()
            .item()
        ):
            print("⚠️ ROTAION IS NOT CORRECT!")
            exit()

        # '''
        #     aslında olan alttaki line -- merged human oluşturma
        # '''
        # global_merged_human["xyz"] = deformed_smpl_at_canonical[0].to("cpu")
        # global_merged_human["scaling"][valid_mask] = canonical_human_scales.to("cpu")
        # global_merged_human["rotation"][valid_mask] = canonical_human_rotation.to("cpu")
        # global_merged_human["opacity"][valid_mask] = features_human["opacity"][matches].to("cpu")
        # global_merged_human["features_dc"][valid_mask] = features_human["features_dc"][matches].to("cpu")
        # all_valid_masks[valid_mask] = 1

        #######
        raw_norm = canonical_human_rotation.norm(dim=-1, keepdim=True)  # (..., 1)
        normed_rots = torch.nn.functional.normalize(
            canonical_human_rotation, dim=-1
        )  # (..., 4)
        gs_rotmat = quaternion_to_matrix(normed_rots)  # (..., 3, 3)
        deformed_gs_rotmat = lbs_T[valid_mask, :3, :3] @ gs_rotmat  # (..., 3, 3)
        deformed_gs_rotq = matrix_to_quaternion(deformed_gs_rotmat)  # (..., 4)
        deformed_gs_rotq = torch.nn.functional.normalize(deformed_gs_rotq, dim=-1)
        deformed_gs_rotq = deformed_gs_rotq * raw_norm
        deformed_gs_rotq_canon = deformed_gs_rotq.clone()
        render_scale = canonical_human_scales + smpl_scale.log()
        new_features = {
            "xyz": deformed_smpl_at_canonical[0][valid_mask] * smpl_scale + smpl_transl,
            "features_dc": features_human["features_dc"][matches],
            "features_rest": features_human["features_rest"][matches],
            "opacity": features_human["opacity"][matches],
            "scaling": render_scale,
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
        if step % 100 == 0:
            # continue
            torchvision.utils.save_image(
                image, f"./_irem/{folder_name}/{image_idx}_canonical_human_black.png"
            )

        set_gaussian_model_features(gaussians, new_features)
        render_pkg = render_gaussians(gaussians, camera_list, bg="white")
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )
        if step % 100 == 0:
            # continue
            torchvision.utils.save_image(
                image, f"./_irem/{folder_name}/{image_idx}_canonical_human_white.png"
            )
        #######
        #######
        new_features = {
            "xyz": features_human["xyz"][matches],
            "features_dc": features_human["features_dc"][matches],
            "features_rest": features_human["features_rest"][matches],
            "opacity": features_human["opacity"][matches],
            "scaling": features_human["scaling"][matches],
            "rotation": features_human["rotation"][matches],
        }
        set_gaussian_model_features(gaussians, new_features)
        render_pkg = render_gaussians(gaussians, camera_list)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )
        if step % 100 == 0:
            # continue
            torchvision.utils.save_image(
                image, f"./_irem/{folder_name}/{image_idx}_matched_gaussians.png"
            )
        #######

        """
            aslında olan alttaki line
        """
        deformed_xyz, human_smpl_scales, deformed_gs_rotq, _ = (
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

        new_features = {
            "xyz": deformed_xyz[0][valid_mask],
            "features_dc": features_human["features_dc"][matches],
            "features_rest": features_human["features_rest"][matches],
            "opacity": features_human["opacity"][matches],
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
        if step % 100 == 0:
            # continue
            torchvision.utils.save_image(
                image, f"./_irem/{folder_name}/{image_idx}_canon2pose_human.png"
            )
        #######

        #######
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
        set_gaussian_model_features(gaussians, merged_features)
        render_pkg = render_gaussians(gaussians, camera_list)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )
        if step % 100 == 0:
            # continue
            torchvision.utils.save_image(
                image,
                f"./_irem/{folder_name}/{image_idx}_canon2pose_human_with_scene.png",
            )
        #######

        rr.set_time_seconds("frame", 0)
        rr.log(
            f"world/xyz",
            rr.Arrows3D(
                vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            ),
        )
        rr.log(
            f"world/scene",
            rr.Points3D(
                positions=features_scene["xyz"].detach().cpu().numpy(),
                colors=features_scene["features_dc"].squeeze(1).detach().cpu().numpy(),
            ),
        )
        rr.log(
            f"world/original_human_matched",
            rr.Points3D(
                positions=features_human["xyz"][matches].detach().cpu().numpy()
            ),
        )
        rr.log(
            f"world/original_human",
            rr.Points3D(positions=features_human["xyz"].detach().cpu().numpy()),
        )
        rr.log(
            f"world/deformed_human",
            rr.Points3D(
                positions=deformed_smpl_at_canonical.squeeze(0).detach().cpu().numpy()
            ),
        )
        rr.log(
            f"world/posed_smpl",
            rr.Points3D(positions=posed_smpl_at_canonical.detach().cpu().numpy()),
        )
        rr.log(
            f"world/posed_smpl",
            rr.Points3D(
                positions=((posed_smpl_at_canonical * smpl_scale) + smpl_transl)
                .detach()
                .cpu()
                .numpy()
            ),
        )

        rr.log(
            f"world/posed_from_canonical_human",
            rr.Points3D(positions=deformed_xyz[0].detach().cpu().numpy()),
        )

        """
            global_merged_human["xyz"] = deformed_smpl_at_canonical[0].to("cpu")
            global_merged_human["scaling"][valid_mask] = canonical_human_scales.to("cpu")
            global_merged_human["rotation"][valid_mask] = canonical_human_rotation.to("cpu")
            global_merged_human["opacity"][valid_mask] = features_human["opacity"][matches].to("cpu")
            global_merged_human["features_dc"][valid_mask] = features_human["features_dc"][matches].to("cpu")
            all_valid_masks[valid_mask] = 1
        """

        canonical_human_features = {
            "i": i,
            "image_idx": image_idx,
            "smpl_scale": smpl_scale,
            "smpl_transl": smpl_transl,
            "smpl_global_orient": smpl_global_orient,
            "camera": camera,
            "xyz": deformed_smpl_at_canonical[0].to("cuda"),
            "features_dc": features_human["features_dc"][matches].to("cuda"),
            "features_rest": features_human["features_rest"][matches].to("cuda"),
            "opacity": features_human["opacity"][matches].to("cuda"),
            "gs_rotq": deformed_gs_rotq_canon.to("cuda"),
            "scaling": canonical_human_scales.to("cuda"),
            "rotation": canonical_human_rotation.to("cuda"),
            "visibility_mask": valid_mask,
        }

        per_frame_canonical_human.append(canonical_human_features)

        print(
            f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! SAVED AFTER: {image_idx} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        )
        with open("per_frame_canonical_human.pkl", "wb") as f:
            pickle.dump(per_frame_canonical_human, f)

        continue

    break

    # rr.set_time_seconds("frame", 0)
    # rr.log(f"world/scene", rr.Points3D(positions=features_scene["xyz"].detach().cpu().numpy()))
    # rr.log(f"world/original_human", rr.Points3D(positions=features_human["xyz"].detach().cpu().numpy()))
    # rr.log(f"world/deformed_human", rr.Points3D(positions=deformed_xyz.detach().cpu().numpy()))

    # print("")

    # break

    # ### mask
    # mask = data["mask"]
    # bbox = data["bbox"]
    # x1, y1, x2, y2 = data["bbox"].int().tolist()

    # gt_crop = data["rgb"][:, x1:x2, y1:y2]
    # pred_crop = image[:, x1:x2, y1:y2]

    # l1_human = torch.abs(pred_crop - gt_crop).mean()
    # # print("l1_human", l1_human.item())

    # # loss = torch.nn.functional.mse_loss(image, data[0]["rgb"])
    # l1_loss = torch.abs((image - data["rgb"])).mean()
    # ssim_loss = (1 - ssim(image, data["rgb"]))
    # lpips_loss = lpips(image.clip(max=1), data["rgb"]).mean()
    # l_full = lambda_1 * l1_loss + lambda_2 * ssim_loss + lambda_3 * lpips_loss

    # # h, w = gt_crop.shape[-2:]
    # # adaptive_window = min(h, w, 11)
    # ssim_human =  (1 - ssim(pred_crop, gt_crop))
    # lpips_human = lpips(pred_crop.clip(max=1), gt_crop).mean()
    # loss_human = lambda_1 * l1_human + lambda_2 * ssim_human + lambda_3 * lpips_human

    # loss = l_full + loss_human

    # # print(loss)
    # if step % 100 == 1:
    #     l1_losses.append(l1_loss.item())
    #     ssim_losses.append(ssim_loss.item())
    #     lpips_losses.append(lpips_loss.item())
    #     steps.append(step)

    #     l1_human_losses.append(l1_human.item())
    #     ssim_human_losses.append(ssim_human.item())
    #     lpips_human_losses.append(lpips_human.item())

    # loss.backward()

    # total_norm = 0.0
    # for p in model.parameters():
    #     if p.grad is not None:
    #         param_norm = p.grad.data.norm(2)
    #         total_norm += param_norm.item() ** 2
    # total_norm = total_norm ** 0.5

    # # torch.nn.utils.clip_grad_norm_(model.gs_head_feats.parameters(), max_norm=0.5)

    # tqdm.write(f"Step {step}, Cam {i}, L1: {loss.item():.6f}, SSIM: {ssim_loss.item():.6f}, , LPIPS: {lpips_loss.item():.6f}, L1-Human: {l1_human.item():.6f}, SSIM-Human: {ssim_human.item():.6f}, LPIPS-Human: {lpips_human.item():.6f}, Grad-Norm: {total_norm:.3f}")

    # optimizer.step()
    # optimizer.zero_grad()


# # plt.figure()
# # plt.plot(steps, l1_losses, label='L1 Loss')
# # plt.plot(steps, ssim_losses, label='SSIM Loss')
# # plt.plot(steps, lpips_losses, label='LPIPS Loss')
# # plt.plot(steps, l1_human_losses, label='L1 Human Loss')
# # plt.plot(steps, ssim_human_losses, label='SSIM Human Loss')
# # plt.plot(steps, lpips_human_losses, label='LPIPS Human Loss')
# # plt.xlabel('Step')
# # plt.ylabel('Loss')
# # plt.title('Losses over Time')
# # plt.legend()
# # plt.grid(True)
# # plt.savefig(f'./_irem/{folder_name}/losses_plot.png')
# # plt.close()

# # # SAVE THE MODEL
# # torch.save(model.gs_head_feats.state_dict(), f"./_irem/{folder_name}/gs_head_feats.pth")
