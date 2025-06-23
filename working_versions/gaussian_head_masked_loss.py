import sys

sys.path.append(".")

import torch
from vggt.models.our_vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from hugs.datasets import NeumanDataset
import math
from instantsplat.scene import GaussianModel
from instantsplat.scene.cameras import Camera
from instantsplat.gaussian_renderer import render
from tqdm import tqdm
from hugs.trainer.gs_trainer import get_train_dataset
from omegaconf import OmegaConf
from hugs.cfg.config import cfg as default_cfg
from hugs.renderer.gs_renderer import render_human_scene
import torchvision
import torch.nn.functional as F
from instantsplat.arguments import PipelineParams, ArgumentParser
import numpy as np

import rerun as rr

rr.init("debugging", recording_id="v0.1")
rr.connect_tcp("0.0.0.0:9876")
rr.log(
    f"world/xyz",
    rr.Arrows3D(
        vectors=[[1, 0, 0], [0, 2, 0], [0, 0, 3]],
        colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
    ),
)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

for param in model.parameters():
    param.requires_grad = False
# for param in model.gs_head_xyz.parameters():
#     param.requires_grad = True
# for param in model.gs_head_feats.parameters():
#     param.requires_grad = True


for param in model.gs_head_opacity.parameters():
    param.requires_grad = True
for param in model.gs_head_rotation.parameters():
    param.requires_grad = True
for param in model.gs_head_scale.parameters():
    param.requires_grad = True
for param in model.gs_head_sh.parameters():
    param.requires_grad = True
for param in model.gs_head_xyz_offset.parameters():
    param.requires_grad = True

for param in model.aggregator.parameters():
    param.requires_grad = False

model.gs_head_rotation.train()
model.gs_head_opacity.train()
model.gs_head_sh.train()
model.gs_head_scale.train()
# model.gs_head_xyz_offset.train()
# model.aggregator.train()


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

    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    render_pkg = render(viewpoint_cam, gaussians, pipe, bg, camera_pose=pose)

    return render_pkg


image_names = [
    f"/local/home/idemir/Desktop/3dv_dynamichumansplatting/data/neuman/dataset/lab/images/{i:05}.png"
    for i in range(1)
]
images = load_and_preprocess_images(image_names).to(device)
with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        images = images[None]  # add batch dimension
        aggregated_tokens_list, ps_idx = model.aggregator(images)

data = [get_data(i) for i in range(1)]

# rr.set_time_seconds("frame", 0)
# rr.log("world/neuman_camera", rr.Pinhole(
#     image_from_camera=data[0]["cam_intrinsics"].cpu().numpy(),
#     resolution=[data[0]["image_width"], data[0]["image_height"]]  # width x height
# ))

# # Log the camera pose (as translation + rotation)
# rr.log("world/neuman_camera", rr.Transform3D(
#     translation=data[0]["camera_center"].cpu().numpy(),
# ))

# rr.log("world/neuman_camera/image", rr.Image(data[0]["rgb"].permute(1, 2, 0).cpu().numpy()))


point_map, point_conf = model.point_head(aggregated_tokens_list, images, ps_idx)
# rr.set_time_seconds("frame", 0)
# rr.log(f"world/irem_inittt_{0}", rr.Points3D(positions=point_map[0, 0, :, :, 0:3].reshape(-1, 3).detach().cpu().numpy()))


pose_enc = model.camera_head(aggregated_tokens_list)[-1]
extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
camera_to_world = np.eye(4, dtype=np.float32)
camera_to_world[:3, :] = extrinsic[0, 0, :, :].cpu().numpy()  # Fill the top 3 rows

# rr.set_time_seconds("frame", 0)
# rr.log("world/camera", rr.Pinhole(
#     image_from_camera=intrinsic[0][0].cpu().numpy(),
#     resolution=[data[0]["image_width"], data[0]["image_height"]]  # width x height
# ))
# rr.log("world/camera", rr.Transform3D(
#     translation=extrinsic[0, 0, :, 3:].cpu().numpy(),
#     rotation=extrinsic[0, 0, :, :3].cpu().numpy()
# ))
# rr.log("world/camera/image", rr.Image(data[0]["rgb"].permute(1, 2, 0).cpu().numpy()))

# for i in range(1):
#     extr_4x4 = torch.eye(4, device='cuda:0')
#     extr_4x4[:3, :] = extrinsic[0][i]  # Fill rotation and translation
#     c2w = torch.inverse(extr_4x4)

#     data[i]["world_view_transform"] = extr_4x4
#     data[i]["c2w"] = c2w
#     data[i]["cam_intrinsics"] = intrinsic[0][i]
#     data[i]["camera_center"] = c2w[:3, 3]
#     data[i]["transl"] = torch.tensor([0, 0, 0]).cuda()

camera_list = []
for i in range(1):
    znear = data[i]["near"]
    zfar = data[i]["far"]

    # Step 1: Construct world_to_camera matrix (4x4)
    extr_4x4 = torch.eye(4, device="cuda:0")
    extr_4x4[:3, :] = extrinsic[0][i]  # [3x4] -> [4x4]
    world_to_camera = extr_4x4
    c2w = torch.inverse(world_to_camera)

    # Step 2: Extract intrinsics
    K = intrinsic[0][i]
    width = data[i]["image_width"]
    height = data[i]["image_height"]

    # Step 3: Compute FoV from intrinsics
    fovx = 2 * torch.atan(width / (2 * K[0, 0]))
    fovy = 2 * torch.atan(height / (2 * K[1, 1]))

    def get_projection_matrix(znear, zfar, fovX, fovY):
        tanHalfFovY = math.tan((fovY / 2))
        tanHalfFovX = math.tan((fovX / 2))

        top = tanHalfFovY * znear
        bottom = -top
        right = tanHalfFovX * znear
        left = -right

        P = torch.zeros(4, 4)

        z_sign = 1.0

        P[0, 0] = 2.0 * znear / (right - left)
        P[1, 1] = 2.0 * znear / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[3, 2] = z_sign
        P[2, 2] = z_sign * zfar / (zfar - znear)
        P[2, 3] = -(zfar * znear) / (zfar - znear)
        return torch.tensor(P).cuda()

    projection_matrix = get_projection_matrix(znear, zfar, fovx, fovy).transpose(0, 1)

    # Step 5: Compute full projection transform
    full_proj_transform = (
        world_to_camera.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
    ).squeeze(0)

    # Step 6: Camera center (in world space)
    camera_center = c2w[:3, 3]

    # Update dictionary
    data[i].update(
        {
            "fovx": fovx,
            "fovy": fovy,
            "image_height": height,
            "image_width": width,
            "world_view_transform": world_to_camera,
            "c2w": c2w,
            "full_proj_transform": full_proj_transform,
            "camera_center": camera_center,
            "cam_intrinsics": K.float(),
            "near": znear,
            "far": zfar,
        }
    )

    # Extract R and T from extrinsics
    R = extrinsic[0][i][:, :3].cpu().numpy()
    T = extrinsic[0][i][:, 3].cpu().numpy()

    # Create Camera instance
    cam = Camera(
        colmap_id=i,
        R=R,
        T=T,
        FoVx=fovx,
        FoVy=fovy,
        image=data[i]["rgb"],  # assuming shape [3, H, W] and values in [0, 1]
        gt_alpha_mask=data[i].get("alpha_mask", None),
        image_name=data[i].get("image_name", f"img_{i}.png"),
        uid=i,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
        data_device="cuda:0",
    )

    camera_list.append(cam)


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


cameras = {1.0: camera_list}
print("CAMERAS init")

# # ####################################################################################################

model_args = torch.load("gaussian_object.pth")
gaussians_from_other = GaussianModel(0)
gaussians_from_other.restore(model_args, None)

# # ################################################################################################### pre-training
# optimizer_xyz = torch.optim.AdamW(model.gs_head_xyz.parameters(), lr=2e-4)
# for step in tqdm(range(1000)):
#     gs_map, gs_map_conf = model.gs_head_xyz(aggregated_tokens_list, images, ps_idx)

#     pred_xyz = gs_map[0, 0, :, :, 0:3].reshape(-1, 3)
#     gt_xyz = gaussians_from_other._xyz.data
#     loss = F.mse_loss(pred_xyz, gt_xyz)


#     if step % 50 == 0:
#         rr.set_time_seconds("frame", step)
#         rr.log(f"world/pred_pc{0}", rr.Points3D(positions=gs_map[0, 0, :, :, 0:3].reshape(-1, 3).detach().cpu().numpy()))

#     # Compute loss (you should define ground_truth appropriately)
#     # loss = get_3d_loc_loss(gs_map, point_map)

#     print(loss)
#     loss.backward()
#     optimizer_xyz.step()
#     optimizer_xyz.zero_grad()

# torch.save(model.gs_head_xyz.state_dict(), "gs_xyz_head.pth")


# model.gs_head_xyz.load_state_dict(torch.load("gs_xyz_head.pth"))

# #################################################################################################### pre-training
# optimizer_feats = torch.optim.AdamW(model.gs_head_feats.parameters(), lr=2e-4)

# final_conv_layer = model.gs_head_feats.scratch.output_conv2[-1]
# splits_and_inits = [
#     (3, 0.00003, -7.0),  # Scales
#     (4, 1.0, 0.0),  # Rotations
#     (3 * 1, 1.0, 0.0),  # Spherical Harmonics
#     (1, 1.0, -2.0),  # Opacity
# ]

# start_channels = 0
# for out_channel, std, bias_val in splits_and_inits:
#     end_channels = start_channels + out_channel

#     # Xavier init with gain based on std (even though std is not gain in Xavier, it approximates intent)
#     torch.nn.init.xavier_uniform_(
#         final_conv_layer.weight[start_channels:end_channels],
#         gain=std
#     )
#     torch.nn.init.constant_(
#         final_conv_layer.bias[start_channels:end_channels],
#         bias_val
#     )

#     start_channels = end_channels

# from splatt3r.src.mast3r_src.mast3r.catmlp_dpt_head import VGGT_GaussianHead
# head = VGGT_GaussianHead(None).to("cuda")
# head.train()

# gaussians = GaussianModel(0)
# gaussians.init_RT_seq(cameras)

# optimizer_feats = torch.optim.AdamW(
#     list(head.parameters()),
#     lr=2e-4
# )

# prev_loss = 0
# for step in tqdm(range(10000)):
#     # gs_map, gs_map_conf = model.gs_head_feats(aggregated_tokens_list, images, ps_idx)
#     gs_map = head([t.squeeze(0)[:, 5:, :] for t in aggregated_tokens_list], (294, 518))

#     fmap = gs_map.permute(0, 2, 3, 1)  # B,H,W,D

#     pts3d, conf, desc, desc_conf, offset, scales, rotations, sh, opacities = torch.split(fmap, [3, 1, 16, 1, 3, 3, 4, 3 * 1, 1], dim=-1)

#     print("a")
#     pred_features = {
#         "features_dc": sh.view(-1, 1, 3),
#         "features_rest": torch.zeros((sh.view(-1, 3, 1).shape[0], 15, 3), device=gs_map.device),
#         "opacity": opacities.view(-1, 1),
#         "scaling": scales.view(-1, 3),
#         "rotation": rotations.view(-1, 4),
#     }
#     pred_features["xyz"] = pts3d.view(-1, 3)

#     gt_features = {
#         "features_dc": gaussians_from_other._features_dc.data,
#         "features_rest": gaussians_from_other._features_rest.data,
#         "opacity": gaussians_from_other._opacity.data,
#         "scaling": gaussians_from_other._scaling.data,
#         "rotation": gaussians_from_other._rotation.data,
#     }

#     set_gaussian_model_features(gaussians, pred_features)

#     camera_list = []
#     camera = torch.load("viewpoint_cam.pth")
#     camera_list.append(camera)

#     render_pkg = render_gaussians(gaussians, camera_list)
#     image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
#     if step % 1 == 0:
#         torchvision.utils.save_image(image, f"./_irem/pred_{step}.png")


#     loss = (
#         F.mse_loss(pred_features["features_dc"], gt_features["features_dc"]) +
#         F.mse_loss(pred_features["features_rest"], gt_features["features_rest"]) +
#         F.mse_loss(pred_features["opacity"], gt_features["opacity"]) +
#         F.mse_loss(pred_features["scaling"], gt_features["scaling"]) +
#         F.mse_loss(pred_features["rotation"], gt_features["rotation"])
#     )

#     # if step != 0 and (loss - prev_loss > 0.8):
#     #     print("irem")
#     #     print("aaa")

#     prev_loss = loss

#     print(loss)
#     loss.backward()
#     optimizer_feats.step()
#     optimizer_feats.zero_grad()

# torch.save(model.gs_head_feats.state_dict(), "gs_feat_head.pth")
###################################################################################################


gaussians = GaussianModel(0)
gaussians.init_RT_seq(cameras)


print("GAUSSIANS init")

final_convs = [
    model.gs_head_xyz_offset.scratch.output_conv2[-1],
    model.gs_head_scale.scratch.output_conv2[-1],
    model.gs_head_rotation.scratch.output_conv2[-1],
    model.gs_head_sh.scratch.output_conv2[-1],
    model.gs_head_opacity.scratch.output_conv2[-1],
]

splits_and_inits = [
    (3, 0.00001, 0.0),  # 3D mean offsets
    (3, 0.00003, -7.0),  # Scales
    (4, 1.0, 0.0),  # Rotations
    (3, 1.0, 0.0),  # Spherical Harmonics
    (1, 1.0, 0.0),  # Opacity
]

for (out_channel, std, bias_val), final_conv_layer in zip(
    splits_and_inits, final_convs
):
    end_channels = out_channel

    # Xavier init with gain based on std (even though std is not gain in Xavier, it approximates intent)
    torch.nn.init.xavier_uniform_(final_conv_layer.weight[0:end_channels], gain=std)
    torch.nn.init.constant_(final_conv_layer.bias[0:end_channels], bias_val)

print("FINAL LAYER INITS")

# optimizer = torch.optim.AdamW(
#     # list(model.aggregator.parameters()) +
#     # list(model.gs_head_xyz_offset.parameters()) +
#     list(model.gs_head_rotation.parameters()) +
#     list(model.gs_head_opacity.parameters()) +
#     list(model.gs_head_scale.parameters()) +
#     list(model.gs_head_sh.parameters()),
#     lr=2e-4
# )

optimizer = torch.optim.AdamW(
    [
        {"params": model.gs_head_rotation.parameters(), "lr": 0.0001},
        {"params": model.gs_head_opacity.parameters(), "lr": 0.0005},
        {"params": model.gs_head_scale.parameters(), "lr": 0.00005},
        {"params": model.gs_head_sh.parameters(), "lr": 0.00025},
    ],
    lr=2e-4,
    betas=(0.9, 0.999),
    eps=1e-15,
    weight_decay=0.0,
)


print("optimizer init")

from hugs.losses.utils import ssim
from lpips import LPIPS

lpips = LPIPS(net="vgg", pretrained=True).to("cuda")
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
exp_name = f"MASKED_LOSS_2000_{lambda_1}_{lambda_2}_{lambda_3}"
import os

os.makedirs(f"./_irem/{exp_name}", exist_ok=True)
print("STARTING TRAINING")
for step in range(2000):
    # Forward pass through gs_head
    offset, _ = model.gs_head_xyz_offset(aggregated_tokens_list, images, ps_idx)
    op, _ = model.gs_head_opacity(aggregated_tokens_list, images, ps_idx)
    rot, _ = model.gs_head_rotation(aggregated_tokens_list, images, ps_idx)
    scale, _ = model.gs_head_scale(aggregated_tokens_list, images, ps_idx)
    sh, _ = model.gs_head_sh(aggregated_tokens_list, images, ps_idx)

    gs_map_xyz = gaussians_from_other._xyz.data

    model_args = torch.load("gaussian_object.pth")
    gaussians_from_other = GaussianModel(0)
    gaussians_from_other.restore(model_args, None)
    # camera = torch.load("camera_0.pth")
    camera = torch.load("viewpoint_cam.pth")

    camera_list[0] = camera

    features = {
        "xyz": gs_map_xyz,
        "features_dc": sh[0, i, :, :, 0:3].view(-1, 1, 3),
        "features_rest": torch.zeros(
            (sh[0, i, :, :, 0:3].detach().view(-1, 3, 1).shape[0], 15, 3),
            device=gs_map_xyz.device,
        ),
        "opacity": op[0, i, :, :, 0].view(-1, 1),
        "scaling": scale[0, i, :, :, 0:3].view(-1, 3),
        "rotation": rot[0, i, :, :, 0:4].view(-1, 4),
    }
    set_gaussian_model_features(gaussians, features)

    # render_pkg = render_gaussians(gaussians_from_other, camera_list)
    render_pkg = render_gaussians(gaussians, camera_list)
    image, viewspace_point_tensor, visibility_filter, radii = (
        render_pkg["render"],
        render_pkg["viewspace_points"],
        render_pkg["visibility_filter"],
        render_pkg["radii"],
    )
    if step % 50 == -1:
        torchvision.utils.save_image(
            image, f"./_irem/{exp_name}/frozen_rendered_pred_{step}_{exp_name}.png"
        )

    ### mask
    mask = data[0]["mask"]
    bbox = data[0]["bbox"]
    x1, y1, x2, y2 = data[0]["bbox"].int().tolist()

    gt_crop = data[0]["rgb"][:, x1:x2, y1:y2]
    pred_crop = image[:, x1:x2, y1:y2]

    l1_human = torch.abs(pred_crop - gt_crop).mean()
    # print("l1_human", l1_human.item())

    # loss = torch.nn.functional.mse_loss(image, data[0]["rgb"])
    l1_loss = torch.abs((image - data[0]["rgb"])).mean()
    ssim_loss = 1 - ssim(image, data[0]["rgb"])
    lpips_loss = lpips(image.clip(max=1), data[0]["rgb"]).mean()
    l_full = lambda_1 * l1_loss + lambda_2 * ssim_loss + lambda_3 * lpips_loss

    # h, w = gt_crop.shape[-2:]
    # adaptive_window = min(h, w, 11)
    ssim_human = 1 - ssim(pred_crop, gt_crop)
    lpips_human = lpips(pred_crop.clip(max=1), gt_crop).mean()
    loss_human = lambda_1 * l1_human + lambda_2 * ssim_human + lambda_3 * lpips_human

    loss = l_full + loss_human

    # print(loss)
    tqdm.write(
        f"Step {step}, L1: {loss.item():.6f}, SSIM-Loss: {ssim_loss.item():.6f}, , LPIPS-Loss: {lpips_loss.item():.6f}, L1-Human: {l1_human.item():.6f}, SSIM-Human: {ssim_human.item():.6f}, LPIPS-Human: {lpips_human.item():.6f}"
    )

    l1_losses.append(l1_loss.item())
    ssim_losses.append(ssim_loss.item())
    lpips_losses.append(lpips_loss.item())
    steps.append(step)

    l1_human_losses.append(l1_human.item())
    ssim_human_losses.append(ssim_human.item())
    lpips_human_losses.append(lpips_human.item())

    loss.backward()

    optimizer.step()
    optimizer.zero_grad()


plt.figure()
plt.plot(steps, l1_losses, label="L1 Loss")
plt.plot(steps, ssim_losses, label="SSIM Loss")
plt.plot(steps, lpips_losses, label="LPIPS Loss")
plt.plot(steps, l1_human_losses, label="L1 Human Loss")
plt.plot(steps, ssim_human_losses, label="SSIM Human Loss")
plt.plot(steps, lpips_human_losses, label="LPIPS Human Loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Losses over Time")
plt.legend()
plt.grid(True)
plt.savefig(f"./_irem/{exp_name}/losses_plot.png")
plt.close()


# for step in range(1000):
#     # Forward pass through gs_head
#     gs_map_feats, _ = model.gs_head_feats(aggregated_tokens_list, images, ps_idx)
#     gs_map_xyz, _   = model.gs_head_xyz(aggregated_tokens_list, images, ps_idx)

#     model_args = torch.load("gaussian_object.pth")
#     gaussians_from_other = GaussianModel(0)
#     gaussians_from_other.restore(model_args, None)
#     # camera = torch.load("camera_0.pth")
#     camera = torch.load("viewpoint_cam.pth")

#     camera_list[0] = camera

#     features = {
#         "xyz":              gs_map_xyz[0, i, :, :, 0:3].view(-1, 3),
#         "features_dc":      gs_map_feats[0, i, :, :, 7:10].view(-1, 1, 3),
#         "features_rest": torch.zeros((gs_map_feats[0, i, :, :, 0:3].view(-1, 3, 1).shape[0], 15, 3), device=gs_map_feats.device),
#         "opacity":          gs_map_feats[0, i, :, :, 10:11].view(-1, 1),
#         "scaling":          gs_map_feats[0, i, :, :, 0:3].view(-1, 3),
#         "rotation":         gs_map_feats[0, i, :, :, 3:7].view(-1, 4),
#     }
#     set_gaussian_model_features(gaussians, features)


#     if step % 50 == 0:
#         rr.set_time_seconds("frame", step)
#         rr.log(f"world/pred_pc{0}", rr.Points3D(positions=gs_map_xyz[0, 0, :, :, 0:3].reshape(-1, 3).detach().cpu().numpy()))

#     # render_pkg = render_gaussians(gaussians_from_other, camera_list)
#     render_pkg = render_gaussians(gaussians, camera_list)
#     image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
#     if step % 1 == 0:
#         torchvision.utils.save_image(image, f"./_irem/rendered_pred_{step}.png")

#     # loss = torch.abs((image - data[0]["rgb"])).mean()
#     loss = 0.8 * torch.nn.functional.mse_loss(image, data[0]["rgb"])
#     loss += 0.2 * (1 - ssim(image, data[0]["rgb"]))

#     print(loss)

#     loss.backward()

#     optimizer.step()

#     optimizer.zero_grad()
