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


# point_map, point_conf = model.point_head(aggregated_tokens_list, images, ps_idx)

image_names = [
    f"/local/home/idemir/Desktop/3dv_dynamichumansplatting/data/neuman/dataset/lab/images/{i:05}.png"
    for i in range(0, 4)
]
images_list, tokens_list, ps_idx_list, data_list, camera_list = [], [], [], [], []


for i, img_path in enumerate(image_names):
    print(f"Processing image {i+1}/{len(image_names)}: {img_path}")

    image = load_and_preprocess_images([img_path]).to(device)
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            image = image[None]  # add batch dimension
            images_list.append(image)
            aggregated_tokens_list, ps_idx = model.aggregator(image)
            tokens_list.append(aggregated_tokens_list)
            ps_idx_list.append(ps_idx)
        data = dataset[i]
    data_list.append(data)

    pose_enc = model.camera_head(aggregated_tokens_list)[-1]
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, image.shape[-2:])
    camera_to_world = np.eye(4, dtype=np.float32)
    camera_to_world[:3, :] = extrinsic[0, 0, :, :].cpu().numpy()

    znear = data["near"]
    zfar = data["far"]

    # Step 1: Construct world_to_camera matrix (4x4)
    extr_4x4 = torch.eye(4, device="cuda:0")
    extr_4x4[:3, :] = extrinsic[0][0]  # [3x4] -> [4x4]
    world_to_camera = extr_4x4
    c2w = torch.inverse(world_to_camera)

    # Step 2: Extract intrinsics
    K = intrinsic[0][0]
    width = data["image_width"]
    height = data["image_height"]

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
    data.update(
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
    R = extrinsic[0][0][:, :3].cpu().numpy()
    T = extrinsic[0][0][:, 3].cpu().numpy()

    # Create Camera instance
    cam = Camera(
        colmap_id=i,
        R=R,
        T=T,
        FoVx=fovx,
        FoVy=fovy,
        image=data["rgb"],  # assuming shape [3, H, W] and values in [0, 1]
        gt_alpha_mask=data.get("alpha_mask", None),
        image_name=data.get("image_name", f"img_{i}.png"),
        uid=i,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
        data_device="cuda:0",
    )

    camera_list.append(cam)

camera0 = {1.0: [camera_list[0]]}
camera1 = {1.0: [camera_list[1]]}
camera2 = {1.0: [camera_list[2]]}
camera3 = {1.0: [camera_list[3]]}

print("CAMERAS init")

# # ####################################################################################################

gaussian0 = GaussianModel(0)
gaussian0.init_RT_seq(camera0)

gaussian1 = GaussianModel(0)
gaussian1.init_RT_seq(camera1)

gaussian2 = GaussianModel(0)
gaussian2.init_RT_seq(camera2)

gaussian3 = GaussianModel(0)
gaussian3.init_RT_seq(camera3)

gaussians_list = [gaussian0, gaussian1, gaussian2, gaussian3]


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
iterations = 10000
exp_name = f"4_IMAGES_{iterations}_{lambda_1}_{lambda_2}_{lambda_3}"
import os

os.makedirs(f"./_irem/{exp_name}", exist_ok=True)
print("STARTING TRAINING")
import random

for step in tqdm(range(iterations), desc="Training", total=iterations):
    # GET THE DATA POINT:
    i = random.randint(0, len(images_list) - 1)
    images = images_list[i]
    aggregated_tokens_list = tokens_list[i]
    ps_idx = ps_idx_list[i]
    data = data_list[i]
    gaussians = gaussians_list[i]

    # Forward pass through gs_head
    offset, _ = model.gs_head_xyz_offset(aggregated_tokens_list, images, ps_idx)
    op, _ = model.gs_head_opacity(aggregated_tokens_list, images, ps_idx)
    rot, _ = model.gs_head_rotation(aggregated_tokens_list, images, ps_idx)
    scale, _ = model.gs_head_scale(aggregated_tokens_list, images, ps_idx)
    sh, _ = model.gs_head_sh(aggregated_tokens_list, images, ps_idx)

    point_map, _ = model.point_head(aggregated_tokens_list, images, ps_idx)
    xyz = point_map[0, 0].reshape(-1, 3)

    # camera = torch.load("camera_0.pth")
    # camera = torch.load("viewpoint_cam.pth")
    camera = torch.load(
        f"/local/home/idemir/Desktop/3dv_dynamichumansplatting/instantsplat/lab_{i}/viewpoint_cam.pth"
    )
    # camera = torch.load("viewpoint_cam.pth")
    camera_list = [camera]

    features = {
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
    set_gaussian_model_features(gaussians, features)

    # render_pkg = render_gaussians(gaussians_from_other, camera_list)
    render_pkg = render_gaussians(gaussians, camera_list)
    image, viewspace_point_tensor, visibility_filter, radii = (
        render_pkg["render"],
        render_pkg["viewspace_points"],
        render_pkg["visibility_filter"],
        render_pkg["radii"],
    )
    if step % 1 == 0:
        # continue
        torchvision.utils.save_image(
            image,
            f"./_irem/{exp_name}/frozen_rendered_pred_{step}_{exp_name}_cam_{i}.png",
        )

    ### mask
    mask = data["mask"]
    bbox = data["bbox"]
    x1, y1, x2, y2 = data["bbox"].int().tolist()

    gt_crop = data["rgb"][:, x1:x2, y1:y2]
    pred_crop = image[:, x1:x2, y1:y2]

    l1_human = torch.abs(pred_crop - gt_crop).mean()
    # print("l1_human", l1_human.item())

    # loss = torch.nn.functional.mse_loss(image, data[0]["rgb"])
    l1_loss = torch.abs((image - data["rgb"])).mean()
    ssim_loss = 1 - ssim(image, data["rgb"])
    lpips_loss = lpips(image.clip(max=1), data["rgb"]).mean()
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
    if step % 100 == 1:
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

# SAVE THE MODEL
# torch.save(model.gs_head_xyz_offset.state_dict(), f"./_irem/{exp_name}/gs_xyz_offset_head.pth")
torch.save(
    model.gs_head_rotation.state_dict(), f"./_irem/{exp_name}/gs_rotation_head.pth"
)
torch.save(
    model.gs_head_opacity.state_dict(), f"./_irem/{exp_name}/gs_opacity_head.pth"
)
torch.save(model.gs_head_scale.state_dict(), f"./_irem/{exp_name}/gs_scale_head.pth")
torch.save(model.gs_head_sh.state_dict(), f"./_irem/{exp_name}/gs_sh_head.pth")
