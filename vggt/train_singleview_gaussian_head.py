import sys
sys.path.append('.')

import torch
from vggt.models.our_vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from hugs.datasets import NeumanDataset
import math
from instantsplat.scene import GaussianModel
from instantsplat.scene.gaussian_model_w_act import GaussianModel_With_Act
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
import random

import rerun as rr
rr.init("debugging", recording_id="v0.1")
rr.connect_tcp("0.0.0.0:9876")
rr.log(f"world/xyz", rr.Arrows3D(vectors=[[1, 0, 0], [0, 2, 0], [0, 0, 3]], colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]))

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

    model.max_radii2D = torch.zeros((features["xyz"].shape[0]), device=features["xyz"].device)

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

#point_map, point_conf = model.point_head(aggregated_tokens_list, images, ps_idx)

image_ids = [0, 1, 2, 3]
image_names = [f"/local/home/idemir/Desktop/3dv_dynamichumansplatting/data/neuman/dataset/lab/images/{i:05}.png" for i in image_ids]  
images_list, tokens_list, ps_idx_list, data_list, camera_list = [], [], [], [], []


for img_path, i in zip(image_names, image_ids):
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

# # ####################################################################################################

final_conv_layer = model.gs_head_feats.scratch.output_conv2[-1]

splits_and_inits = [
    (3, 0.00003, -7.0),  # Scales
    (4, 1.0, 0.0),  # Rotations
    (3, 1.0, 0.0),  # Spherical Harmonics
    (1, 1.0, 0.0),  # Opacity
    (3, 0.00001, 0.0),  # 3D mean offsets
]

start = 0
for out_channel, std, bias_val in splits_and_inits:
    end = start + out_channel

    # Xavier init for weight slice
    torch.nn.init.xavier_uniform_(
        final_conv_layer.weight[start:end],
        gain=std
    )
    
    # Constant init for bias slice
    torch.nn.init.constant_(
        final_conv_layer.bias[start:end],
        bias_val
    )
    
    start = end



print("FINAL LAYER INITS")

optimizer = torch.optim.AdamW(
    [
        {"params": model.gs_head_feats.parameters(), "lr": 0.00005}
    ],
    lr=2e-4,
    betas=(0.9, 0.999),
    eps=1e-15,
    weight_decay=0.0
)


print("optimizer init")

from hugs.losses.utils import ssim
from lpips import LPIPS
lpips = LPIPS(net="vgg", pretrained=True).to('cuda')
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
exp_name = f"10_MAY_SINGLE_HEAD_W_OFFSET_{len(images_list)}_IMAGE_{iterations}_{lambda_1}_{lambda_2}_{lambda_3}"

import os
from datetime import datetime
timestamp = datetime.now().strftime("%d-%m_%H-%M")
# folder_name = f"{timestamp}_{exp_name}"
folder_name = f"{exp_name}"
os.makedirs(f"./_irem/{folder_name}", exist_ok=True)
print("STARTING TRAINING")


for step in tqdm(range(iterations), desc="Training", total=iterations):

    # GET THE DATA POINT:
    i = random.randint(0, len(images_list) - 1)
    images = images_list[i]
    aggregated_tokens_list = tokens_list[i]
    ps_idx = ps_idx_list[i]
    data = data_list[i]
    # gaussians = gaussians_list[i]

    
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
    camera = torch.load(f"/local/home/idemir/Desktop/3dv_dynamichumansplatting/instantsplat/lab_{i:05}/viewpoint_cam_0.pth") 
    # camera = torch.load("viewpoint_cam.pth") 
    camera_list = [camera]

    gaussians = GaussianModel_With_Act(0)
    gaussians.init_RT_seq({1.0: camera_list})

    
    features = {
        "xyz":              xyz,
        "features_dc":      sh[0, 0, :, :, 0:3].view(-1, 1, 3),
        "features_rest": torch.zeros((sh[0, 0, :, :, 0:3].detach().view(-1, 3, 1).shape[0], 15, 3), device=xyz.device),
        "opacity":          op[0, 0, :, :, 0].view(-1, 1),
        "scaling":          scale[0, 0, :, :, 0:3].view(-1, 3),
        "rotation":         rot[0, 0, :, :, 0:4].view(-1, 4),
    }
    set_gaussian_model_features(gaussians, features)

    # render_pkg = render_gaussians(gaussians_from_other, camera_list)
    render_pkg = render_gaussians(gaussians, camera_list)
    image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
    if step % 100 == 0:
        #continue
        torchvision.utils.save_image(image, f"./_irem/{folder_name}/render_step_{step}_cam_{i}_exp_{exp_name}.png")

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
    ssim_loss = (1 - ssim(image, data["rgb"]))
    lpips_loss = lpips(image.clip(max=1), data["rgb"]).mean()
    l_full = lambda_1 * l1_loss + lambda_2 * ssim_loss + lambda_3 * lpips_loss
    
    # h, w = gt_crop.shape[-2:]
    # adaptive_window = min(h, w, 11)  
    ssim_human =  (1 - ssim(pred_crop, gt_crop))
    lpips_human = lpips(pred_crop.clip(max=1), gt_crop).mean()
    loss_human = lambda_1 * l1_human + lambda_2 * ssim_human + lambda_3 * lpips_human


    loss = l_full + loss_human

    # print(loss)
    if step % 100 == 1:
        l1_losses.append(l1_loss.item())
        ssim_losses.append(ssim_loss.item())
        lpips_losses.append(lpips_loss.item())
        steps.append(step)

        l1_human_losses.append(l1_human.item())
        ssim_human_losses.append(ssim_human.item())
        lpips_human_losses.append(lpips_human.item())

    loss.backward()


    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    # torch.nn.utils.clip_grad_norm_(model.gs_head_feats.parameters(), max_norm=0.5)



    tqdm.write(f"Step {step}, Cam {i}, L1: {loss.item():.6f}, SSIM: {ssim_loss.item():.6f}, , LPIPS: {lpips_loss.item():.6f}, L1-Human: {l1_human.item():.6f}, SSIM-Human: {ssim_human.item():.6f}, LPIPS-Human: {lpips_human.item():.6f}, Grad-Norm: {total_norm:.3f}")

    optimizer.step()
    optimizer.zero_grad()

    

plt.figure()
plt.plot(steps, l1_losses, label='L1 Loss')
plt.plot(steps, ssim_losses, label='SSIM Loss')
plt.plot(steps, lpips_losses, label='LPIPS Loss')
plt.plot(steps, l1_human_losses, label='L1 Human Loss')
plt.plot(steps, ssim_human_losses, label='SSIM Human Loss')
plt.plot(steps, lpips_human_losses, label='LPIPS Human Loss')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Losses over Time')
plt.legend()
plt.grid(True)
plt.savefig(f'./_irem/{folder_name}/losses_plot.png')
plt.close()

# SAVE THE MODEL
torch.save(model.gs_head_feats.state_dict(), f"./_irem/{folder_name}/gs_head_feats.pth")

