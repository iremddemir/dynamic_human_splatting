import sys
sys.path.append('.')

import os
import math
import random
from datetime import datetime
from pathlib import Path
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms as TF
from torch.utils.tensorboard import SummaryWriter

from vggt.models.our_vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from hugs.datasets import NeumanDataset
from hugs.trainer.gs_trainer import get_train_dataset_ordered
from hugs.trainer.gs_trainer import get_train_dataset
from hugs.cfg.config import cfg as default_cfg
from hugs.renderer.gs_renderer import render_human_scene
from hugs.losses.utils import ssim
from hugs.utils.rotations import matrix_to_axis_angle
from vggt.utils.geometry import closed_form_inverse_se3

from instantsplat.scene import GaussianModel
from instantsplat.scene.gaussian_model_w_act import GaussianModel_With_Act
from instantsplat.scene.cameras import Camera
from instantsplat.gaussian_renderer import render
from instantsplat.arguments import PipelineParams, ArgumentParser

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import rerun as rr
from lpips import LPIPS
from omegaconf import OmegaConf


random.seed(1881)

rr.init("debugging_20_images", recording_id="v0.1")
rr.connect_tcp("0.0.0.0:9876")
rr.log("world/xyz", rr.Arrows3D(
    vectors=[[1, 0, 0], [0, 2, 0], [0, 0, 3]],
    colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]
))

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


### GAUSSIAN HEAD
def set_gaussian_model_features(model: GaussianModel, features: dict):
    model._xyz = features["xyz"]
    model._features_dc = features["features_dc"]
    model._features_rest = features["features_rest"]
    model._opacity = features["opacity"]
    model._scaling = features["scaling"]
    model._rotation = features["rotation"]
    model.max_radii2D = torch.zeros((features["xyz"].shape[0]), device=features["xyz"].device)

model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
print("Model is loaded!")

for param in model.parameters():
    param.requires_grad = False
for param in model.gs_head_feats.parameters():
    param.requires_grad = True
for param in model.aggregator.parameters():
    param.requires_grad = False
model.gs_head_feats.train()

final_conv_layer = model.gs_head_feats.scratch.output_conv2[-1]
splits_and_inits = [
    (3, 0.00003, -7.0),  # Scales
    (4, 1.0, 0.0),       # Rotations
    (3, 1.0, 0.0),       # SH
    (1, 1.0, 0.0),       # Opacity
    (3, 0.00001, 0.0)    # Mean offset
]
start = 0
for out_channel, std, bias_val in splits_and_inits:
    end = start + out_channel
    torch.nn.init.xavier_uniform_(final_conv_layer.weight[start:end], gain=std)
    torch.nn.init.constant_(final_conv_layer.bias[start:end], bias_val)
    start = end

### DATASET
cfg_file_path = "./output/human_scene/neuman/lab/hugs_trimlp/demo-dataset.seq=lab/2025-03-30_19-55-08/config_train.yaml"
cfg = OmegaConf.merge(default_cfg, OmegaConf.load(cfg_file_path))
dataset = get_train_dataset(cfg)

AVAILABLE_SEQUENCES = ["lab", "bike", "citron", "jogging"]

def load_dataset_for_sequence(seq_name):
    cfg_path = f"./output/human_scene/neuman/lab/hugs_trimlp/demo-dataset.seq=lab/2025-03-30_19-55-08/config_train.yaml"
    cfg = OmegaConf.merge(default_cfg, OmegaConf.load(cfg_path))
    cfg.dataset.seq = seq_name
    return get_train_dataset(cfg)

ALL_DATASETS = {seq: load_dataset_for_sequence(seq) for seq in AVAILABLE_SEQUENCES}

def get_data_sequence(seq_name, idx):
    dataset_seq = ALL_DATASETS[seq_name]
    return dataset_seq[idx]

def get_data(dataset, idx):
    return dataset[idx]

### RENDERER
def render_gaussians(gaussians, camera_list):
    parser = ArgumentParser(description="Training script parameters")
    pipe = PipelineParams(parser)
    args = parser.parse_args(sys.argv[1:])
    pipe = pipe.extract(args)
    pose = gaussians.get_RT(0)
    bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    return render(camera_list[0], gaussians, pipe, bg, camera_pose=pose)


### OPTIMIZER
optimizer = torch.optim.AdamW(
    [{"params": model.gs_head_feats.parameters(), "lr": 0.00005}],
    lr=2e-4,
    betas=(0.9, 0.999),
    eps=1e-15,
    weight_decay=0.0
)
print("optimizer init")

### losses
lpips = LPIPS(net="vgg", pretrained=True).to(device)
for param in lpips.parameters():
    param.requires_grad = False

l1_losses, ssim_losses, lpips_losses = [], [], []
l1_human_losses, ssim_human_losses, lpips_human_losses = [], [], []
steps = []

### TRAINING PARAMS
lambda_1, lambda_2, lambda_3 = 0.8, 0.2, 1.0
iterations = 30000
N_OF_IMAGES_IN_ONE_STEP_EVAL = 3
MAX_IMAGES = 40
#image_ids = dataset.train_split[:20]
notes_for_folder_name = 'WHITE_BG'
exp_name = f"MULTI_IMG_GS_TRAINING_SINGLE_HEAD_W_OFFSET_RANDOM_LENGTH_IMAGE_{iterations}_{lambda_1}_{lambda_2}_{lambda_3}"
folder_name = f"{exp_name}-{notes_for_folder_name}-{datetime.now().strftime('%d-%m_%H-%M')}"
os.makedirs(f"./_irem/{folder_name}", exist_ok=True)


### FIXED SEQUENCE, OVERFITTING IMAGES
# img_path = [f"./data/neuman/dataset/lab/images/{i:05}.png" for i in image_ids]
# images = load_and_preprocess_images(img_path).to(device)[None]
# aggregated_tokens_list, ps_idx = model.aggregator(images)

# pose_enc = model.camera_head(aggregated_tokens_list)[-1]
# extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
# fov_h, fov_w = pose_enc[..., 7], pose_enc[..., 8]
to_tensor = TF.ToTensor()

writer = SummaryWriter(log_dir=f'./_irem/{folder_name}/tb_logs')



from collections import defaultdict
IMAGE_CACHE = defaultdict(dict)
print("Caching all images to memory...")
for seq_name in AVAILABLE_SEQUENCES:
    print(f"Loading sequence: {seq_name}")
    dataset = ALL_DATASETS[seq_name]
    for image_id in tqdm(dataset.train_split, desc=f"Caching {seq_name}"):
        img_path = f"./data/neuman/dataset/{seq_name}/images/{image_id:05}.png"
        image = load_and_preprocess_images([img_path]).squeeze(0).to(device)  # CHW tensor
        IMAGE_CACHE[seq_name][image_id] = image  # Do not batch it here


model.gs_head_feats.load_state_dict(torch.load("./_irem/MULTI_IMG_GS_TRAINING_SINGLE_HEAD_W_OFFSET_RANDOM_LENGTH_IMAGE_30000_0.8_0.2_1.0-WHITE_BG-24-05_13-49/gs_head_feats_step_2000.pth"))

### TRAINING LOOP
for step in tqdm(range(iterations), desc="Training", total=iterations):
    ## DYNAMIC SEQUENCE LENGTH AND RANDOM IMAGES
    chosen_seq = random.sample(AVAILABLE_SEQUENCES, 1)[0]
    chosen_dataset = ALL_DATASETS[chosen_seq]

    used_image_count = random.randint(1, min(len(chosen_dataset.train_split), MAX_IMAGES))
    image_ids = random.sample(chosen_dataset.train_split, used_image_count)
    img_path = [f"./data/neuman/dataset/{chosen_seq}/images/{i:05}.png" for i in image_ids]

    # images = load_and_preprocess_images(img_path).to(device)[None]
    images = torch.stack([IMAGE_CACHE[chosen_seq][i] for i in image_ids])[None]  # [1, B, C, H, W]

    aggregated_tokens_list, ps_idx = model.aggregator(images)
    pose_enc = model.camera_head(aggregated_tokens_list)[-1]
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
    fov_h, fov_w = pose_enc[..., 7], pose_enc[..., 8]


    # for i in range(len(image_ids)):
    feats, _ = model.gs_head_feats(aggregated_tokens_list, images, ps_idx)
    point_map, _ = model.point_head(aggregated_tokens_list, images, ps_idx)

    supervised_indices = random.sample(range(used_image_count), min(N_OF_IMAGES_IN_ONE_STEP_EVAL, used_image_count))
    total_loss = 0
    total_l1, total_ssim, total_lpips = 0, 0, 0
    total_l1_human, total_ssim_human, total_lpips_human = 0, 0, 0

    # Forward pass through gs_head
    scale   = feats[:,:,:,:,0:3]
    rot     = feats[:,:,:,:,3:7]
    sh      = feats[:,:,:,:,7:10]
    op      = feats[:,:,:,:,10:11]
    offset  = feats[:,:,:,:,11:14]

    for b in supervised_indices:

        image_idx = image_ids[b]


        cam_to_world_extrinsic = closed_form_inverse_se3(extrinsic[0, b, :, :][None])[0].detach().cpu().numpy()
        R_cam_to_world = cam_to_world_extrinsic[:3, :3]
        t_cam_to_world = cam_to_world_extrinsic[:3, 3]
        # R_w2c = R_cam_to_world.T
        T_w2c = -R_cam_to_world.T @ t_cam_to_world

        # R_c2w = extrinsic[0, b, :, :3].cpu().detach().numpy()  # camera-to-world
        # T_c2w = extrinsic[0, b, :, 3:].squeeze(1).cpu().detach().numpy()
        # R_w2c = R_c2w.T
        # T_w2c = -R_c2w.T @ T_c2w
        cam = Camera(
            colmap_id=1,
            R=R_cam_to_world,
            T=T_w2c,
            FoVx=fov_w[0, b],
            FoVy=fov_h[0, b],
            image=to_tensor(Image.open(img_path[b]).convert('RGB')),
            gt_alpha_mask=None,
            image_name=f'{image_idx:05}',
            uid=0
        )

        get_data_train_index = chosen_dataset.train_split.index(image_idx)
        data = get_data(chosen_dataset, get_data_train_index)
        if (to_tensor(Image.open(img_path[b]).convert('RGB')).cuda() - data["rgb"]).abs().sum().item() > 0.01:
            raise Exception("oops!")
            
        xyz = point_map[0,b].reshape(-1, 3) + offset[0, b, :, :, 0:3].view(-1, 3)
        features = {
            "xyz":              xyz,
            "features_dc":      sh[0, b, :, :, 0:3].view(-1, 1, 3),
            "features_rest": torch.zeros((sh[0, b, :, :, 0:3].detach().view(-1, 3, 1).shape[0], 15, 3), device=xyz.device),
            "opacity":          op[0, b, :, :, 0].view(-1, 1),
            "scaling":          scale[0, b, :, :, 0:3].view(-1, 3),
            "rotation":         rot[0, b, :, :, 0:4].view(-1, 4),
        }

        # rr.set_time_seconds("frame", 0)
        # rr.log("world/xyz", rr.Arrows3D(
        #     vectors=[[1, 0, 0], [0, 2, 0], [0, 0, 3]],
        #     colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]
        # ))
        # rr.log(f"world/scene", rr.Points3D(positions=features["xyz"].detach().cpu().numpy()))
        # rr.log(f"world/scene_all", rr.Points3D(positions=point_map.view(-1, 3).detach().cpu().numpy()))
        # pos = cam.camera_center.cpu().detach().numpy()
        # axis_angle = matrix_to_axis_angle(torch.tensor(cam.R)).cpu().detach().numpy()
        # rr.log(
        #     f"world/camera",
        #     rr.Transform3D(
        #         translation=pos,
        #         rotation=rr.RotationAxisAngle(axis=axis_angle[:3] / np.linalg.norm(axis_angle), angle=np.linalg.norm(axis_angle))
        #     )
        # )
        # rr.log(f"world/camera/image", rr.Pinhole(image_from_camera=intrinsic[0, b, :, :].cpu().detach().numpy()))
        # rr.log(f"world/camera/image", rr.Image(cam.original_image.permute(1,2,0).cpu().detach().numpy()))



        gaussians = GaussianModel_With_Act(0)
        gaussians.init_RT_seq({1.0: [cam]})


        set_gaussian_model_features(gaussians, features)

        render_pkg = render_gaussians(gaussians,  [cam])
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        if step % 100 == 0:
            #continue
            torchvision.utils.save_image(image, f"./_irem/{folder_name}/render_step_{step}_seq_{chosen_seq}_cam_{image_idx}_used_{used_image_count}_images_exp_{exp_name}.png")

        mask = data["mask"]
        bbox = data["bbox"]
        x1, y1, x2, y2 = data["bbox"].int().tolist()

        gt_crop = data["rgb"][:, x1:x2, y1:y2]
        pred_crop = image[:, x1:x2, y1:y2]

        l1_human = torch.abs(pred_crop - gt_crop).mean()

        l1_loss = torch.abs((image - data["rgb"])).mean()
        ssim_loss = (1 - ssim(image, data["rgb"]))
        lpips_loss = lpips(image.clip(max=1), data["rgb"]).mean()
        l_full = lambda_1 * l1_loss + lambda_2 * ssim_loss + lambda_3 * lpips_loss
    
        ssim_human =  (1 - ssim(pred_crop, gt_crop))
        lpips_human = lpips(pred_crop.clip(max=1), gt_crop).mean()
        loss_human = lambda_1 * l1_human + lambda_2 * ssim_human + lambda_3 * lpips_human


        loss_full = lambda_1 * l1_loss + lambda_2 * ssim_loss + lambda_3 * lpips_loss
        loss_human = lambda_1 * l1_human + lambda_2 * ssim_human + lambda_3 * lpips_human
        loss = loss_full + loss_human

        total_loss += loss
        total_l1 += l1_loss.item()
        total_ssim += ssim_loss.item()
        total_lpips += lpips_loss.item()
        total_l1_human += l1_human.item()
        total_ssim_human += ssim_human.item()
        total_lpips_human += lpips_human.item()


    total_loss = total_loss / N_OF_IMAGES_IN_ONE_STEP_EVAL
    total_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    writer.add_scalar("Loss/L1", total_l1 / len(supervised_indices), step)
    writer.add_scalar("Loss/SSIM", total_ssim / len(supervised_indices), step)
    writer.add_scalar("Loss/LPIPS", total_lpips / len(supervised_indices), step)
    writer.add_scalar("Loss/L1_Human", total_l1_human / len(supervised_indices), step)
    writer.add_scalar("Loss/SSIM_Human", total_ssim_human / len(supervised_indices), step)
    writer.add_scalar("Loss/LPIPS_Human", total_lpips_human / len(supervised_indices), step)

    steps.append(step)
    l1_losses.append(total_l1 / len(supervised_indices))
    ssim_losses.append(total_ssim / len(supervised_indices))
    lpips_losses.append(total_lpips / len(supervised_indices))
    l1_human_losses.append(total_l1_human / len(supervised_indices))
    ssim_human_losses.append(total_ssim_human / len(supervised_indices))
    lpips_human_losses.append(total_lpips_human / len(supervised_indices))

    if step % 10 == 1:
        tqdm.write(f"Step {step:07}, Avg L1: {total_l1/4:.4f}, SSIM: {total_ssim/4:.4f}, LPIPS: {total_lpips/4:.4f}, "
                   f"L1-H: {total_l1_human/4:.4f}, SSIM-H: {total_ssim_human/4:.4f}, LPIPS-H: {total_lpips_human/4:.4f},"
                   f"Used Image Count: {used_image_count}")
    
    if step % 1000 == 0 and step != 0:
        print("GS_FEAT_MODEL CKPT IS SAVED!!!!!!!")
        torch.save(model.gs_head_feats.state_dict(), f"./_irem/{folder_name}/gs_head_feats_step_{step}.pth")

    if step % 100 == 0 and step != 0:
        print("GS_FEAT_MODEL CKPT IS SAVED!!!!!!!")
        torch.save(model.gs_head_feats.state_dict(), f"./_irem/{folder_name}/gs_head_feats_step.pth")

    

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

# # # # SAVE THE MODEL
torch.save(model.gs_head_feats.state_dict(), f"./_irem/{folder_name}/gs_head_feats.pth")

