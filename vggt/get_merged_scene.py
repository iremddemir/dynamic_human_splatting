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
from hugs.trainer.gs_trainer import get_train_dataset, get_train_dataset_ordered
from omegaconf import OmegaConf
from hugs.cfg.config import cfg as default_cfg
import torchvision
from instantsplat.arguments import PipelineParams, ArgumentParser
import random
from PIL import Image
from hugs.losses.utils import ssim
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


dataset = get_train_dataset(cfg)


# dataset = get_train_dataset_ordered(cfg)
def get_data(idx):
    chosen_idx = dataset.train_split.index(idx)
    data = dataset[chosen_idx]
    return data


# image_ids = dataset.train_split[:20]
image_ids = dataset.train_split
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
exp_name = f"6_GET_SCENE"

import os
from datetime import datetime

timestamp = datetime.now().strftime("%d-%m_%H-%M")
# folder_name = f"{timestamp}_{exp_name}"
folder_name = f"{exp_name}-{timestamp}"
os.makedirs(f"./_irem/{folder_name}", exist_ok=True)
print("STARTING TRAINING")


#########################################################################################################################


per_frame_canonical_human = []


random.seed(1)


model.gs_head_feats.load_state_dict(torch.load("last_gs_head_feats_step.pth"))
for param in model.parameters():
    param.requires_grad = False
model.eval()


n_of_smpl_vertices = smpl().vertices.shape[1]
global_merged_human = {
    "xyz": torch.zeros((n_of_smpl_vertices, 3), device="cpu"),
    "features_dc": torch.zeros((n_of_smpl_vertices, 1, 3), device="cpu"),
    "features_rest": torch.zeros((n_of_smpl_vertices, 15, 3), device="cpu"),
    "opacity": torch.zeros((n_of_smpl_vertices, 1), device="cpu"),
    "scaling": torch.zeros((n_of_smpl_vertices, 3), device="cpu"),
    "rotation": torch.zeros((n_of_smpl_vertices, 4), device="cpu"),
}
# all_valid_masks = torch.zeros((n_of_smpl_vertices,))
all_valid_masks = torch.ones((n_of_smpl_vertices,), dtype=torch.bool, device="cuda")


from torchvision import transforms as TF

to_tensor = TF.ToTensor()

img_path = image_names[::5]
image_ids = image_ids[::5]
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


global_scene = {
    "xyz": [],
    "features_dc": [],
    "features_rest": [],
    "opacity": [],
    "scaling": [],
    "rotation": [],
}

for i in range(0, len(image_ids), 1):
    print(i)

    b = i

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

    # preprocessed_mask = preprocess_masks([data["mask"]]).view(-1)
    raw_mask = preprocess_masks([data["mask"]])[0]  # [1, H, W]
    kernel = torch.ones((1, 1, 9, 9), device=raw_mask.device)  # 5x5 dilation kernel
    dilated_mask = (
        F.conv2d(raw_mask.unsqueeze(0).float(), kernel, padding=4).clamp(0, 1).bool()
    )
    preprocessed_mask = dilated_mask.view(-1)

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

    features_scene = {
        "xyz": features_scene_human["xyz"][~preprocessed_mask.bool()],
        "features_dc": features_scene_human["features_dc"][~preprocessed_mask.bool()],
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
        features_scene,
        save_path=f"./_irem/{folder_name}/{image_idx}_scene_rendered.png",
        bg="white",
    )

    global_scene["xyz"].append(features_scene["xyz"].cpu())
    global_scene["features_dc"].append(features_scene["features_dc"].cpu())
    global_scene["features_rest"].append(features_scene["features_rest"].cpu())
    global_scene["opacity"].append(features_scene["opacity"].cpu())
    global_scene["scaling"].append(features_scene["scaling"].cpu())
    global_scene["rotation"].append(features_scene["rotation"].cpu())

    continue
    # continue
    ############################################################################


for key in global_scene:
    global_scene[key] = torch.cat(global_scene[key], dim=0).to(device)

import pickle

with open("global_scene.pkl", "wb") as f:
    pickle.dump(global_scene, f)


for i in range(0, len(image_ids), 1):
    print(i)

    b = i

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
        T=T_w2c,
        FoVx=fov_w[0, b],
        FoVy=fov_h[0, b],
        image=to_tensor(Image.open(img_path[b]).convert("RGB")),
        gt_alpha_mask=None,
        image_name=f"{image_idx:05}",
        uid=0,
    )
    camera_list = [camera]

    gaussians = GaussianModel_With_Act(0)
    gaussians.init_RT_seq({1.0: camera_list})

    render_and_save(
        gaussians,
        camera_list,
        global_scene,
        save_path=f"./_irem/{folder_name}/{image_idx}_global_scene_rendered.png",
        bg="white",
    )
