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
exp_name = f"1000_EXPERIMENTAL_OFFSET"

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

# img_path = image_names
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


import torch
from instantsplat.utils.general_utils import inverse_sigmoid
from instantsplat.utils.general_utils import build_scaling_rotation, strip_symmetric
from instantsplat.utils.general_utils import build_rotation


def split_features_by_mask(
    features: dict,
    mask: torch.Tensor,
    N: int = 2,
    scaling_modifier: float = 0.8,
) -> dict:
    """
    Split each gaussian where mask[i]==1 into N new gaussians sampled
    from its own covariance, shrink their scales, and then remove the originals.

    Args:
        features: dict with keys
            "xyz"           : (P,3) float Tensor
            "features_dc"   : (P,1,F_dc) float Tensor
            "features_rest" : (P,1,F_rest) float Tensor
            "opacity"       : (P,1) float Tensor (logit space)
            "scaling"       : (P,3) float Tensor (log scale)
            "rotation"      : (P,4) float Tensor (quaternions, unnormalized)
        mask:  (P,) bool or {0,1} Tensor selecting which gaussians to split.
        N:     number of children per selected Gaussian.
        scaling_modifier: factor in denominator when shrinking child scales.

    Returns:
        new_features: same keys as input, with length
                      P - M + M*N  where M=mask.sum().
    """
    device = features["xyz"].device
    mask = mask.bool().to(device)
    P = features["xyz"].shape[0]
    M = int(mask.sum().item())
    if M == 0:
        return features.copy()

    # 1) extract parent params
    parent_xyz = features["xyz"][mask]  # (M,3)
    parent_fdc = features["features_dc"][mask]  # (M,1,Fdc)
    parent_frest = features["features_rest"][mask]  # (M,1,Fr)
    parent_op = features["opacity"][mask]  # (M,1)
    parent_scaling = torch.exp(
        features["scaling"][mask]
    )  # convert log-scale -> scale (M,3)
    parent_rot = features["rotation"][mask]  # (M,4)

    # 2) sample N children per parent
    stds = parent_scaling.repeat(N, 1)  # (N*M,3)
    samples = torch.normal(mean=torch.zeros_like(stds), std=stds)  # (N*M,3)
    rots_mat = build_rotation(parent_rot).repeat(N, 1, 1)  # (N*M,3,3)
    world_offsets = torch.bmm(rots_mat, samples.unsqueeze(-1)).squeeze(-1)
    child_xyz = world_offsets + parent_xyz.repeat(N, 1)  # (N*M,3)

    # 3) shrink child scales in log-space
    child_scaling = torch.log(parent_scaling.repeat(N, 1) / (scaling_modifier * N))

    # 4) replicate other attributes
    child_rot = parent_rot.repeat(N, 1)  # (N*M,4)
    child_fdc = parent_fdc.repeat(N, 1, 1)  # (N*M,1,Fdc)
    child_frest = parent_frest.repeat(N, 1, 1)  # (N*M,1,Fr)
    child_op = parent_op.repeat(N, 1)  # (N*M,1)

    # 5) keep only the unmasked originals
    keep = ~mask
    out_xyz = features["xyz"][keep]  # (P-M,3)
    out_fdc = features["features_dc"][keep]
    out_frest = features["features_rest"][keep]
    out_op = features["opacity"][keep]
    out_scaling = features["scaling"][keep]
    out_rotation = features["rotation"][keep]

    # 6) concatenate children
    new_features = {
        "xyz": torch.cat([out_xyz, child_xyz], dim=0),
        "features_dc": torch.cat([out_fdc, child_fdc], dim=0),
        "features_rest": torch.cat([out_frest, child_frest], dim=0),
        "opacity": torch.cat([out_op, child_op], dim=0),
        "scaling": torch.cat([out_scaling, child_scaling], dim=0),
        "rotation": torch.cat([out_rotation, child_rot], dim=0),
    }

    # 7) build the new mask: first (P-M) False, then (N*M) True
    keep_count = P - M
    child_count = N * M
    new_mask = torch.cat(
        [
            torch.zeros(keep_count, dtype=torch.bool, device=device),
            torch.ones(child_count, dtype=torch.bool, device=device),
        ],
        dim=0,
    )

    return new_features, new_mask


for i in range(0, len(image_ids), 2):
    img_path = [image_names[i]]
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

    print(i)

    # b = i
    b = 0

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

    # features_scene_human, preprocessed_mask = split_features_by_mask(features_scene_human, preprocessed_mask.bool(), N=3, scaling_modifier=0.5)
    # render_and_save(gaussians, camera_list, features_scene_human, save_path=f"./_irem/{folder_name}/{image_idx}_splitted.png", bg='white')

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
        features_scene_human,
        save_path=f"./_irem/{folder_name}/{image_idx}_rendered_whole_scene.png",
        bg="white",
    )
    render_and_save(
        gaussians,
        camera_list,
        features_scene_human,
        save_path=f"./_irem/{folder_name}/{image_idx}_rendered_whole_scene_green.png",
        bg="green",
    )
    render_and_save(
        gaussians,
        camera_list,
        features_human,
        save_path=f"./_irem/{folder_name}/{image_idx}_human_gaussians_black.png",
        bg="black",
    )
    render_and_save(
        gaussians,
        camera_list,
        features_human,
        save_path=f"./_irem/{folder_name}/{image_idx}_human_gaussians_white.png",
        bg="white",
    )

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

    preprocessed_rgb = load_and_preprocess_images(
        [f"./_irem/{folder_name}/{image_idx}_rendered_whole_scene.png"]
    ).to(device)
    overlay_points_and_save(
        preprocessed_rgb,
        smpl_locs[~eq],
        f"./_irem/{folder_name}/{image_idx}_overlay_removed_smpl_vertices_by_mask.png",
    )
    overlay_points_and_save(
        preprocessed_rgb,
        mask_pixels_locs,
        f"./_irem/{folder_name}/{image_idx}_overlay_gt_human_mask.png",
    )

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

    render_and_save(
        gaussians,
        camera_list,
        new_features,
        save_path=f"./_irem/{folder_name}/{image_idx}_human_gaussians_masked_black.png",
        bg="black",
    )
    render_and_save(
        gaussians,
        camera_list,
        new_features,
        save_path=f"./_irem/{folder_name}/{image_idx}_human_gaussians_masked_white.png",
        bg="white",
    )

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

    preprocessed_rgb = load_and_preprocess_images(
        [f"./_irem/{folder_name}/{image_idx}_rendered_whole_scene.png"]
    ).to(device)
    overlay_points_and_save(
        preprocessed_rgb,
        smpl_locs[valid_mask],
        f"./_irem/{folder_name}/{image_idx}_overlay_matched_smpl_vertices.png",
    )

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
    render_and_save(
        gaussians,
        camera_list,
        unmatched_human_gaussian_features,
        save_path=f"./_irem/{folder_name}/{image_idx}_unmatched_human_gaussians.png",
        bg="white",
    )
    preprocessed_rgb = load_and_preprocess_images(
        [f"./_irem/{folder_name}/{image_idx}_rendered_whole_scene.png"]
    ).to(device)
    overlay_points_and_save(
        preprocessed_rgb,
        unmatched_human_gaussian_locs,
        f"./_irem/{folder_name}/{image_idx}_overlay_unmatched_gaussians.png",
    )

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
    render_and_save(
        gaussians,
        camera_list,
        matched_human_gaussian_features,
        save_path=f"./_irem/{folder_name}/{image_idx}_matched_human_gaussians.png",
        bg="white",
    )

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
    render_smpl_gaussians(
        canonical_human_features_at_smpl,
        f"./_irem/{folder_name}/{image_idx}_canonical_human_black.png",
        "black",
    )
    render_smpl_gaussians(
        canonical_human_features_at_smpl,
        f"./_irem/{folder_name}/{image_idx}_canonical_human_white.png",
        "white",
    )
    render_smpl_gaussians_gif(
        canonical_human_features_at_smpl,
        f"./_irem/{folder_name}/{image_idx}_canonical_human_white.gif",
    )

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
    render_smpl_gaussians(
        offsets_gaussians_canonical,
        f"./_irem/{folder_name}/{image_idx}_canonical_human_only_offsets_black.png",
        "black",
    )
    render_smpl_gaussians(
        offsets_gaussians_canonical,
        f"./_irem/{folder_name}/{image_idx}_canonical_human_only_offsets_white.png",
        "white",
    )
    render_smpl_gaussians_gif(
        offsets_gaussians_canonical,
        f"./_irem/{folder_name}/{image_idx}_canonical_human_only_offsets_white.gif",
    )

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
    render_smpl_gaussians(
        canonical_human_with_offset,
        f"./_irem/{folder_name}/{image_idx}_canonical_human_with_offsets_black.png",
        "black",
    )
    render_smpl_gaussians(
        canonical_human_with_offset,
        f"./_irem/{folder_name}/{image_idx}_canonical_human_with_offsets_white.png",
        "white",
    )
    render_smpl_gaussians_gif(
        canonical_human_with_offset,
        f"./_irem/{folder_name}/{image_idx}_canonical_human_with_offsets_white.gif",
    )

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
    render_and_save(
        gaussians,
        camera_list,
        offsets_gaussians_canonical_in_vggt_world,
        save_path=f"./_irem/{folder_name}/{image_idx}_canonical_offset_in_vggt_world_white.png",
        bg="white",
    )

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

    deformed_xyz, human_smpl_scales, deformed_gs_rotq, lbs_T = (
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

    #######
    new_features = {
        "xyz": deformed_xyz[0][valid_mask],
        "features_dc": features_human["features_dc"][matches],
        "features_rest": features_human["features_rest"][matches],
        "opacity": features_human["opacity"][matches],
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
    #######

    offsets_vggt = (
        torch.matmul(offsets_canonical, smpl_R) * smpl_scale
    )  # no need for transl, because the offset is relative
    deformed_gs_rotq = transform_rotations_through_lbs(
        offsets_gaussians_canonical["rotation"], lbs_T[valid_smpl_indices, :3, :3]
    )
    offsets_gaussians_canonical_to_posed = {
        "xyz": deformed_xyz[0][valid_smpl_indices] + offsets_vggt,
        "features_dc": offsets_gaussians_canonical["features_dc"],
        "features_rest": offsets_gaussians_canonical["features_rest"],
        "opacity": offsets_gaussians_canonical["opacity"],
        "scaling": offsets_gaussians_canonical["scaling"] + smpl_scale.log(),
        "rotation": deformed_gs_rotq,
    }
    render_and_save(
        gaussians,
        camera_list,
        offsets_gaussians_canonical_to_posed,
        save_path=f"./_irem/{folder_name}/{image_idx}_canon2pose_only_offsets.png",
        bg="white",
    )

    merged_features = {
        "xyz": torch.cat(
            [
                features_scene["xyz"],
                new_features["xyz"],
                offsets_gaussians_canonical_to_posed["xyz"],
            ],
            dim=0,
        ),
        "features_dc": torch.cat(
            [
                features_scene["features_dc"],
                new_features["features_dc"],
                offsets_gaussians_canonical_to_posed["features_dc"],
            ],
            dim=0,
        ),
        "features_rest": torch.cat(
            [
                features_scene["features_rest"],
                new_features["features_rest"],
                offsets_gaussians_canonical_to_posed["features_rest"],
            ],
            dim=0,
        ),
        "opacity": torch.cat(
            [
                features_scene["opacity"],
                new_features["opacity"],
                offsets_gaussians_canonical_to_posed["opacity"],
            ],
            dim=0,
        ),
        "scaling": torch.cat(
            [
                features_scene["scaling"],
                new_features["scaling"],
                offsets_gaussians_canonical_to_posed["scaling"],
            ],
            dim=0,
        ),
        "rotation": torch.cat(
            [
                features_scene["rotation"],
                new_features["rotation"],
                offsets_gaussians_canonical_to_posed["rotation"],
            ],
            dim=0,
        ),
    }
    render_and_save(
        gaussians,
        camera_list,
        merged_features,
        save_path=f"./_irem/{folder_name}/{image_idx}_canon2pose_human_with_scene_with_offsets.png",
        bg="white",
    )
    #######

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
        # "gs_rotq" :        deformed_gs_rotq_canon.to("cuda"),
        "scaling": canonical_human_scales.to("cuda"),
        "rotation": canonical_human_rotation.to("cuda"),
        "visibility_mask": valid_mask,
    }

    per_frame_canonical_human.append(canonical_human_features)

    print(
        f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! SAVED AFTER: {image_idx} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    )
    with open(f"./_irem/{folder_name}/per_frame_canonical_human.pkl", "wb") as f:
        pickle.dump(per_frame_canonical_human, f)

    # rr.set_time_seconds("frame", 0)
    # rr.log(f"world/xyz", rr.Arrows3D(vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]))
    # rr.log(f"world/scene", rr.Points3D(positions=features_scene["xyz"].detach().cpu().numpy(), colors=features_scene["features_dc"].squeeze(1).detach().cpu().numpy()))
    # rr.log(f"world/original_human_matched", rr.Points3D(positions=features_human["xyz"][matches].detach().cpu().numpy()))
    # rr.log(f"world/original_human", rr.Points3D(positions=features_human["xyz"].detach().cpu().numpy()))
    # rr.log(f"world/deformed_human", rr.Points3D(positions=deformed_smpl_at_canonical.squeeze(0).detach().cpu().numpy()))
    # rr.log(f"world/posed_smpl", rr.Points3D(positions=posed_smpl_at_canonical.detach().cpu().numpy()))
    # rr.log(f"world/posed_smpl", rr.Points3D(positions=((posed_smpl_at_canonical*smpl_scale) + smpl_transl).detach().cpu().numpy()))
    # rr.log(f"world/posed_from_canonical_human", rr.Points3D(positions=deformed_xyz[0].detach().cpu().numpy()))
