import sys

sys.path.append(".")

import torch
import torch.nn as nn
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
from hugs.utils.subdivide_smpl import subdivide_smpl_model
from hugs.models.modules.smpl_layer import SMPL
from hugs.models.modules.lbs import lbs_extra
from hugs.models.modules.lbs import *
from hugs.models.hugs_wo_trimlp import smpl_lbsweight_top_k
from hugs.cfg.constants import SMPL_PATH
from hugs.utils.rotations import quaternion_to_matrix, matrix_to_quaternion
import pdb


device = "cuda"
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# smpl = SMPL(SMPL_PATH).to(device)


subdivide_times = 1
smpl = subdivide_smpl_model(smoothing=True, n_iter=subdivide_times).to(device)
smpl_template = subdivide_smpl_model(smoothing=True, n_iter=subdivide_times).to(device)

posedirs = None
disable_posedirs = True


# Rendering utils

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

def render_gaussians(gaussians, camera_list, bg='white'):
    viewpoint_cam = camera_list[0]

    parser = ArgumentParser(description="Training script parameters")
    pipe = PipelineParams(parser)
    args = parser.parse_args(sys.argv[1:])
    pipe = pipe.extract(args)

    pose = gaussians.get_RT(viewpoint_cam.uid) 

    if bg == 'black':
        bg = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)
    elif bg == 'white':
        bg = torch.tensor([1, 1, 1], dtype=torch.float32, device=device)
    elif bg == 'green':
        bg = torch.tensor([0, 1, 0], dtype=torch.float32, device=device)

    render_pkg = render(viewpoint_cam, gaussians, pipe, bg, camera_pose=pose)
    
    return render_pkg


def render_and_save(gaussians, camera_list, features: dict, save_path: str, bg: str = 'white'):
    with torch.no_grad():
        set_gaussian_model_features(gaussians, features)
        render_pkg = render_gaussians(gaussians, camera_list, bg=bg)
        img = (render_pkg['render']
               .detach()
               .cpu()
               .contiguous()
               .clamp(0,1))
        torchvision.utils.save_image(img, save_path)
    return render_pkg['render']

def overlay_points_and_save(rgb_image, pts, save_path):
    rgb = rgb_image.squeeze(0).clone()

    if pts.numel() > 0:
        height, width = rgb.shape[-2], rgb.shape[-1]

        xs = pts[:, 0].long()
        ys = pts[:, 1].long()


        #rgb[:, ys, xs] = 0.5

        rgb[0, ys, xs] = 1.0  # Red channel
        rgb[1, ys, xs] = 0.0  # Green channel
        rgb[2, ys, xs] = 0.0  # Blue channel

    torchvision.utils.save_image(rgb, save_path)

def render_smpl_gaussians(features, save_path, bg):
    R = np.array([
        [1.0,  0.0,  0.0],
        [0.0, -1.0,  0.0],
        [0.0,  0.0, -1.0]
    ])  # Camera looking straight at origin
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
        scale=1.0
    )

    gaussians = GaussianModel_With_Act(0)
    gaussians.init_RT_seq({1.0: [cam]})
    set_gaussian_model_features(gaussians, features)
    render_pkg = render_gaussians(gaussians, [cam], bg=bg)
    torchvision.utils.save_image(render_pkg['render'], save_path)


def render_smpl_gaussians_gif(features, save_path="smpl_360.gif"):
    from hugs.datasets.utils import get_rotating_camera
    camera_params = get_rotating_camera(
        dist=5.0, img_size=512, 
        nframes=36, device='cuda',
        angle_limit=2*torch.pi,
    )

    frames = []

    for cam_param in camera_params:
        # === Extract R and T from cam_ext ===
        cam_ext = cam_param['cam_ext'].T.cpu()  # (4x4 matrix)
        R = cam_ext[:3, :3].numpy()           # 3x3 rotation matrix
        T = cam_ext[:3, 3].numpy()            # 3D translation vector

        T[1] = -0.35

        # === Other parameters ===
        FoVx = cam_param['fovx']
        FoVy = cam_param['fovy']
        image = torch.ones(3, cam_param['image_height'], cam_param['image_width'])  # dummy image
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
            uid=uid
        )

        gaussians = GaussianModel_With_Act(0)
        gaussians.init_RT_seq({1.0: [cam]})
        set_gaussian_model_features(gaussians, features)
        render_pkg = render_gaussians(gaussians, [cam], bg="white")
        # Convert rendered tensor to PIL Image
        image_tensor = render_pkg['render'].detach().cpu().clamp(0, 1)
        image_pil = torchvision.transforms.functional.to_pil_image(image_tensor)
        frames.append(image_pil)

    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        duration=100,  # milliseconds per frame
        loop=0         # loop forever
    )




def get_deformed_gs_xyz(
    xyz,
    xyz_offset,
    gs_scales,
    rot4d_quaternion,
    betas,
    body_pose,
    global_orient,
    smpl_scale=None,
    transl=None,
):
    def get_vitruvian_verts():
        vitruvian_pose = torch.zeros(69, dtype=smpl.dtype, device=device)
        vitruvian_pose[2] = 1.0
        vitruvian_pose[5] = -1.0
        smpl_output = smpl(
            body_pose=vitruvian_pose[None], betas=betas[None], disable_posedirs=False
        )
        vitruvian_verts = smpl_output.vertices[0]
        A_t2vitruvian = smpl_output.A[0].detach()
        T_t2vitruvian = smpl_output.T[0].detach()
        inv_T_t2vitruvian = torch.inverse(T_t2vitruvian)
        inv_A_t2vitruvian = torch.inverse(A_t2vitruvian)
        canonical_offsets = smpl_output.shape_offsets + smpl_output.pose_offsets
        canonical_offsets = canonical_offsets[0].detach()
        vitruvian_verts = vitruvian_verts.detach()
        return vitruvian_verts.detach(), inv_A_t2vitruvian

    def get_vitruvian_verts_template():
        vitruvian_pose = torch.zeros(69, dtype=smpl_template.dtype, device=device)
        vitruvian_pose[2] = 1.0
        vitruvian_pose[5] = -1.0
        smpl_output = smpl_template(
            body_pose=vitruvian_pose[None], betas=betas[None], disable_posedirs=False
        )
        vitruvian_verts = smpl_output.vertices[0]
        return vitruvian_verts.detach()

    vitruvian_verts, inv_A_t2vitruvian = get_vitruvian_verts()

    smpl_output = smpl(
        betas=betas.unsqueeze(0),
        body_pose=body_pose.unsqueeze(0),
        global_orient=global_orient.unsqueeze(0),
        disable_posedirs=False,
        return_full_pose=True,
    )

    gt_lbs_weights = None
    A_t2pose = smpl_output.A[0]
    A_vitruvian2pose = A_t2pose @ inv_A_t2vitruvian

    if xyz is None:
        t_pose_verts = get_vitruvian_verts_template()
        xyz = nn.Parameter(t_pose_verts.requires_grad_(False))

    if xyz_offset is None:
        gs_xyz = xyz
    else:
        gs_xyz = xyz + xyz_offset

    with torch.no_grad():
        # gt lbs is needed for lbs regularization loss
        # predicted lbs should be close to gt lbs
        _, gt_lbs_weights = smpl_lbsweight_top_k(
            lbs_weights=smpl.lbs_weights,
            points=gs_xyz.unsqueeze(0),
            template_points=vitruvian_verts.unsqueeze(0),
        )
        gt_lbs_weights = gt_lbs_weights.squeeze(0)
        if abs(gt_lbs_weights.sum(-1).mean().item() - 1) < 1e-7:
            pass
        else:
            print(
                f"GT LBS weights should sum to 1, but it is: {gt_lbs_weights.sum(-1).mean().item()}"
            )

        deformed_xyz, _, lbs_T, _, _ = lbs_extra(
            A_vitruvian2pose[None],
            gs_xyz[None],
            posedirs,
            gt_lbs_weights,
            smpl_output.full_pose,
            disable_posedirs=disable_posedirs,
            pose2rot=True,
        )

        deformed_xyz = deformed_xyz.squeeze(0)
        lbs_T = lbs_T.squeeze(0)

    if smpl_scale is not None:
        deformed_xyz = deformed_xyz * smpl_scale.unsqueeze(0)
        if gs_scales is not None:
            gs_scales = gs_scales * smpl_scale.unsqueeze(0)

    if transl is not None:
        deformed_xyz = deformed_xyz + transl.unsqueeze(0)

    if rot4d_quaternion is not None:
        gs_rotmat = quaternion_to_matrix(rot4d_quaternion)
        deformed_gs_rotmat = lbs_T[:, :3, :3] @ gs_rotmat
        deformed_gs_rotq = matrix_to_quaternion(deformed_gs_rotmat)

    return deformed_xyz  # , deformed_gs_rotq, gs_scales


def get_deformed_gs_xyz_with_faces(
    xyz,
    xyz_offset,
    gs_scales,
    rot4d_quaternion,
    betas,
    body_pose,
    global_orient,
    smpl_scale=None,
    transl=None,
):
    def get_vitruvian_verts():
        vitruvian_pose = torch.zeros(69, dtype=smpl.dtype, device=device)
        vitruvian_pose[2] = 1.0
        vitruvian_pose[5] = -1.0
        smpl_output = smpl(
            body_pose=vitruvian_pose[None], betas=betas[None], disable_posedirs=False
        )
        vitruvian_verts = smpl_output.vertices[0]
        A_t2vitruvian = smpl_output.A[0].detach()
        T_t2vitruvian = smpl_output.T[0].detach()
        inv_T_t2vitruvian = torch.inverse(T_t2vitruvian)
        inv_A_t2vitruvian = torch.inverse(A_t2vitruvian)
        canonical_offsets = smpl_output.shape_offsets + smpl_output.pose_offsets
        canonical_offsets = canonical_offsets[0].detach()
        vitruvian_verts = vitruvian_verts.detach()
        return vitruvian_verts.detach(), inv_A_t2vitruvian

    def get_vitruvian_verts_template():
        vitruvian_pose = torch.zeros(69, dtype=smpl_template.dtype, device=device)
        vitruvian_pose[2] = 1.0
        vitruvian_pose[5] = -1.0
        smpl_output = smpl_template(
            body_pose=vitruvian_pose[None], betas=betas[None], disable_posedirs=False
        )
        vitruvian_verts = smpl_output.vertices[0]
        return vitruvian_verts.detach()

    faces = torch.from_numpy(smpl.faces.astype("int64")).to(device)  # (F,3)

    vitruvian_verts, inv_A_t2vitruvian = get_vitruvian_verts()

    smpl_output = smpl(
        betas=betas.unsqueeze(0),
        body_pose=body_pose.unsqueeze(0),
        global_orient=global_orient.unsqueeze(0),
        disable_posedirs=False,
        return_full_pose=True,
    )

    gt_lbs_weights = None
    A_t2pose = smpl_output.A[0]
    A_vitruvian2pose = A_t2pose @ inv_A_t2vitruvian

    if xyz is None:
        t_pose_verts = get_vitruvian_verts_template()
        xyz = nn.Parameter(t_pose_verts.requires_grad_(False))

    if xyz_offset is None:
        gs_xyz = xyz
    else:
        gs_xyz = xyz + xyz_offset

    with torch.no_grad():
        # gt lbs is needed for lbs regularization loss
        # predicted lbs should be close to gt lbs
        _, gt_lbs_weights = smpl_lbsweight_top_k(
            lbs_weights=smpl.lbs_weights,
            points=gs_xyz.unsqueeze(0),
            template_points=vitruvian_verts.unsqueeze(0),
        )
        gt_lbs_weights = gt_lbs_weights.squeeze(0)
        if abs(gt_lbs_weights.sum(-1).mean().item() - 1) < 1e-7:
            pass
        else:
            print(
                f"GT LBS weights should sum to 1, but it is: {gt_lbs_weights.sum(-1).mean().item()}"
            )

        deformed_xyz, _, lbs_T, _, _ = lbs_extra(
            A_vitruvian2pose[None],
            gs_xyz[None],
            posedirs,
            gt_lbs_weights,
            smpl_output.full_pose,
            disable_posedirs=disable_posedirs,
            pose2rot=True,
        )

        deformed_xyz = deformed_xyz.squeeze(0)
        lbs_T = lbs_T.squeeze(0)

    if smpl_scale is not None:
        deformed_xyz = deformed_xyz * smpl_scale.unsqueeze(0)
        # gs_scales = gs_scales * smpl_scale.unsqueeze(0)

    if transl is not None:
        deformed_xyz = deformed_xyz + transl.unsqueeze(0)

    # gs_rotmat = quaternion_to_matrix(rot4d_quaternion)
    # deformed_gs_rotmat = lbs_T[:, :3, :3] @ gs_rotmat
    # deformed_gs_rotq = matrix_to_quaternion(deformed_gs_rotmat)

    return deformed_xyz, faces


import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


def preprocess_masks(mask_tensor_list, mode="crop", target_size=518):
    """
    Preprocess mask tensors (0/1 values) similarly to images:
    - Resize, crop or pad to make size (target_size, target_size)
    - Adjust dimensions divisible by 14

    Args:
        mask_tensor_list (list): List of mask tensors with shape (H, W)
        mode (str, optional): "crop" or "pad"
        target_size (int, optional): Target size for width/height (default=518)

    Returns:
        torch.Tensor: Batched mask tensor with shape (N, 1, H, W)
    """
    if len(mask_tensor_list) == 0:
        raise ValueError("At least 1 mask is required")

    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    masks = []
    shapes = set()

    for mask in mask_tensor_list:
        assert mask.ndim == 2, f"Mask must have 2 dimensions (H, W), got {mask.shape}"

        mask = mask.float().unsqueeze(0)  # Add channel dimension (1, H, W)

        _, height, width = mask.shape

        if mode == "pad":
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14
        else:  # mode == "crop"
            new_width = target_size
            new_height = round(height * (new_width / width) / 14) * 14

        # Resize
        mask = TF.resize(
            mask, [new_height, new_width], interpolation=TF.InterpolationMode.NEAREST
        )

        # Center crop if needed
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            mask = mask[:, start_y : start_y + target_size, :]

        # Pad if needed
        if mode == "pad":
            h_padding = target_size - mask.shape[1]
            w_padding = target_size - mask.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left
                mask = F.pad(
                    mask,
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode="constant",
                    value=0.0,
                )

        masks.append(mask)
        shapes.add((mask.shape[1], mask.shape[2]))

    # Pad to same shape if needed
    if len(shapes) > 1:
        print(f"Warning: Found masks with different shapes: {shapes}")
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        padded_masks = []
        for mask in masks:
            h_padding = max_height - mask.shape[1]
            w_padding = max_width - mask.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left
                mask = F.pad(
                    mask,
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode="constant",
                    value=0.0,
                )
            padded_masks.append(mask)
        masks = padded_masks

    masks = torch.stack(masks)  # Stack into (N, 1, H, W)

    return masks

    # return {
    #     "xyz": deformed_xyz,
    #     "rot4d": deformed_gs_rotq,
    #     "scales": gs_scales,
    # }


def preprocess_masks_and_transforms(mask_tensor_list, mode="crop", target_size=518):
    """
    Returns:
      masks:      (N,1,Hc,Wc) processed masks
      transforms: list of dicts with keys
                  { 'scale_x','scale_y',
                    'offset_x','offset_y',
                    'mode' }
    """
    masks = []
    transforms = []
    for mask in mask_tensor_list:
        H0, W0 = mask.shape
        mask = mask.float().unsqueeze(0)  # (1,H0,W0)

        # 1) decide new intermediate size H1,W1
        if mode == "pad":
            if W0 >= H0:
                W1 = target_size
                H1 = round(H0 * (W1 / W0) / 14) * 14
            else:
                H1 = target_size
                W1 = round(W0 * (H1 / H0) / 14) * 14
        else:  # crop
            W1 = target_size
            H1 = round(H0 * (W1 / W0) / 14) * 14

        # record scale
        sx = W1 / W0
        sy = H1 / H0

        # 2) resize
        mask_resized = TF.resize(
            mask, [H1, W1], interpolation=TF.InterpolationMode.NEAREST
        )

        # 3) crop or pad
        if mode == "crop":
            # center-crop to target_size
            dy = max(0, H1 - target_size)
            dx = max(0, W1 - target_size)
            top = dy // 2
            left = dx // 2
            mask_proc = mask_resized[
                :, top : top + target_size, left : left + target_size
            ]
            off_x, off_y = left, top

        else:  # pad
            dh = target_size - H1
            dw = target_size - W1
            top = dh // 2
            bottom = dh - top
            left = dw // 2
            right = dw - left
            mask_proc = F.pad(mask_resized, (left, right, top, bottom), value=0.0)
            off_x, off_y = left, top

        masks.append(mask_proc)
        transforms.append(
            {
                "scale_x": sx,
                "scale_y": sy,
                "offset_x": off_x,
                "offset_y": off_y,
                "mode": mode,
            }
        )

    masks = torch.stack(masks, dim=0)  # (N,1,Ht,Wt)
    return masks, transforms


def transform_indices(
    pixel_coords: torch.LongTensor,  # (N,2), may contain (-1,-1) for invalids
    transform,
    target_size_x=518,
    target_size_y=294,
) -> torch.LongTensor:
    sx, sy = transform["scale_x"], transform["scale_y"]
    ox, oy = transform["offset_x"], transform["offset_y"]
    device = pixel_coords.device
    N = pixel_coords.shape[0]

    out = torch.full((N, 2), -1, dtype=torch.long, device=device)

    # mask of valid points
    valid = (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 1] >= 0)
    if valid.sum() == 0:
        return out

    pts = pixel_coords[valid].float()  # (M,2)
    xs = (pts[:, 0] * sx + ox).round().long().clamp(0, target_size_x - 1)
    ys = (pts[:, 1] * sy + oy).round().long().clamp(0, target_size_y - 1)

    out[valid, 0] = xs
    out[valid, 1] = ys
    return out


###########


def compute_vertex_normals(
    vertices: torch.Tensor, faces: torch.LongTensor
) -> torch.Tensor:
    """
    vertices: (N,3) in world coords
    faces:    (F,3) indices into vertices
    returns:  (N,3) unit normals in world coords
    """
    device = vertices.device
    N = vertices.shape[0]

    # 1) face normals
    tris = vertices[faces]  # (F,3,3)
    v0, v1, v2 = tris.unbind(1)  # each (F,3)
    fn = torch.cross(v1 - v0, v2 - v0, dim=-1)  # (F,3)
    fn = fn / fn.norm(dim=1, keepdim=True)  # normalize

    # 2) accumulate per-vertex
    vert_normals = torch.zeros((N, 3), device=device)
    vert_counts = torch.zeros((N, 1), device=device)
    for i in range(3):
        idx = faces[:, i]  # (F,)
        vert_normals.index_add_(0, idx, fn)  # sum face normals
        vert_counts.index_add_(
            0, idx, torch.ones_like(idx, dtype=torch.float, device=device).unsqueeze(1)
        )

    vert_normals = vert_normals / vert_counts  # average
    vert_normals = vert_normals / vert_normals.norm(dim=1, keepdim=True)
    return vert_normals


def project_vertices_to_pixels(
    vertices: torch.Tensor,  # (N,3) world-space
    faces: torch.LongTensor,  # (F,3) mesh triangles
    c2w: torch.Tensor,  # (4,4) camera-to-world
    fovx,  # scalar float or 0-dim tensor, radians
    fovy,  # scalar float or 0-dim tensor, radians
    height: int,
    width: int,
):
    device = vertices.device
    N = vertices.shape[0]

    # --- 1) extrinsics: invert c2w → world→camera  ---
    R_wc = c2w[:3, :3]  # cam→world
    t_wc = c2w[:3, 3]  # cam pos in world
    R_cw = R_wc.T  # world→cam
    t_cw = -R_cw @ t_wc  # world→cam translation

    # --- 2) transform points to camera space ---
    X_cam = (vertices @ R_cw.T) + t_cw  # (N,3)
    zs = X_cam[:, 2].clamp(min=1e-6)

    # --- 3) compute & transform normals to camera space ---
    normals_world = compute_vertex_normals(vertices, faces)  # (N,3)
    normals_cam = normals_world @ R_cw.T  # (N,3)

    # --- 4) back-face cull in camera space ---
    # view_dir in camera coords is just the vector from camera origin to point:
    # view_dir = X_cam / zs.unsqueeze(1)       # normalize by Z gives a direction
    # front_facing = (normals_cam * view_dir).sum(dim=1) > 0
    view_dir = -X_cam / zs.unsqueeze(1)
    front_facing = (normals_cam * view_dir).sum(dim=1) > 0

    # --- 5) intrinsics from FOVs ---
    fx = (width / 2) / math.tan(float(fovx) / 2)
    fy = (height / 2) / math.tan(float(fovy) / 2)
    cx = (width - 1) / 2
    cy = (height - 1) / 2

    # --- 6) project to pixel coords ---
    us = (X_cam[:, 0] / zs) * fx + cx
    vs = (X_cam[:, 1] / zs) * fy + cy

    x_pix = us.round().long().clamp(0, width - 1)
    y_pix = vs.round().long().clamp(0, height - 1)

    # --- 7) z‐buffer, skipping back‐facing points ---
    depth_buffer = torch.full((height, width), float("inf"), device=device)
    idx_buffer = torch.full((height, width), -1, dtype=torch.long, device=device)

    for i in range(N):
        if not front_facing[i]:
            continue
        x, y, z = x_pix[i].item(), y_pix[i].item(), zs[i].item()
        if z < depth_buffer[y, x]:
            depth_buffer[y, x] = z
            idx_buffer[y, x] = i

    # --- 8) invert the pixel→point map to per-point coords & mask ---
    pixel_coords = torch.full((N, 2), -1, dtype=torch.long, device=device)
    valid_mask = torch.zeros(N, dtype=torch.bool, device=device)

    ys, xs = torch.nonzero(idx_buffer >= 0, as_tuple=True)
    for y, x in zip(ys.tolist(), xs.tolist()):
        pid = idx_buffer[y, x].item()
        pixel_coords[pid] = torch.tensor([x, y], device=device)
        valid_mask[pid] = True

    return pixel_coords, valid_mask, normals_cam, zs, view_dir


def project_vertices_to_pixels_without_faces(
    vertices: torch.Tensor,  # (N, 3) world-space points
    c2w: torch.Tensor,  # (4, 4) camera-to-world
    fovx,  # scalar float or 0-dim tensor, radians
    fovy,  # scalar float or 0-dim tensor, radians
    height: int,
    width: int,
):
    """
    Projects 3D points to pixel coordinates with z-buffering, without using mesh faces.

    Args:
        vertices: (N, 3) tensor of world-space points
        c2w:      (4, 4) camera-to-world transformation
        fovx:     horizontal field of view in radians
        fovy:     vertical field of view in radians
        height:   image height in pixels
        width:    image width in pixels

    Returns:
        pixel_coords: (N, 2) tensor of integer pixel coordinates (x, y), or (-1, -1) if not visible
        valid_mask:   (N,) boolean tensor indicating which points were rasterized
    """
    device = vertices.device
    N = vertices.shape[0]

    # 1) Extrinsics: world -> camera
    R_wc = c2w[:3, :3]  # cam -> world
    t_wc = c2w[:3, 3]  # cam position in world
    R_cw = R_wc.T  # world -> cam
    t_cw = -R_cw @ t_wc  # world -> cam translation

    # 2) Transform points to camera space
    X_cam = vertices @ R_cw.T + t_cw  # (N, 3)
    zs = X_cam[:, 2]  # (N,)

    # Only keep points in front of camera
    in_front = zs > 0

    # 3) Intrinsics from FOVs
    fx = (width / 2) / math.tan(float(fovx) / 2)
    fy = (height / 2) / math.tan(float(fovy) / 2)
    cx = (width - 1) / 2
    cy = (height - 1) / 2

    # 4) Project to pixel coordinates
    us = (X_cam[:, 0] / zs) * fx + cx
    vs = (X_cam[:, 1] / zs) * fy + cy

    x_pix = us.round().long().clamp(0, width - 1)
    y_pix = vs.round().long().clamp(0, height - 1)

    # 5) Z-buffer rasterization
    depth_buffer = torch.full((height, width), float("inf"), device=device)
    idx_buffer = torch.full((height, width), -1, dtype=torch.long, device=device)

    for i in range(N):
        if not in_front[i].item():
            continue
        x = x_pix[i].item()
        y = y_pix[i].item()
        z = zs[i].item()
        if z < depth_buffer[y, x]:
            depth_buffer[y, x] = z
            idx_buffer[y, x] = i

    # 6) Invert pixel->point map to per-point coords & mask
    pixel_coords = torch.full((N, 2), -1, dtype=torch.long, device=device)
    valid_mask = torch.zeros(N, dtype=torch.bool, device=device)

    ys, xs = torch.nonzero(idx_buffer >= 0, as_tuple=True)
    for y, x in zip(ys.tolist(), xs.tolist()):
        pid = idx_buffer[y, x].item()
        pixel_coords[pid] = torch.tensor([x, y], device=device)
        valid_mask[pid] = True

    return pixel_coords, valid_mask


from hugs.utils.rotations import matrix_to_axis_angle, axis_angle_to_matrix


def estimate_global_rotation(gaussians, smpl_vertices, knn_indices):
    # Step 1: average SMPL neighbors
    V_corr = smpl_vertices[knn_indices]  # (8000, 6, 3)
    V_corr_mean = V_corr.mean(dim=1)  # (8000, 3)

    # Step 2: center both sets
    G_mean = gaussians.mean(dim=0, keepdim=True)
    V_mean = V_corr_mean.mean(dim=0, keepdim=True)
    G_centered = gaussians - G_mean
    V_centered = V_corr_mean - V_mean

    # Step 3: Kabsch algorithm
    H = V_centered.T @ G_centered  # (3, 3)
    U, S, Vt = torch.linalg.svd(H)
    R = Vt.T @ U.T

    # Reflection correction
    if torch.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    return R


def get_transformed_axis_angle(gaussians, smpl_vertices, knn_indices, axis_angle_1):
    R = estimate_global_rotation(gaussians, smpl_vertices, knn_indices[:, :2])
    R_axis_angle_1 = axis_angle_to_matrix(axis_angle_1)
    R_axis_angle_2 = R @ R_axis_angle_1
    axis_angle_2 = matrix_to_axis_angle(R_axis_angle_2)
    return axis_angle_2


def find_smpl_to_gaussian_correspondence(data):
    c2w = data["c2w"].to(device)
    # R = c2w[:3, :3]    # 3x3
    # t = c2w[:3, 3]     # 3x1
    human_in_hugs_world, human_faces = get_deformed_gs_xyz_with_faces(
        None,
        None,
        None,
        None,
        data["betas"],
        data["body_pose"],
        data["global_orient"],
        data["smpl_scale"],
        data["transl"],
    )

    locs, filters, cam_normals, zs, view_dir = project_vertices_to_pixels(
        human_in_hugs_world,
        human_faces,
        data["c2w"],
        data["fovx"],
        data["fovy"],
        data["image_height"],
        data["image_width"],
    )
    downsampled_mask, transforms = preprocess_masks_and_transforms([data["mask"]])
    new_coords_list = transform_indices(locs, transforms[0])
    return (
        new_coords_list,
        filters,
        human_in_hugs_world,
        human_faces,
        cam_normals,
        downsampled_mask,
        zs,
        view_dir,
    )


def find_gaussians_rendered_pixel_locs(
    gaussians, c2w, fovx, fovy, image_height, image_width, mask
):
    locs, filters = project_vertices_to_pixels_without_faces(
        gaussians, c2w, fovx, fovy, image_height, image_width
    )
    _, transforms = preprocess_masks_and_transforms([mask])
    new_coords_list = transform_indices(locs, transforms[0])
    return new_coords_list, filters


# def get_deformed_human_using_image_correspondences(xyz, smpl_vertices, gauss_pixels, vert_pixels, visibility, betas, body_pose, global_orient, smpl_scale=None, transl=None):
#     def get_vitruvian_verts():
#         vitruvian_pose = torch.zeros(69, dtype=smpl.dtype, device=device)
#         vitruvian_pose[2] = 1.0
#         vitruvian_pose[5] = -1.0
#         smpl_output = smpl(body_pose=vitruvian_pose[None], betas=betas[None], disable_posedirs=False)
#         vitruvian_verts = smpl_output.vertices[0]
#         A_t2vitruvian = smpl_output.A[0].detach()
#         T_t2vitruvian = smpl_output.T[0].detach()
#         inv_T_t2vitruvian = torch.inverse(T_t2vitruvian)
#         inv_A_t2vitruvian = torch.inverse(A_t2vitruvian)
#         canonical_offsets = smpl_output.shape_offsets + smpl_output.pose_offsets
#         canonical_offsets = canonical_offsets[0].detach()
#         vitruvian_verts = vitruvian_verts.detach()
#         return vitruvian_verts.detach(), inv_A_t2vitruvian


#     vitruvian_verts, inv_A_t2vitruvian = get_vitruvian_verts()

#     _, gt_lbs_weights, knn_idx, gaussians_masks = smpl_lbsweight_top_k_2d_image_plane(
#         lbs_weights=smpl.lbs_weights,
#         gauss_pixels=gauss_pixels,
#         vert_pixels=vert_pixels,
#         visibility=visibility,
#     )
#     gt_lbs_weights = gt_lbs_weights.squeeze(0)
#     if abs(gt_lbs_weights.sum(-1).mean().item() - 1) < 1e-7:
#         pass
#     else:
#         print(f"GT LBS weights should sum to 1, but it is: {gt_lbs_weights.sum(-1).mean().item()}")


#     #########
#     global_orient_transformed = get_transformed_axis_angle(xyz, smpl_vertices, knn_idx, global_orient)
#     smpl_output = smpl(
#         betas=betas.unsqueeze(0),
#         body_pose=body_pose.unsqueeze(0),
#         global_orient=global_orient_transformed.unsqueeze(0),
#         disable_posedirs=False,
#         return_full_pose=True,
#         transl=torch.tensor([0.13, 0.05, 0.63], device=device).unsqueeze(0)
#     )
#     A_t2pose = smpl_output.A[0]
#     # A_vitruvian2pose = A_t2pose @ inv_A_t2vitruvian
#     # A_pose2vitruvian = torch.inverse(A_vitruvian2pose)

#     A_pose2vitruvian = torch.inverse(A_t2pose)
#     #########


#     # canonical_xyz, _, lbs_T, _, _ = lbs_extra(
#     #     A_pose2vitruvian[None].contiguous(),
#     #     xyz[gaussians_masks][None],
#     #     posedirs,
#     #     gt_lbs_weights[gaussians_masks],
#     #     smpl_output.full_pose,
#     #     disable_posedirs=disable_posedirs,
#     #     pose2rot=True
#     # )

#     canonical_xyz, _, lbs_T, _, _ = lbs_extra(
#         A_pose2vitruvian[None].contiguous(),
#         smpl_output.vertices[0][None],
#         posedirs,
#         smpl.lbs_weights,
#         smpl_output.full_pose,
#         disable_posedirs=disable_posedirs,
#         pose2rot=True
#     )

#     return canonical_xyz, smpl_output.vertices[0], knn_idx


def get_deform_from_T_to_pose(
    human_smpl_points_at_T_pose,
    human_smpl_scales,
    human_smpl_rots,
    smpl_visibility,
    betas,
    body_pose,
    global_orient,
    smpl_scale,
    smpl_transl,
):
    smpl_output = smpl(
        betas=betas.unsqueeze(0),
        body_pose=body_pose.unsqueeze(0),
        global_orient=global_orient.unsqueeze(0),
        disable_posedirs=False,
        return_full_pose=True,
    )
    A_t2pose = smpl_output.A[0]

    deformed_xyz, _, lbs_T, _, _ = lbs_extra(
        A_t2pose[None].contiguous(),
        human_smpl_points_at_T_pose[None],
        posedirs,
        smpl.lbs_weights,
        smpl_output.full_pose,
        disable_posedirs=disable_posedirs,
        pose2rot=True,
    )
    lbs_T = lbs_T.squeeze(0)

    if human_smpl_scales is not None and smpl_scale is not None:
        deformed_xyz = deformed_xyz * smpl_scale.unsqueeze(0)
        human_smpl_scales_out = human_smpl_scales + smpl_scale.log()

    # if human_smpl_rots is not None:
    #     gs_rotmat = quaternion_to_matrix(human_smpl_rots)
    #     deformed_gs_rotmat = lbs_T[smpl_visibility, :3, :3] @ gs_rotmat
    #     deformed_gs_rotq = matrix_to_quaternion(deformed_gs_rotmat)
    if human_smpl_rots is not None:
        # human_smpl_rots = human_smpl_rots[smpl_visibility]
        raw_norm = human_smpl_rots.norm(dim=-1, keepdim=True)  # (..., 1)
        normed_rots = torch.nn.functional.normalize(human_smpl_rots, dim=-1)  # (..., 4)

        ######
        gs_rotmat = quaternion_to_matrix(normed_rots)  # (..., 3, 3)
        deformed_gs_rotmat = lbs_T[smpl_visibility, :3, :3] @ gs_rotmat  # (..., 3, 3)
        deformed_gs_rotq = matrix_to_quaternion(deformed_gs_rotmat)  # (..., 4)
        ######

        deformed_gs_rotq = torch.nn.functional.normalize(deformed_gs_rotq, dim=-1)
        deformed_gs_rotq = deformed_gs_rotq * raw_norm

    if smpl_transl is not None:
        deformed_xyz = deformed_xyz + smpl_transl.unsqueeze(0)

    return deformed_xyz, human_smpl_scales_out, deformed_gs_rotq, lbs_T


def get_deformed_human_using_image_correspondences(
    human_points,
    human_scales,
    human_rots,
    smpl_vertices,
    gauss_pixels,
    vert_pixels,
    visibility,
    betas,
    body_pose,
    global_orient,
):
    _, gt_lbs_weights, knn_idx, gaussians_masks = smpl_lbsweight_top_k_2d_image_plane(
        lbs_weights=smpl.lbs_weights,
        gauss_pixels=gauss_pixels,
        vert_pixels=vert_pixels,
        visibility=visibility,
    )
    gt_lbs_weights = gt_lbs_weights.squeeze(0)
    if abs(gt_lbs_weights.sum(-1).mean().item() - 1) < 1e-7:
        pass
    else:
        print(
            f"GT LBS weights should sum to 1, but it is: {gt_lbs_weights.sum(-1).mean().item()}"
        )

    global_orient_transformed = get_transformed_axis_angle(
        human_points, smpl_vertices, knn_idx, global_orient
    )
    smpl_output = smpl(
        betas=betas.unsqueeze(0),
        body_pose=body_pose.unsqueeze(0),
        global_orient=global_orient_transformed.unsqueeze(0),
        disable_posedirs=False,
        return_full_pose=True,
    )
    A_t2pose = smpl_output.A[0]
    A_pose2t = torch.inverse(A_t2pose)

    canonical_xyz, _, lbs_T, _, _ = lbs_extra(
        A_pose2t[None].contiguous(),
        smpl_output.vertices[0][None],
        posedirs,
        smpl.lbs_weights,
        smpl_output.full_pose,
        disable_posedirs=disable_posedirs,
        pose2rot=True,
    )
    # canonical_xyz, _, lbs_T, _, _ = lbs_extra(
    #     A_pose2t[None].contiguous(),
    #     (human_points*5.26)[None],
    #     posedirs,
    #     gt_lbs_weights,
    #     smpl_output.full_pose,
    #     disable_posedirs=disable_posedirs,
    #     pose2rot=True
    # )
    lbs_T = lbs_T.squeeze(0)

    X_smpl = smpl_output.vertices[0][visibility]
    smpl_scale, smpl_R, smpl_transl = estimate_similarity_transform(
        X_smpl, human_points
    )

    if human_scales is not None:
        human_scales = human_scales - smpl_scale.log()

    # if human_rots is not None:
    #     gs_rotmat = quaternion_to_matrix(human_rots)
    #     # deformed_gs_rotmat = lbs_T[:, :3, :3] @ gs_rotmat
    #     deformed_gs_rotmat = lbs_T[visibility, :3, :3] @ gs_rotmat
    #     deformed_gs_rotq = matrix_to_quaternion(deformed_gs_rotmat)
    if human_rots is not None:
        raw_norm = human_rots.norm(dim=-1, keepdim=True)  # (..., 1)
        normed_rots = torch.nn.functional.normalize(human_rots, dim=-1)  # (..., 4)

        ######
        gs_rotmat = quaternion_to_matrix(normed_rots)  # (..., 3, 3)
        deformed_gs_rotmat = lbs_T[visibility, :3, :3] @ gs_rotmat  # (..., 3, 3)
        deformed_gs_rotq = matrix_to_quaternion(deformed_gs_rotmat)  # (..., 4)
        ######

        deformed_gs_rotq = torch.nn.functional.normalize(deformed_gs_rotq, dim=-1)
        deformed_gs_rotq = deformed_gs_rotq * raw_norm

    faces = torch.from_numpy(smpl.faces.astype("int64")).to(canonical_xyz.device)
    normals = compute_vertex_normals(canonical_xyz.squeeze(0), faces)

    return (
        canonical_xyz,
        smpl_output.vertices[0],
        knn_idx,
        smpl_scale,
        smpl_R,
        smpl_transl,
        global_orient_transformed,
        human_scales,
        deformed_gs_rotq,
        lbs_T,
        normals,
    )


def smpl_lbsweight_top_k_2d_image_plane(
    lbs_weights: torch.Tensor,  # (V,24)
    gauss_pixels: torch.LongTensor,  # (G,2)
    vert_pixels: torch.LongTensor,  # (V,2)
    visibility: torch.BoolTensor,  # (V,)
    K: int = 6,
    weight_std: float = 0.1,
    dist_thresh: float = 4.0,  # new threshold in pixels
):
    """
    Returns:
      dists_2d:  (G,1)
      gauss_lbs: (G,24)
      knn_idx:   (G,K)
      valid_mask: (G,) bool indicating which Gaussians passed threshold
    """
    device = lbs_weights.device
    G = gauss_pixels.shape[0]

    # 1) select only visible vertices
    vis_idx = torch.nonzero(visibility, as_tuple=True)[0]
    if vis_idx.numel() == 0:
        raise RuntimeError("No visible vertices!")
    vp_vis = vert_pixels[vis_idx]
    lw_vis = lbs_weights[vis_idx]

    # 2) all-pairs 2D distance
    d2d = torch.cdist(gauss_pixels.float(), vp_vis.float(), p=2.0)  # (G, V_vis)

    # 3) top-K on that subset
    knn_dists, knn_rel = d2d.topk(K, dim=1, largest=False)  # (G,K)

    # 3.5) create mask: valid if closest distance < threshold
    valid_mask = knn_dists[:, 0] < dist_thresh  # (G,)

    # 4) map back to global vertex IDs
    knn_idx = vis_idx[knn_rel]  # (G,K)

    # 5) gather weights
    neigh_w = lbs_weights[knn_idx]  # (G,K,24)

    # 6) confidence mask
    weight_var = 2.0 * weight_std**2
    diff = torch.abs(neigh_w - neigh_w[:, :1, :])  # (G,K,24)
    conf = torch.exp(-diff.sum(-1) / weight_var)  # (G,K)
    conf = (conf > 0.9).float()

    # 7) distance-based weights
    w = torch.exp(-knn_dists) * conf  # (G,K)
    w = w / (w.sum(-1, keepdim=True) + 1e-8)

    # 8) blend
    gauss_lbs = (w.unsqueeze(-1) * neigh_w).sum(dim=1)  # (G,24)

    # 9) average dist
    dists_2d = (w * knn_dists).sum(dim=1, keepdim=True)  # (G,1)

    return dists_2d, gauss_lbs, knn_idx, valid_mask


def transform_pointcloud_world1_to_world2(points_world1, c2w1, c2w2):
    """
    Args:
        points_world1: (N, 3) torch.Tensor in world1 coordinates
        c2w1: (4, 4) torch.Tensor, camera-to-world for world1
        c2w2: (4, 4) torch.Tensor, camera-to-world for world2
    Returns:
        points_world2: (N, 3) torch.Tensor, point cloud in world2 coordinates
    """
    assert points_world1.shape[1] == 3, "Input must be (N, 3)"

    # Convert to homogeneous coords (N, 4)
    ones = torch.ones_like(points_world1[:, :1])
    points_homo = torch.cat([points_world1, ones], dim=1)  # (N, 4)

    # World1 → camera1 → camera2 → world2:
    # Compute: world1 → camera1
    w2c1 = torch.inverse(c2w1)
    # Compute: camera2 → world2
    c2w2 = c2w2

    # Combined transform: world1 → world2
    w1_to_w2 = c2w2 @ w2c1  # (4, 4)

    # Apply transformation
    points_world2_homo = (w1_to_w2 @ points_homo.T).T  # (N, 4)
    points_world2 = points_world2_homo[:, :3] / points_world2_homo[:, 3:]

    return points_world2


def estimate_similarity_transform(X_smpl, X_gauss):
    """
    Estimate similarity transform (scale, rotation, translation) from X_smpl to X_gauss.
    Args:
        X_smpl: (N, 3) SMPL vertices
        X_gauss: (N, 3) Corresponding Gaussian centers
    Returns:
        s: scalar scale
        R: (3, 3) rotation matrix
        t: (3,) translation vector
    """
    assert X_smpl.shape == X_gauss.shape
    N = X_smpl.shape[0]

    mu_smpl = X_smpl.mean(dim=0)
    mu_gauss = X_gauss.mean(dim=0)

    Xs = X_smpl - mu_smpl
    Xg = X_gauss - mu_gauss

    # Compute rotation
    cov = Xs.T @ Xg / N
    U, S, Vt = torch.linalg.svd(cov)
    R = Vt.T @ U.T
    # Correct reflection
    if torch.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Compute scale
    var_Xs = (Xs**2).sum() / N
    s = (S.sum()) / var_Xs

    # Compute translation
    t = mu_gauss - s * R @ mu_smpl

    return s, R, t


def transform_rotations_through_lbs(
    quat: torch.Tensor, lbs_rot: torch.Tensor
) -> torch.Tensor:
    """
    Applies LBS transformation to a batch of quaternions.

    Args:
        quat: (N, 4) unnormalized quaternions
        lbs_rot: (N, 3, 3) rotation matrices from LBS
        raw_norm: (N, 1) optional precomputed norm of `quat`. If not given, it will be computed.

    Returns:
        (N, 4) transformed quaternions (same norm as input)
    """
    quat_normalized = torch.nn.functional.normalize(quat, dim=-1)  # (N, 4)
    rotmat = quaternion_to_matrix(quat_normalized)  # (N, 3, 3)
    deformed_rotmat = torch.matmul(lbs_rot, rotmat)  # (N, 3, 3)
    deformed_quat = matrix_to_quaternion(deformed_rotmat)  # (N, 4)
    deformed_quat = torch.nn.functional.normalize(
        deformed_quat, dim=-1
    )  # Ensure unit quaternion
    return deformed_quat


def match_smpl_to_gaussians_batched(
    smpl_locs,
    gauss_pixels,
    smpl_visibility,
    threshold=5.0,
    batch_size=2000,
    device="cuda",
):
    """
    Matches each visible SMPL vertex to the closest Gaussian with a 1-to-1 constraint,
    using memory-efficient batch processing.

    Args:
        smpl_locs: (V, 2) SMPL vertex image coords
        gauss_pixels: (G, 2) Gaussian image coords
        smpl_visibility: (V,) boolean tensor of visible vertices
        threshold: max pixel distance to accept a match
        batch_size: number of SMPL vertices to process at once
    """
    V = smpl_locs.shape[0]
    G = gauss_pixels.shape[0]
    device = smpl_locs.device

    visible_indices = smpl_visibility.nonzero(as_tuple=True)[0]
    visible_smpl_locs = smpl_locs[visible_indices]  # (V_visible, 2)
    V_visible = visible_smpl_locs.shape[0]

    all_matches = []

    # Process in batches of visible SMPL vertices
    for i in range(0, V_visible, batch_size):
        batch_smpl = visible_smpl_locs[i : i + batch_size]
        batch_indices = torch.arange(i, min(i + batch_size, V_visible), device=device)

        # Compute (B, G) distances
        dists = torch.cdist(batch_smpl.float(), gauss_pixels.float(), p=2)

        # Filter matches under threshold
        valid = dists < threshold
        valid_pairs = valid.nonzero(as_tuple=False)
        valid_dists = dists[valid]

        v_batch_ids = batch_indices[valid_pairs[:, 0]]
        g_ids = valid_pairs[:, 1]

        all_matches.append(torch.stack([v_batch_ids, g_ids, valid_dists], dim=1))

    if not all_matches:
        return torch.full((V,), -1, dtype=torch.long, device=device)  # no match at all

    # Concatenate all matches and sort by distance
    matches = torch.cat(all_matches, dim=0)  # (M, 3): [v_idx_in_visible, g_idx, dist]
    matches = matches[matches[:, 2].argsort()]  # sort by dist

    # Greedy unique matching (vectorized version with masks)
    assigned_vertices = torch.zeros(V_visible, dtype=torch.bool, device=device)
    assigned_gaussians = torch.zeros(G, dtype=torch.bool, device=device)
    final_assignments = torch.full((V_visible,), -1, dtype=torch.long, device=device)

    for i in range(matches.shape[0]):
        v, g = matches[i, 0].long(), matches[i, 1].long()
        if not assigned_vertices[v] and not assigned_gaussians[g]:
            assigned_vertices[v] = True
            assigned_gaussians[g] = True
            final_assignments[v] = g

    # Map back to full SMPL vertex set
    matched_gaussian_indices = torch.full((V,), -1, dtype=torch.long, device=device)
    matched_gaussian_indices[visible_indices] = final_assignments

    return matched_gaussian_indices


import torch
import numpy as np
import faiss


def match_smpl_to_gaussians_fast(
    smpl_locs: torch.Tensor,
    gauss_pixels: torch.Tensor,
    smpl_visibility: torch.Tensor,
    threshold: float = 5.0,
) -> torch.Tensor:
    """
    Match each visible SMPL vertex to its nearest Gaussian (in pixel space)
    using FAISS on the GPU. Returns a (V,) tensor of indices into `gauss_pixels`,
    or -1 if no Gaussian is within `threshold` pixels.

    Args:
        smpl_locs:        (V,2) float Tensor of (x,y) pixel coords
        gauss_pixels:     (G,2) float Tensor of (x,y) pixel coords
        smpl_visibility:  (V,)   bool Tensor, True for vertices to match
        threshold:        max allowed Euclidean distance in pixels

    Returns:
        matched_indices:  (V,) long Tensor: matched_indices[v] = g or -1
    """
    V = smpl_locs.shape[0]
    device = smpl_locs.device
    # Initialize all to -1
    matched_indices = torch.full((V,), -1, dtype=torch.long, device=device)

    # Which SMPL vertices are visible?
    vis_idx = torch.nonzero(smpl_visibility, as_tuple=False).view(-1)
    if vis_idx.numel() == 0:
        return matched_indices

    # Move data to CPU/numpy for FAISS
    # FAISS expects float32 numpy arrays
    gauss_np = gauss_pixels.cpu().numpy().astype("float32")  # (G,2)
    smpl_np = smpl_locs[vis_idx].cpu().numpy().astype("float32")  # (Vvis,2)

    # Build a GPU-backed L2 index
    res = faiss.StandardGpuResources()  # single GPU
    index_cpu = faiss.IndexFlatL2(2)  # 2 dims → L2
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
    gpu_index.add(gauss_np)  # now index has G vectors

    # Query: for each visible SMPL vertex, get its nearest Gaussian
    D, I = gpu_index.search(smpl_np, 1)  # both are (Vvis,1)
    D = D[:, 0]  # squared L2 distances
    I = I[:, 0].astype(np.int64)  # indices of nearest Gaussians

    # Enforce distance threshold (FAISS returns squared distances)
    thr_sq = threshold * threshold
    valid = D < thr_sq  # (Vvis,)

    # Scatter valid matches back into the full V-length result
    valid_vis_idx = vis_idx[valid]  # which SMPL verts matched
    matched_gauss = torch.from_numpy(I[valid]).to(device)
    matched_indices[valid_vis_idx] = matched_gauss

    return matched_indices


def match_smpl_to_gaussians_fast_one2one(
    smpl_locs: torch.Tensor,
    gauss_pixels: torch.Tensor,
    smpl_visibility: torch.Tensor,
    threshold: float = 3.0,
    knn: int = 8,
) -> torch.Tensor:
    """
    1-to-1 match each visible SMPL vertex to a Gaussian, globally greedy by distance.

    Args:
        smpl_locs:       (V,2) float Tensor of (x,y) pixel coords
        gauss_pixels:    (G,2) float Tensor of (x,y) pixel coords
        smpl_visibility: (V,)   bool Tensor
        threshold:       max allowed Euclidean distance in pixels
        knn:             number of nearest neighbours to pull from FAISS per vertex

    Returns:
        matched_indices: (V,) long Tensor: matched_indices[v] = g or -1
    """
    V = smpl_locs.shape[0]
    device = smpl_locs.device
    matched = torch.full((V,), -1, dtype=torch.long, device=device)

    # 1) Which verts to process?
    vis_idx = torch.nonzero(smpl_visibility, as_tuple=False).view(-1)
    if vis_idx.numel() == 0:
        return matched

    # 2) Prepare numpy inputs for FAISS
    gauss_np = gauss_pixels.cpu().numpy().astype("float32")  # (G,2)
    smpl_np = smpl_locs[vis_idx].cpu().numpy().astype("float32")  # (Vvis,2)

    # 3) Build & query FAISS
    res = faiss.StandardGpuResources()
    index_cpu = faiss.IndexFlatL2(2)
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
    gpu_index.add(gauss_np)
    # D is squared‐L2 distances, I is indices, both (Vvis, knn)
    D, I = gpu_index.search(smpl_np, knn)

    thr_sq = threshold * threshold

    # 4) Flatten only those within threshold
    #    entries: (vert_global_idx, gauss_idx, dist_sq)
    candidates = []
    for vi in range(D.shape[0]):
        global_v = int(vis_idx[vi])
        for k in range(knn):
            if D[vi, k] < thr_sq:
                candidates.append((global_v, int(I[vi, k]), float(D[vi, k])))

    if not candidates:
        return matched

    # 5) Sort all candidate pairs by increasing distance
    candidates.sort(key=lambda x: x[2])

    # 6) Greedy one-to-one assignment
    used_v = set()
    used_g = set()
    for v, g, _ in candidates:
        if v not in used_v and g not in used_g:
            matched[v] = g
            used_v.add(v)
            used_g.add(g)

    return matched


def interpolate_laplacian(
        verts: torch.Tensor,
        faces: torch.LongTensor,
        offsets: torch.Tensor,
        mask: torch.BoolTensor,
        num_iters: int = 200,
        lam: float = 0.5,
    ) -> torch.Tensor:
        """
        Harmonic extension via Jacobi-style Laplacian smoothing.
        Repeatedly replace each unknown vertex's value with the average of its neighbors,
        then re-clamp the known rim offsets each iteration.

        Args:
            verts:    (V,3) vertex positions (unused here but often kept for reference)
            faces:    (F,3) triangle indices
            offsets:  (V,)   initial offset values (zeros off-rim)
            mask:     (V,)   True for known rim vertices
            num_iters:     number of smoothing passes
            lam:           relaxation factor (0 < lam <= 1)

        Returns:
            out:      (V,)   completed offset field
        """
        # build adjacency list
        V = verts.size(0)
        # 1) Build sparse adjacency A (unweighted 0/1 edges)
        #    each undirected edge i–j becomes two entries (i,j) & (j,i)
        idx_i = faces.view(-1)                        # (3F,)
        idx_j = faces[:, [1,2,0]].reshape(-1)          # rotating for (i,j),(j,k),(k,i)
        row = torch.cat([idx_i, idx_j], dim=0)
        col = torch.cat([idx_j, idx_i], dim=0)
        A = torch.sparse_coo_tensor(torch.stack([row, col]), torch.ones(row.shape[0], device=offsets.device), (V, V)).coalesce()
        
        # 2) Degree vector and its reciprocal
        deg = torch.sparse.sum(A, dim=1).to_dense()    # (V,)
        inv_deg = 1.0 / deg.clamp(min=1e-12)           # avoid div0

        # 3) Pre‐clamp the boundary
        u = offsets.clone()
        boundary_vals = offsets[mask]
        
        # 4) Iterative Jacobi
        for _ in range(num_iters):
            # sparse‐dense matmul: sums neighbor values for each vertex
            nbr_sum = torch.sparse.mm(A, u.unsqueeze(1)).squeeze(1)  # (V,)
            u_new = (1 - lam) * u + lam * (inv_deg * nbr_sum)
            # re‐clamp boundary
            u_new[mask] = boundary_vals
            u = u_new
        return u