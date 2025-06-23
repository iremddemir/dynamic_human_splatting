import sys
sys.path.append('.')

import torch
from PIL import Image
from torchvision import transforms
import torchvision
from vggt.models.our_vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from instantsplat.scene import GaussianModel
from instantsplat.scene.gaussian_model_w_act import GaussianModel_With_Act
from instantsplat.scene.cameras import Camera
from instantsplat.gaussian_renderer import render
from instantsplat.arguments import PipelineParams, ArgumentParser
import sys

def split_gaussians_by_mask(features: dict, mask: torch.Tensor, N_SPLITS: int = 2, local_std_scale: float = 0.00001) -> dict:
    """
    Split selected Gaussians into tightly-clustered clones using small anisotropic offsets.

    Args:
        features: Dict of Gaussian tensors ['xyz', 'features_dc', ..., 'rotation']
        mask: Bool tensor of shape (N,)
        N_SPLITS: Number of child Gaussians per selected one
        local_std_scale: Scale multiplier for offset (e.g. 0.2 → 20% of original scale)

    Returns:
        New features dict with appended Gaussians
    """
    device = features["xyz"].device
    selected_xyz = features["xyz"][mask]
    selected_scaling = features["scaling"][mask]
    selected_rotation = features["rotation"][mask]

    M = selected_xyz.shape[0]

    from instantsplat.utils.general_utils import build_rotation

    rot_matrices = build_rotation(selected_rotation)                     # (M, 3, 3)
    rot_matrices = rot_matrices.repeat(N_SPLITS, 1, 1)                   # (N*M, 3, 3)
    local_std = selected_scaling * local_std_scale                      # (M, 3)
    local_std = local_std.repeat(N_SPLITS, 1)                           # (N*M, 3)

    noise = torch.randn(M * N_SPLITS, 3, device=device) * local_std     # (N*M, 3)
    offset = torch.bmm(rot_matrices, noise.unsqueeze(-1)).squeeze(-1)  # (N*M, 3)

    new_xyz = selected_xyz.repeat(N_SPLITS, 1) + offset
    new_scaling = selected_scaling.repeat(N_SPLITS, 1)
    new_rotation = selected_rotation.repeat(N_SPLITS, 1)

    def repeat_feat(x): return x[mask].repeat(N_SPLITS, 1, 1) if x.ndim == 3 else x[mask].repeat(N_SPLITS, 1)

    return {
        "xyz":           torch.cat([features["xyz"], new_xyz], dim=0),
        "features_dc":   torch.cat([features["features_dc"], repeat_feat(features["features_dc"])], dim=0),
        "features_rest": torch.cat([features["features_rest"], repeat_feat(features["features_rest"])], dim=0),
        "opacity":       torch.cat([features["opacity"], repeat_feat(features["opacity"])], dim=0),
        "scaling":       torch.cat([features["scaling"], new_scaling], dim=0),
        "rotation":      torch.cat([features["rotation"], new_rotation], dim=0)
    }

# Setup
device = "cuda"
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
model.eval()
# image_path = "example_img.jpeg"
image_path = "data/neuman/dataset/lab/images/00000.png"

# Load & encode
image = load_and_preprocess_images([image_path]).to(device)[None]  # (1, 3, H, W)
tokens, ps_idx = model.aggregator(image)
pose_enc = model.camera_head(tokens)[-1]
extr, intr = pose_encoding_to_extri_intri(pose_enc, image.shape[-2:])
fov_h, fov_w = pose_enc[..., 7], pose_enc[..., 8]

model.gs_head_feats.load_state_dict(torch.load("last_gs_head_feats_step.pth"))
for param in model.parameters():
    param.requires_grad = False
model.eval()

# Feature prediction
with torch.no_grad():
    feats, _ = model.gs_head_feats(tokens, image, ps_idx)
    # point_map, _ = model.point_head(tokens, image, ps_idx)

    pose_enc = model.camera_head(tokens)[-1]
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, image.shape[-2:])
    fov_h = pose_enc[..., 7]
    fov_w = pose_enc[..., 8]
    depth_map, depth_conf = model.depth_head(tokens, image, ps_idx)
    point_map = unproject_depth_map_to_point_map(depth_map.squeeze(0), 
                                                extrinsic.squeeze(0), 
                                                intrinsic.squeeze(0))[None]
    point_map = torch.tensor(point_map, dtype=torch.float32, device=device)  # [1, B, H, W, 3]


scale = feats[..., 0:3]
rot   = feats[..., 3:7]
sh    = feats[..., 7:10]
op    = feats[..., 10:11]
off   = feats[..., 11:14]
xyz   = point_map[0, 0].reshape(-1, 3) + off[0, 0].reshape(-1, 3)

# Camera
cam_to_world = closed_form_inverse_se3(extr[0, 0][None])[0].detach().cpu().numpy()
R = cam_to_world[:3, :3]
T = -R.T @ cam_to_world[:3, 3]  # Adjust translation
cam = Camera(colmap_id=0, R=R, T=T, FoVx=fov_w[0, 0], FoVy=fov_h[0, 0],
             image=transforms.ToTensor()(Image.open(image_path)), gt_alpha_mask=None, image_name="00000", uid=0)

# Gaussians
features = {
    "xyz": xyz,
    "features_dc": sh[0, 0].reshape(-1, 1, 3),
    "features_rest": torch.zeros((xyz.shape[0], 15, 3), device=device),
    "opacity": op[0, 0].reshape(-1, 1),
    "scaling": scale[0, 0].reshape(-1, 3),
    "rotation": rot[0, 0].reshape(-1, 4)
}


# import ipdb
# ipdb.set_trace()
features = split_gaussians_by_mask(features, op[0, 0].reshape(-1) > 2, N_SPLITS=2)



gaussians = GaussianModel_With_Act(0)
gaussians.init_RT_seq({1.0: [cam]})
gaussians._xyz = features["xyz"]
gaussians._features_dc = features["features_dc"]
gaussians._features_rest = features["features_rest"]
gaussians._opacity = features["opacity"]
gaussians._scaling = features["scaling"]
gaussians._rotation = features["rotation"]

# Render
parser = ArgumentParser(description="Training script parameters")
pipe = PipelineParams(parser)
args = parser.parse_args(sys.argv[1:])
pipe = pipe.extract(args)
render_out = render(cam, gaussians, pipe, torch.tensor([1., 1., 1.], device=device), camera_pose=gaussians.get_RT(0))
rendered_image = render_out["render"]

# Save
torchvision.utils.save_image(rendered_image, "rendered_output.png")