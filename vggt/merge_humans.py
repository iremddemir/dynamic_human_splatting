import sys

sys.path.append(".")
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import gc
import time
import pickle
import random
from random import sample
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from omegaconf import OmegaConf
import wandb
from lpips import LPIPS
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    quaternion_to_matrix,
    matrix_to_quaternion,
)
from torchvision.transforms.functional import to_pil_image

from vggt.utils.load_fn import load_and_preprocess_images
from instantsplat.scene.gaussian_model_w_act import GaussianModel_With_Act
from hugs.trainer.gs_trainer import get_train_dataset
from hugs.cfg.config import cfg as default_cfg
from hugs.losses.utils import ssim, l1_loss, psnr
from hugs.models.modules.triplane import TriPlane
from hugs.models.modules.decoders import MyAppearanceDecoder, MyGeometryDecoder
from utils import *

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import gc
from random import sample

from utils import *

from hugs.models.modules.triplane import TriPlane
from hugs.models.modules.decoders import MyAppearanceDecoder, MyGeometryDecoder


PROJ_DIR = "proj_dir"
PKL_PATH = "/work/courses/3dv/14/data/human_gaussians/per_frame_canonical_human.pkl"  # enter here the path of the dumbed pkl form previous script
PKL_SCENE_PATH = (
    "/work/courses/3dv/14/data/human_gaussians/per_frame_scene_features.pkl"
)
SAVE_DIR = "/work/scratch/jmirlach/MERGING_OUTPUTS"
os.makedirs(SAVE_DIR, exist_ok=True)
device = "cuda"
dtype = torch.float32
cfg_file = f"{PROJ_DIR}3dv_dynamichumansplatting/output/human_scene/neuman/lab/hugs_trimlp/demo-dataset.seq=lab/2025-03-30_19-55-08/config_train.yaml"
cfg_file = OmegaConf.load(cfg_file)
cfg = OmegaConf.merge(default_cfg, cfg_file)

exp_name = f"GAUSSIANS_LEARNED_WITH_OFFSETS"

dataset = get_train_dataset(cfg)


if __name__ == "__main__":

    ############## Load PKL and print details ##############

    with open(PKL_SCENE_PATH, "rb") as f:
        scene_gaussians = pickle.load(f)

    with open(PKL_PATH, "rb") as f:
        all_gaussians = pickle.load(f)
    print(f"Top-level type: {type(all_gaussians)}")
    print(f"Type of gaussian items: {type(all_gaussians[0])}")
    print(f"Loaded {len(all_gaussians)} canonical humans.")
    print("Sample Gaussian structure:")
    for k, v in all_gaussians[0].items():
        print(f"  {k}: type={type(v)}, shape={getattr(v, 'shape', None)}")
    print(all_gaussians[0]["visibility_mask"].sum())
    for k, v in all_gaussians[1].items():
        print(f"  {k}: type={type(v)}, shape={getattr(v, 'shape', None)}")
    print(all_gaussians[1]["visibility_mask"].sum())
    print("\n\n\n")

    ############# Merge Gaussians ################

    def pad_to_full(g, key):
        """
        Given one gaussian‐dict `g`, expand g[key] from shape
        [Ni, …] back to [110210, …] using its visibility_mask.
        """
        mask = g["visibility_mask"].bool()  # [110210]
        tensor = g[key]
        full_shape = (mask.shape[0],) + tensor.shape[1:]
        full = tensor.new_zeros(full_shape)
        full[mask] = tensor
        return full

    def masked_mean(stacked, masks, dim=0):
        # 1) lift the mask to the same rank as 'stacked'
        mask = masks
        while mask.ndim < stacked.ndim:
            mask = mask.unsqueeze(-1)
        # 2) sum
        feature_sum = stacked.sum(dim)  # → [P, d1, d2, …]
        valid_sum = mask.sum(dim)  # → [P, 1,  1,  …]
        # 3) broadcast division
        return feature_sum / valid_sum  # → [P, d1, d2, …]

    def average_quaternions_batched(all_full_rots, masks):
        """
        Batched quaternion averaging without Python loop.
        all_full_rots: [N, P, 4]
        masks:        [N, P]  (bool or 0/1)
        """
        # 1) Align signs
        ref = all_full_rots[0:1]  # [1, P, 4]
        dots = (all_full_rots * ref).sum(-1, keepdim=True)  # [N, P, 1]
        quats_signed = torch.where(dots < 0, -all_full_rots, all_full_rots)
        # 2) Zero-out invisibles
        quats = quats_signed * masks.unsqueeze(-1)
        # 3) Build covariance matrices in one go
        A = torch.einsum("npi,npj->pij", quats, quats)  # [P, 4, 4]
        # 4) Batched eigendecomposition
        eigvals, eigvecs = torch.linalg.eigh(A)  # eigvecs: [P, 4, 4]
        avg = eigvecs[..., -1]  # [P, 4]
        # 5) Normalize
        avg = avg / avg.norm(dim=-1, keepdim=True)
        # 6) Fallback for points unseen in any frame
        avg[masks.sum(dim=0) == 0] = torch.tensor(
            [1, 0, 0, 0], device=avg.device, dtype=avg.dtype
        )
        return avg

    def merge_canonical_gaussians_mean(gaussians_list: List[Dict]) -> Dict:

        masks = torch.stack([g["visibility_mask"].bool() for g in gaussians_list], 0)

        # Average trivial features
        merged = {}
        for key in ["scaling", "opacity", "features_dc", "features_rest"]:
            fulls = torch.stack(
                [pad_to_full(g, key) for g in gaussians_list], 0
            )  # [N,P,...]
            merged[key] = masked_mean(fulls, masks, dim=0)
        full_offsets = torch.stack([g["offset_scales"] for g in gaussians_list], 0)
        offsets_mask = torch.stack(
            [g["found_valid_offset"].bool() for g in gaussians_list], 0
        )
        offset_scales = torch.nan_to_num(
            masked_mean(full_offsets, offsets_mask, dim=0), nan=0.0
        )
        merged["offset_scales"] = interpolate_laplacian(
            gaussians_list[0]["smpl_vertices"],
            gaussians_list[0]["smpl_faces"],
            offset_scales,
            torch.zeros_like(offsets_mask.any(dim=0)).bool(),
        )

        # Average rotations
        all_full_rots = torch.stack(
            [pad_to_full(g, "rotation") for g in gaussians_list], dim=0
        )  # [N,P,4]
        rot_merged = average_quaternions_batched(all_full_rots, masks)

        # Final merged canonical
        merged.update(
            {
                "at_least_one_valid": masks.any(dim=0),
                "xyz": gaussians_list[0][
                    "xyz"
                ],  # All XYZ should be the same, so just takes first
                "rotation": rot_merged,
                "smpl_scales": [g["smpl_scale"] for g in gaussians_list],
                "smpl_transls": [g["smpl_transl"] for g in gaussians_list],
                "smpl_global_orients": [
                    g["smpl_global_orient"] for g in gaussians_list
                ],
                "visibility_masks": [g["visibility_mask"] for g in gaussians_list],
                "normals": [g["normals"] for g in gaussians_list],
            }
        )
        return merged

    def masked_median(
        stacked: torch.Tensor, masks: torch.Tensor, dim: int = 0
    ) -> torch.Tensor:
        """
        Like masked_mean, but returns the (elementwise) median across dim,
        ignoring any entries where mask==0.
        Uses torch.nanmedian, so we first fill invalid entries with NaN.
        """
        # 1) Lift mask to same rank as `stacked`
        m = masks
        while m.ndim < stacked.ndim:
            m = m.unsqueeze(-1)
        # 2) Prepare a float copy, mask out invalid entries to NaN
        x = stacked.clone().float()
        x = x.masked_fill(~m, float("nan"))
        # 3) nan‐median along `dim`
        med, _ = torch.nanmedian(x, dim=dim)
        return med

    def merge_canonical_gaussians_median(gaussians_list: List[Dict]) -> Dict:
        masks = torch.stack(
            [g["visibility_mask"].bool() for g in gaussians_list], 0
        )  # [N, P]

        merged = {}
        # median for all “trivial” features
        for key in ["scaling", "opacity", "features_dc", "features_rest"]:
            fulls = torch.stack(
                [pad_to_full(g, key) for g in gaussians_list], 0
            )  # [N, P, …]
            merged[key] = masked_median(fulls, masks, dim=0)
        full_offsets = torch.stack([g["offset_scales"] for g in gaussians_list], 0)
        offsets_mask = torch.stack(
            [g["found_valid_offset"].bool() for g in gaussians_list], 0
        )
        offset_scales = torch.nan_to_num(
            masked_median(full_offsets, offsets_mask, dim=0), nan=0.0
        )
        merged["offset_scales"] = interpolate_laplacian(
            gaussians_list[0]["smpl_vertices"],
            gaussians_list[0]["smpl_faces"],
            offset_scales,
            torch.zeros_like(offsets_mask.any(dim=0)).bool(),
        )

        # for rotations we’ll reuse your batched mean (quaternion‐median requires a more involved solver)
        all_full_rots = torch.stack(
            [pad_to_full(g, "rotation") for g in gaussians_list], dim=0
        )  # [N, P, 4]
        merged["rotation"] = average_quaternions_batched(all_full_rots, masks)

        # assemble the same metadata as before
        merged.update(
            {
                "at_least_one_valid": masks.any(dim=0),
                "xyz": gaussians_list[0]["xyz"],
                "smpl_scales": [g["smpl_scale"] for g in gaussians_list],
                "smpl_transls": [g["smpl_transl"] for g in gaussians_list],
                "smpl_global_orients": [
                    g["smpl_global_orient"] for g in gaussians_list
                ],
                "visibility_masks": [g["visibility_mask"] for g in gaussians_list],
                "normals": [g["normals"] for g in gaussians_list],
            }
        )
        return merged

    def masked_trimmed_mean(
        stacked: torch.Tensor, masks: torch.Tensor, dim: int = 0, trim_frac: float = 0.1
    ) -> torch.Tensor:
        """
        Compute the trimmed mean along `dim` of `stacked`,
        ignoring entries where mask==0, and trimming the lowest
        and highest `trim_frac` fraction of values (per feature).

        Args:
        stacked: Tensor of shape [N, P, ...]
        masks:   Bool Tensor of shape [N, P]
        dim:     Dimension along which to trim & average (usually 0)
        trim_frac: fraction to trim at each end (0 <= trim_frac < 0.5)
        Returns:
        Tensor of shape [P, ...] with the trimmed means.
        """
        # 1) lift mask to same rank as `stacked`
        m = masks
        while m.ndim < stacked.ndim:
            m = m.unsqueeze(-1)  # now m is broadcastable to stacked

        raw_mean = masked_mean(stacked, masks, dim=dim)

        # 2) mask out invalid entries with NaN
        x = stacked.clone().float().masked_fill(~m, float("nan"))

        # 3) compute per-feature quantile cutoffs
        low = torch.nanquantile(x, trim_frac, dim=dim)  # shape [P, ...]
        high = torch.nanquantile(x, 1 - trim_frac, dim=dim)  # shape [P, ...]

        # 4) broadcast low/high back to x’s rank
        L, H = low, high
        while L.ndim < x.ndim:
            L = L.unsqueeze(0)
            H = H.unsqueeze(0)

        # 5) build trimmed-mask: valid & within [low, high]
        trim_mask = (~x.isnan()) & (x >= L) & (x <= H)

        # 6) compute mean over the trimmed values
        trimmed_mean = masked_mean(x, trim_mask, dim=dim)
        return torch.where(torch.isnan(trimmed_mean), raw_mean, trimmed_mean)

    def merge_canonical_gaussians_trimmed(
        gaussians_list: List[Dict], trim_frac: float = 0.2
    ) -> Dict:
        """
        Produces a single merged canonical from a list of per-frame Gaussians,
        using a trimmed mean for all non-rotation features, and reusing
        average_quaternions_batched for rotation.

        trim_frac is the fraction to discard at each end (e.g. 0.2 → drop lowest 20% and highest 20%).
        """
        # stack visibility: [N, P]
        masks = torch.stack(
            [g["visibility_mask"].bool() for g in gaussians_list], dim=0
        )

        merged = {}
        # trimmed-mean for each “trivial” feature
        for key in ["scaling", "opacity", "features_dc", "features_rest"]:
            fulls = torch.stack(
                [pad_to_full(g, key) for g in gaussians_list], dim=0
            )  # [N, P, …]
            merged[key] = masked_trimmed_mean(fulls, masks, dim=0, trim_frac=trim_frac)

        # rotations: fallback to your SO(3) average
        all_rots = torch.stack(
            [pad_to_full(g, "rotation") for g in gaussians_list], dim=0
        )  # [N, P, 4]
        merged["rotation"] = average_quaternions_batched(all_rots, masks)

        # copy over everything else unchanged
        merged.update(
            {
                "at_least_one_valid": masks.any(dim=0),
                "xyz": gaussians_list[0]["xyz"],
                "smpl_scales": [g["smpl_scale"] for g in gaussians_list],
                "smpl_transls": [g["smpl_transl"] for g in gaussians_list],
                "smpl_global_orients": [
                    g["smpl_global_orient"] for g in gaussians_list
                ],
                "visibility_masks": [g["visibility_mask"] for g in gaussians_list],
            }
        )
        return merged

    def learned_merging(
        gaussians_list: List[Dict], offsets: bool = False, iterations: int = 100
    ):

        N = len(gaussians_list)
        P = int(gaussians_list[0]["visibility_mask"].shape[0])

        class GaussianWeightMerge(nn.Module):
            def __init__(
                self,
                gaussians_list: List[Dict],
                triplane_res=256,
                n_features=32,
                batch_size=8,
            ):
                super().__init__()
                # self.raw_weights = nn.Parameter(torch.zeros(N, P))

                masks = torch.stack(
                    [g["visibility_mask"].bool() for g in gaussians_list], dim=0
                ).float()  # [N, P]
                N, P = masks.shape
                visible_idx = masks.nonzero(as_tuple=False)  # [M,2]
                self.logits_vis = nn.Parameter(torch.zeros(visible_idx.shape[0]))

                self.logits_correction = nn.Parameter(torch.zeros(P))

                self.triplane = TriPlane(
                    n_features, resX=triplane_res, resY=triplane_res, resZ=triplane_res
                ).to(device)
                self.appearance_dec = MyAppearanceDecoder(n_features=n_features * 3).to(
                    device
                )
                # deformation_dec = DeformationDecoder(n_features=n_features*3, disable_posedirs=disable_posedirs).to(device)
                self.geometry_dec = MyGeometryDecoder(n_features=n_features * 3).to(
                    device
                )

                self.register_buffer("visible_idx", visible_idx)
                self.register_buffer("N", torch.tensor(N))
                self.register_buffer("P", torch.tensor(P))
                self.register_buffer("masks", masks.bool())

                self.batch_size = batch_size

            def run_in_chunks(self, fn, x, chunk_size=4096):
                outs = []
                for i in range(0, x.shape[0], chunk_size):
                    outs.append(fn(x[i : i + chunk_size]))
                if isinstance(outs[0], dict):
                    merged = {}
                    for k in outs[0].keys():
                        merged[k] = torch.cat([o[k] for o in outs], dim=0)
                    return merged
                elif isinstance(outs[0], torch.Tensor):
                    return torch.cat(outs, dim=0)

            def forward(self, gaussians_list, indices, output_everything=False):
                bs = len(indices)
                # Reinstantiate the full weight matrix
                N, P = int(self.N), int(self.P)
                logits = torch.full(
                    (N, P), float("-inf"), device=self.logits_vis.device
                )
                logits[self.visible_idx[:, 0], self.visible_idx[:, 1]] = self.logits_vis
                logits = logits[indices]
                W = F.softmax(logits, dim=0)  # shape [N, P]

                merged = {}

                def stacked_full(key, gaussians_list):
                    return torch.stack(
                        [pad_to_full(g, key) for g in gaussians_list], dim=0
                    ).to(
                        device
                    )  # [N, P, ...]

                # 3) weighted mean for all “trivial” features
                for key in ["scaling", "opacity", "features_dc", "features_rest"]:
                    fulls = stacked_full(key, gaussians_list)  # [N,P,...]
                    Wb = W.view(bs, -1, *([1] * (fulls.ndim - 2)))  # [N,P,1...]
                    merged[key] = (fulls * Wb).sum(0)  # [P,...]

                full_offsets = torch.stack(
                    [g["offset_scales"] for g in gaussians_list], 0
                ).to(device)
                Wb = W.view(bs, -1, *([1] * (full_offsets.ndim - 2)))
                merged_offsets = (full_offsets * Wb).sum(0)
                offset_scales = torch.nan_to_num(merged_offsets, nan=0.0)
                merged["offset_scales"] = interpolate_laplacian(
                    gaussians_list[0]["smpl_vertices"].to(device),
                    gaussians_list[0]["smpl_faces"].to(device),
                    offset_scales,
                    torch.zeros_like(offset_scales).bool().to(device),
                )

                # rotations: fallback to your SO(3) average
                all_rots = torch.stack(
                    [pad_to_full(g, "rotation") for g in gaussians_list], dim=0
                ).to(
                    device
                )  # [N, P, 4]
                merged["rotation"] = average_quaternions_batched(
                    all_rots, self.masks[indices]
                )

                tri_feats = self.run_in_chunks(
                    self.triplane, gaussians_list[0]["xyz"].detach().to(device)
                )
                geometry_out = self.run_in_chunks(self.geometry_dec, tri_feats)
                appearance_out = self.run_in_chunks(self.appearance_dec, tri_feats)

                correction_weights = torch.sigmoid(self.logits_correction).unsqueeze(1)

                # print(merged["opacity"].shape, merged["features_dc"].shape, merged["features_rest"].shape)
                merged["opacity"] += (
                    correction_weights * appearance_out["opacity_delta"]
                )
                merged["features_dc"] += (
                    correction_weights.unsqueeze(1) * appearance_out["dc_delta"]
                )
                merged["features_rest"] += (
                    correction_weights.unsqueeze(1) * appearance_out["rest_delta"]
                )

                merged["scaling"] += correction_weights * geometry_out["scales_delta"]
                merged["rotation"] = matrix_to_quaternion(
                    quaternion_to_matrix(merged["rotation"])
                    @ axis_angle_to_matrix(geometry_out["rotations_delta"])
                )
                merged["xyz"] = (
                    gaussians_list[0]["xyz"].to(device)
                    + correction_weights * geometry_out["xyz_delta"]
                )

                # WEIGHTED QUATERNION AVERAGING DOESNT WORK YET
                # rots = stacked_full("rotation")               # [N, P, 4]
                # ref  = rots[0:1]                              # [1, P, 4]
                # # align sign
                # dots = (rots * ref).sum(-1, keepdim=True)     # [N, P, 1]
                # rots_signed = torch.where(dots < 0, -rots, rots)

                # # build weighted covariance A[p] = Σᵢ wᵢₚ · (qᵢₚ ⊗ qᵢₚ)
                # Wq = W.unsqueeze(-1)                           # [N, P, 1]
                # weighted_q = rots_signed * Wq                  # [N, P, 4]
                # # A: [P, 4, 4] via einsum over frames
                # A = torch.einsum('npi,npj->pij', weighted_q, rots_signed)

                # # eigendecompose → principal eigenvector is average quaternion
                # eigvals, eigvecs = torch.linalg.eigh(A)       # eigvecs: [P, 4, 4]
                # q_avg = eigvecs[..., -1]                      # [P, 4]
                # q_avg = q_avg / q_avg.norm(dim=-1, keepdim=True)

                # # fallback identity quaternion for never‐visible points
                # unseen = masks.sum(0) == 0                     # [P]
                # q_avg[unseen] = torch.tensor([1, 0, 0, 0],
                #                             device=q_avg.device,
                #                             dtype=q_avg.dtype)
                # merged["rotation"] = q_avg

                merged.update(
                    {
                        "at_least_one_valid": self.masks[indices].any(dim=0),
                    }
                )
                if output_everything:
                    merged.update(
                        {
                            "smpl_scales": [g["smpl_scale"] for g in gaussians_list],
                            "smpl_transls": [g["smpl_transl"] for g in gaussians_list],
                            "smpl_global_orients": [
                                g["smpl_global_orient"] for g in gaussians_list
                            ],
                            "visibility_masks": [
                                g["visibility_mask"] for g in gaussians_list
                            ],
                            "normals": [g["normals"] for g in gaussians_list],
                        }
                    )
                return merged

        merge_layer = GaussianWeightMerge(gaussians_list).to(device)

        param_groups = {
            "weights_vis": [
                p for n, p in merge_layer.named_parameters() if n == "logits_vis"
            ],
            "weigths_corr": [
                p for n, p in merge_layer.named_parameters() if n == "logits_correction"
            ],
            "networks": [
                p
                for n, p in merge_layer.named_parameters()
                if n not in ("logits_vis", "logits_correction")
            ],
        }

        for name, params in param_groups.items():
            n = sum(p.numel() for p in params)
            print(f"{name:4s} group → {n:,} params")
        num_trainable = sum(
            p.numel() for p in merge_layer.parameters() if p.requires_grad
        )
        print(f"Number of total trainable parameters: {num_trainable}")

        wandb.init(
            entity="jmirlach-eth-z-rich",
            project="3dv_learned_merging",
            name=exp_name if not offsets else f"{exp_name}_offsets",
        )
        optimizer = torch.optim.AdamW(
            [
                {"params": param_groups["networks"], "lr": 2e-2},
                {"params": param_groups["weights_vis"], "lr": 1},
                {"params": param_groups["weigths_corr"], "lr": 1e-1},
            ],
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=iterations, eta_min=1e-4
        )
        tqdm_bar = tqdm(range(iterations), desc="Training", total=iterations)
        for step in tqdm_bar:
            optimizer.zero_grad()
            torch.cuda.empty_cache()

            indices = sample(range(len(gaussians_list)), merge_layer.batch_size)
            gaussians_list_batch = [gaussians_list[i] for i in indices]

            merged_learned = merge_layer(gaussians_list_batch, indices)
            gs_xyz = merged_learned["xyz"].to(device)
            gs_scaling = merged_learned["scaling"].to(device)
            gs_opacity = merged_learned["opacity"].to(device)
            gs_features_dc = merged_learned["features_dc"].to(device)
            gs_features_rest = merged_learned["features_rest"].to(device)
            gs_rotation = merged_learned["rotation"].to(device)
            valid_mask = merged_learned["at_least_one_valid"].to(device)

            total_loss = 0.0
            l1_human_losses = []
            ssim_human_losses = []
            gaussians = GaussianModel_With_Act(0)
            for idx, img_gauss in enumerate(gaussians_list_batch):
                image_idx = img_gauss["image_idx"]
                data = dataset[img_gauss["i"]]
                smpl_scale = img_gauss["smpl_scale"].to(device)
                smpl_transl = img_gauss["smpl_transl"].to(device)
                smpl_global_orient = img_gauss["smpl_global_orient"].to(device)
                canonical_normals = img_gauss["normals"].to(device)
                camera = img_gauss["camera"]

                if offsets:
                    gs_xyz += (
                        canonical_normals
                        * merged_learned["offset_scales"].unsqueeze(1)
                        * smpl_scale
                    )

                gaussians.init_RT_seq({1.0: [camera]})

                deformed_xyz, human_smpl_scales, deformed_gs_rotq, _ = (
                    get_deform_from_T_to_pose(
                        gs_xyz,
                        gs_scaling,
                        gs_rotation,
                        valid_mask,
                        data["betas"],
                        data["body_pose"],
                        smpl_global_orient,
                        smpl_scale,
                        smpl_transl,
                        filter_visibility=True,
                    )
                )
                posed_features = {
                    "xyz": deformed_xyz[0],
                    "features_dc": gs_features_dc,
                    "features_rest": gs_features_rest,
                    "opacity": gs_opacity,
                    "scaling": human_smpl_scales,
                    "rotation": deformed_gs_rotq,
                }
                posed_features = {
                    k: (v[valid_mask] if k not in ["rotation"] else v)
                    for k, v in posed_features.items()
                }

                set_gaussian_model_features(gaussians, posed_features)
                b = random.random() < 0.5
                render_pkg = render_gaussians(
                    gaussians, [camera], bg="white" if b else "black"
                )
                image_posed_merged, viewspace_point_tensor, visibility_filter, radii = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                )
                gt_human = torch.where(
                    data["mask"].unsqueeze(0).bool(),
                    data["rgb"],
                    (
                        torch.ones_like(data["rgb"])
                        if b
                        else torch.zeros_like(data["rgb"])
                    ),
                )

                l1_loss_human = l1_loss(image_posed_merged, gt_human, mask=data["mask"])
                ssim_loss_human = (
                    1 - ssim(image_posed_merged, gt_human, mask=data["mask"])
                ) * (data["mask"].sum() / (gt_human.shape[-1] * gt_human.shape[-2]))
                loss_human = l1_loss_human + 100 * ssim_loss_human
                total_loss += loss_human

                l1_human_losses.append(l1_loss_human.item())
                ssim_human_losses.append(ssim_loss_human.item())

            total_loss /= len(gaussians_list_batch)
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                w = F.softmax(merge_layer.logits_vis, dim=0)
            lr_net, lr_vis, lr_corr = scheduler.get_last_lr()
            corr_logits = merge_layer.logits_correction.detach()
            wandb.log(
                {
                    "loss/loss": total_loss.item(),
                    "train/l1_loss_human": np.mean(l1_human_losses),
                    "train/ssim_loss_human": np.mean(ssim_human_losses),
                    "weights_mean": w.mean(),
                    "weights_median": w.median(),
                    "weights_std": w.std(),
                    "logits_corr/mean": corr_logits.mean().item(),
                    "logits_corr/var": corr_logits.var(unbiased=False).item(),
                    "logits_corr/median": corr_logits.median().item(),
                    "lr/net": lr_net,
                    "lr/vis": lr_vis,
                    "lr/corr": lr_corr,
                },
                step=step,
            )

        wandb.finish()
        with torch.no_grad():
            final_out = merge_layer(
                gaussians_list, range(len(gaussians_list)), output_everything=True
            )
        return final_out

    print("Start merging (mean vs. median)")
    t0 = time.time()
    merged_mean = merge_canonical_gaussians_mean(all_gaussians)
    print(f"→ mean done in {time.time() - t0:.2f}s")
    t0 = time.time()
    merged_median = merge_canonical_gaussians_median(all_gaussians)
    print(f"→ median done in {time.time() - t0:.2f}s")
    # t0 = time.time()
    # merged_mean_trimmed = merge_canonical_gaussians_trimmed(all_gaussians, trim_frac=0.2)
    # print(f"→ trimmed done in {time.time() - t0:.2f}s")
    t0 = time.time()
    merged_learned = learned_merging(all_gaussians)
    print(f"→ learned done in {time.time() - t0:.2f}s")
    t0 = time.time()
    merged_learned_offsets = learned_merging(all_gaussians, offsets=True)
    print(f"→ learned offsets done in {time.time() - t0:.2f}s")

    merged_gaussians = {
        "mean": merged_mean,
        "mean_offsets": merged_mean,
        "median": merged_median,
        "median_offset": merged_median,
        # "mean_trimmed": merged_mean_trimmed,
        "learned": merged_learned,
        "learned_offsets": merged_learned_offsets,
    }

    # Send everyting to device
    for _, merged_output in merged_gaussians.items():
        for key in merged_output.keys():
            if isinstance(merged_output[key], torch.Tensor):
                merged_output[key] = merged_output[key].to(device)

    # Save merging results
    for merging_name, merged_output in merged_gaussians.items():

        canonical_human_features = {
            "xyz": merged_output["xyz"],
            "features_dc": merged_output["features_dc"],
            "features_rest": merged_output["features_rest"],
            "opacity": merged_output["opacity"],
            "scaling": merged_output["scaling"],
            "rotation": merged_output["rotation"],
            "at_least_one_valid": merged_output["at_least_one_valid"],
        }

        safe_name = f"{exp_name}_{merging_name}.pkl"
        with open(safe_name, "wb") as f:
            pickle.dump(canonical_human_features, f)
    ##################################################

    lpips = LPIPS(net="vgg", pretrained=True).to(device)
    for param in lpips.parameters():
        param.requires_grad = False

    # image_ids = dataset.train_split
    # image_names = [
    #     f"/work/courses/3dv/14/projects/ml-hugs/data/neuman/dataset/lab/images/{i:05}.png"
    #     for i in image_ids
    # ]

    image_ids_val = dataset.val_split
    image_names_val = [
        f"/work/courses/3dv/14/projects/ml-hugs/data/neuman/dataset/lab/images/{i:05}.png"
        for i in image_ids_val
    ]

    timestamp = datetime.now().strftime("%d-%m_%H-%M")
    folder_name = f"{exp_name}-{timestamp}"
    os.makedirs(f"{SAVE_DIR}/{folder_name}", exist_ok=True)

    #########################################################################################################################

    scores = {
        merge_approach: {
            "l1_loss_human": [],
            "ssim_loss_human": [],
            "psnr_human": [],
            "lpips_human": [],
            "l1_loss_scene": [],
            "ssim_loss_scene": [],
            "psnr_scene": [],
            "lpips_scene": [],
            "l1_loss_scene_green": [],
            "ssim_loss_scene_green": [],
            "psnr_scene_green": [],
            "lpips_scene_green": [],
        }
        for merge_approach in list(merged_gaussians.keys()) + ["unmerged"]
    }

    with torch.no_grad():
        for i in tqdm(range(len(image_ids_val)), "creating merging results"):

            image_idx = image_ids_val[i]
            features_scene = {}
            features_scene["xyz"] = scene_gaussians[i]["xyz"].to(device)
            features_scene["features_dc"] = scene_gaussians[i]["features_dc"].to(device)
            features_scene["features_rest"] = scene_gaussians[i]["features_rest"].to(
                device
            )
            features_scene["opacity"] = scene_gaussians[i]["opacity"].to(device)
            features_scene["scaling"] = scene_gaussians[i]["scaling"].to(device)
            features_scene["rotation"] = scene_gaussians[i]["rotation"].to(device)

            img_path = image_names_val[i]
            images = load_and_preprocess_images([img_path]).to(device)[None]
            data = dataset[all_gaussians[i]["i"]]

            # Render canonical + posed human for each mergin technique
            out_imgs = {}
            for merge_approach, merged in merged_gaussians.items():

                # Frame specific features
                valid_mask = merged["visibility_masks"][i].to(device)
                smpl_scale = merged["smpl_scales"][i].to(device)
                smpl_transl = merged["smpl_transls"][i].to(device)
                smpl_global_orient = merged["smpl_global_orients"][i].to(device)
                if "offset" in merge_approach:
                    canonical_normals = merged["normals"][i].to(device)
                    xyz = (
                        merged["xyz"]
                        + canonical_normals
                        * merged["offset_scales"].unsqueeze(1)
                        * smpl_scale
                    )
                else:
                    xyz = merged["xyz"]

                gaussians_unmerged = {
                    key: pad_to_full(all_gaussians[i], key).to(device)
                    for key in [
                        "scaling",
                        "opacity",
                        "features_dc",
                        "features_rest",
                        "rotation",
                    ]
                }
                gaussians_unmerged["xyz"] = all_gaussians[i]["xyz"].to(device)

                new_features = {
                    "xyz": xyz * smpl_scale + smpl_transl,
                    "features_dc": merged["features_dc"],
                    "features_rest": merged["features_rest"],
                    "opacity": merged["opacity"],
                    "scaling": merged["scaling"] + smpl_scale.log(),
                    "rotation": merged["rotation"],
                }
                new_features = {
                    k: v[merged["at_least_one_valid"]] for k, v in new_features.items()
                }

                gif_features = {
                    "xyz": xyz,
                    "features_dc": merged["features_dc"],
                    "features_rest": merged["features_rest"],
                    "opacity": merged["opacity"],
                    "scaling": merged["scaling"],
                    "rotation": merged["rotation"],
                }
                gif_features = {
                    k: v[merged["at_least_one_valid"]] for k, v in gif_features.items()
                }
                render_smpl_gaussians_gif(
                    gif_features,
                    os.path.join(SAVE_DIR, folder_name, f"{merge_approach}.gif"),
                )

                camera = all_gaussians[i]["camera"]
                camera_list = [camera]
                gaussians = GaussianModel_With_Act(0)
                gaussians.init_RT_seq({1.0: camera_list})

                with torch.no_grad():
                    set_gaussian_model_features(gaussians, new_features)
                    render_pkg = render_gaussians(gaussians, camera_list, bg="white")
                    (
                        image_canonical_merged,
                        viewspace_point_tensor,
                        visibility_filter,
                        radii,
                    ) = (
                        render_pkg["render"],
                        render_pkg["viewspace_points"],
                        render_pkg["visibility_filter"],
                        render_pkg["radii"],
                    )
                    del render_pkg
                    gc.collect()
                    torch.cuda.empty_cache()
                    deformed_xyz, human_smpl_scales, deformed_gs_rotq, _ = (
                        get_deform_from_T_to_pose(
                            xyz,
                            merged["scaling"],
                            merged["rotation"],
                            merged["at_least_one_valid"],
                            data["betas"],
                            data["body_pose"],
                            smpl_global_orient,
                            smpl_scale,
                            smpl_transl,
                            filter_visibility=True,
                        )
                    )
                    new_features = {
                        "xyz": deformed_xyz[0],
                        "features_dc": merged["features_dc"],
                        "features_rest": merged["features_rest"],
                        "opacity": merged["opacity"],
                        "scaling": human_smpl_scales,
                        "rotation": deformed_gs_rotq,
                    }
                    new_features = {
                        k: (
                            v[merged["at_least_one_valid"]]
                            if k not in ["rotation"]
                            else v
                        )
                        for k, v in new_features.items()
                    }

                with torch.no_grad():
                    set_gaussian_model_features(gaussians, new_features)
                    render_pkg = render_gaussians(gaussians, camera_list, bg="white")
                    (
                        image_posed_merged,
                        viewspace_point_tensor,
                        visibility_filter,
                        radii,
                    ) = (
                        render_pkg["render"],
                        render_pkg["viewspace_points"],
                        render_pkg["visibility_filter"],
                        render_pkg["radii"],
                    )
                    del render_pkg
                    gc.collect()
                    torch.cuda.empty_cache()

                    # Loss calculation, here only for tracking and evaluation
                    gt_human = torch.where(
                        data["mask"].unsqueeze(0).bool(),
                        data["rgb"],
                        torch.ones_like(data["rgb"]),
                    )
                    l1_loss_human = l1_loss(
                        image_posed_merged, gt_human, mask=data["mask"]
                    )
                    ssim_loss_human = ssim(
                        image_posed_merged, gt_human, mask=data["mask"]
                    ) * (data["mask"].sum() / (gt_human.shape[-1] * gt_human.shape[-2]))

                    scores[merge_approach]["l1_loss_human"].append(
                        l1_loss_human.cpu().item()
                    )
                    scores[merge_approach]["ssim_loss_human"].append(
                        ssim_loss_human.cpu().item()
                    )
                    scores[merge_approach]["psnr_human"].append(
                        psnr(image_posed_merged, gt_human).item()
                    )
                    scores[merge_approach]["lpips_human"].append(
                        lpips(image_posed_merged.clip(max=1), gt_human).mean().item()
                    )

                merged_features = {
                    "xyz": torch.cat(
                        [
                            features_scene["xyz"],
                            deformed_xyz[0][merged["at_least_one_valid"]],
                        ],
                        dim=0,
                    ),
                    "features_dc": torch.cat(
                        [
                            features_scene["features_dc"],
                            merged["features_dc"][merged["at_least_one_valid"]],
                        ],
                        dim=0,
                    ),
                    "features_rest": torch.cat(
                        [
                            features_scene["features_rest"],
                            merged["features_rest"][merged["at_least_one_valid"]],
                        ],
                        dim=0,
                    ),
                    "opacity": torch.cat(
                        [
                            features_scene["opacity"],
                            merged["opacity"][merged["at_least_one_valid"]],
                        ],
                        dim=0,
                    ),
                    "scaling": torch.cat(
                        [
                            features_scene["scaling"],
                            human_smpl_scales[merged["at_least_one_valid"]],
                        ],
                        dim=0,
                    ),
                    "rotation": torch.cat(
                        [features_scene["rotation"], deformed_gs_rotq], dim=0
                    ),
                }
                with torch.no_grad():
                    rendered_img = render_and_save(
                        gaussians,
                        camera_list,
                        merged_features,
                        save_path=f"{SAVE_DIR}/{folder_name}/{image_idx}_{merge_approach}_white.png",
                        bg="white",
                    )
                    rendered_img_green = render_and_save(
                        gaussians,
                        camera_list,
                        merged_features,
                        save_path=f"{SAVE_DIR}/{folder_name}/{image_idx}_{merge_approach}_green.png",
                        bg="green",
                    )
                    l1_loss_scene = l1_loss(rendered_img, data["rgb"])
                    ssim_loss_scene = ssim(rendered_img, data["rgb"])
                    l1_loss_scene_green = l1_loss(rendered_img_green, data["rgb"])
                    ssim_loss_scene_green = ssim(rendered_img_green, data["rgb"])
                    scores[merge_approach]["l1_loss_scene"].append(
                        l1_loss_scene.cpu().item()
                    )
                    scores[merge_approach]["ssim_loss_scene"].append(
                        ssim_loss_scene.cpu().item()
                    )
                    scores[merge_approach]["lpips_scene"].append(
                        lpips(rendered_img.clip(max=1), data["rgb"]).mean().item()
                    )
                    scores[merge_approach]["psnr_scene"].append(
                        psnr(rendered_img, data["rgb"]).item()
                    )
                    scores[merge_approach]["l1_loss_scene_green"].append(
                        l1_loss_scene_green.cpu().item()
                    )
                    scores[merge_approach]["ssim_loss_scene_green"].append(
                        ssim_loss_scene_green.cpu().item()
                    )
                    scores[merge_approach]["psnr_scene_green"].append(
                        psnr(rendered_img_green, data["rgb"]).item()
                    )
                    scores[merge_approach]["lpips_scene_green"].append(
                        lpips(rendered_img_green.clip(max=1), data["rgb"]).mean().item()
                    )

                out_imgs[merge_approach] = {
                    "canonical": image_canonical_merged.cpu(),
                    "posed": image_posed_merged.detach().cpu(),
                }

            with torch.no_grad():
                gif_features = {
                    "xyz": gaussians_unmerged["xyz"],
                    "features_dc": gaussians_unmerged["features_dc"],
                    "features_rest": gaussians_unmerged["features_rest"],
                    "opacity": gaussians_unmerged["opacity"],
                    "scaling": gaussians_unmerged["scaling"],
                    "rotation": gaussians_unmerged["rotation"],
                }
                gif_features = {k: v[valid_mask] for k, v in gif_features.items()}
                render_smpl_gaussians_gif(
                    gif_features,
                    os.path.join(SAVE_DIR, folder_name, f"unmerged{image_idx}.gif"),
                )
                unmerged_features = {
                    "xyz": gaussians_unmerged["xyz"] * smpl_scale + smpl_transl,
                    "features_dc": gaussians_unmerged["features_dc"],
                    "features_rest": gaussians_unmerged["features_rest"],
                    "opacity": gaussians_unmerged["opacity"],
                    "scaling": gaussians_unmerged["scaling"] + smpl_scale.log(),
                    "rotation": gaussians_unmerged["rotation"],
                }
                unmerged_features = {
                    k: v[valid_mask] for k, v in unmerged_features.items()
                }
                set_gaussian_model_features(gaussians, unmerged_features)
                render_pkg = render_gaussians(gaussians, camera_list, bg="white")
                image_canonical, viewspace_point_tensor, visibility_filter, radii = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                )

                (
                    deformed_xyz,
                    human_smpl_scales_unmerged,
                    deformed_gs_rotq_unmerged,
                    _,
                ) = get_deform_from_T_to_pose(
                    gaussians_unmerged["xyz"],
                    gaussians_unmerged["scaling"],
                    gaussians_unmerged["rotation"],
                    valid_mask,
                    data["betas"],
                    data["body_pose"],
                    smpl_global_orient,
                    smpl_scale,
                    smpl_transl,
                    filter_visibility=True,
                )
                unmerged_features = {
                    "xyz": deformed_xyz[0],
                    "features_dc": gaussians_unmerged["features_dc"],
                    "features_rest": gaussians_unmerged["features_rest"],
                    "opacity": gaussians_unmerged["opacity"],
                    "scaling": human_smpl_scales_unmerged,
                    "rotation": deformed_gs_rotq_unmerged,
                }
                unmerged_features = {
                    k: (v[valid_mask] if k not in ["rotation"] else v)
                    for k, v in unmerged_features.items()
                }

                set_gaussian_model_features(gaussians, unmerged_features)
                render_pkg = render_gaussians(gaussians, camera_list, bg="white")
                image_posed, viewspace_point_tensor, visibility_filter, radii = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                )

                scores["unmerged"]["l1_loss_human"].append(
                    l1_loss(image_posed, gt_human, mask=data["mask"]).cpu().item()
                )
                scores["unmerged"]["ssim_loss_human"].append(
                    (
                        ssim(image_posed, gt_human, mask=data["mask"])
                        * (
                            data["mask"].sum()
                            / (gt_human.shape[-1] * gt_human.shape[-2])
                        )
                    )
                    .cpu()
                    .item()
                )
                scores["unmerged"]["psnr_human"].append(
                    psnr(image_posed, gt_human).item()
                )
                scores["unmerged"]["lpips_human"].append(
                    lpips(image_posed.clip(max=1), gt_human).mean().item()
                )

                merged_features = {
                    "xyz": torch.cat(
                        [features_scene["xyz"], deformed_xyz[0][valid_mask]], dim=0
                    ),
                    "features_dc": torch.cat(
                        [
                            features_scene["features_dc"],
                            gaussians_unmerged["features_dc"][valid_mask],
                        ],
                        dim=0,
                    ),
                    "features_rest": torch.cat(
                        [
                            features_scene["features_rest"],
                            gaussians_unmerged["features_rest"][valid_mask],
                        ],
                        dim=0,
                    ),
                    "opacity": torch.cat(
                        [
                            features_scene["opacity"],
                            gaussians_unmerged["opacity"][valid_mask],
                        ],
                        dim=0,
                    ),
                    "scaling": torch.cat(
                        [
                            features_scene["scaling"],
                            human_smpl_scales_unmerged[valid_mask],
                        ],
                        dim=0,
                    ),
                    "rotation": torch.cat(
                        [features_scene["rotation"], deformed_gs_rotq_unmerged], dim=0
                    ),
                }

                rendered_img = render_and_save(
                    gaussians,
                    camera_list,
                    merged_features,
                    save_path=f"{SAVE_DIR}/{folder_name}/{image_idx}_unmerged_white.png",
                    bg="white",
                )
                rendered_img_green = render_and_save(
                    gaussians,
                    camera_list,
                    merged_features,
                    save_path=f"{SAVE_DIR}/{folder_name}/{image_idx}_unmerged_green.png",
                    bg="green",
                )

                l1_loss_scene = l1_loss(rendered_img, data["rgb"])
                ssim_loss_scene = ssim(rendered_img, data["rgb"])
                l1_loss_scene_green = l1_loss(rendered_img_green, data["rgb"])
                ssim_loss_scene_green = ssim(rendered_img_green, data["rgb"])
                scores["unmerged"]["l1_loss_scene"].append(l1_loss_scene.cpu().item())
                scores["unmerged"]["ssim_loss_scene"].append(
                    ssim_loss_scene.cpu().item()
                )
                scores["unmerged"]["psnr_scene"].append(
                    psnr(rendered_img, data["rgb"]).item()
                )
                scores["unmerged"]["lpips_scene"].append(
                    lpips(rendered_img.clip(max=1), data["rgb"]).mean().item()
                )
                scores["unmerged"]["l1_loss_scene_green"].append(
                    l1_loss_scene_green.cpu().item()
                )
                scores["unmerged"]["ssim_loss_scene_green"].append(
                    ssim_loss_scene_green.cpu().item()
                )
                scores["unmerged"]["psnr_scene_green"].append(
                    psnr(rendered_img_green, data["rgb"]).item()
                )
                scores["unmerged"]["lpips_scene_green"].append(
                    lpips(rendered_img_green.clip(max=1), data["rgb"]).mean().item()
                )

            # Save example images for each merging technique
            for merge_name, imgs in out_imgs.items():
                # prepare file-base
                base = f"{image_idx}_{merge_name}"
                # canonical
                pil_canon = to_pil_image(imgs["canonical"].cpu())
                pil_canon.save(
                    os.path.join(SAVE_DIR, folder_name, f"{base}a_canonical.png")
                )
                # posed
                pil_posed = to_pil_image(imgs["posed"].cpu())
                pil_posed.save(
                    os.path.join(SAVE_DIR, folder_name, f"{base}a_posed.png")
                )

            del features_scene
            gc.collect()
            torch.cuda.empty_cache()

        scores = {
            key: {
                "l1_loss_human": np.mean(scores[key]["l1_loss_human"]),
                "ssim_loss_human": np.mean(scores[key]["ssim_loss_human"]),
                "psnr_human": np.mean(scores[key]["psnr_human"]),
                "lpips_human": np.mean(scores[key]["lpips_human"]),
                "l1_loss_scene": np.mean(scores[key]["l1_loss_scene"]),
                "ssim_loss_scene": np.mean(scores[key]["ssim_loss_scene"]),
                "psnr_scene": np.mean(scores[key]["psnr_scene"]),
                "lpips_scene": np.mean(scores[key]["lpips_scene"]),
                "l1_loss_scene_green": np.mean(scores[key]["l1_loss_scene_green"]),
                "ssim_loss_scene_green": np.mean(scores[key]["ssim_loss_scene_green"]),
                "psnr_scene_green": np.mean(scores[key]["psnr_scene_green"]),
                "lpips_scene_green": np.mean(scores[key]["lpips_scene_green"]),
            }
            for key in scores.keys()
        }

        # print results
        print("Scores:")
        print(
            "                 | L1 Hum |SSIM Hum|PSNR Hum|LPIPS Hum| L1 Sce| SSIM Sce|PSNR Sce|LPIPS Sce|L1 Sce G |SSIM Sce G|PSNR Sce G|LPIPS Sce G"
        )
        for name, sc in scores.items():
            print(
                f"{name:16s} | {sc['l1_loss_human']:.4f} | {sc['ssim_loss_human']:.4f} | {sc['psnr_human']:.2f} | {sc['lpips_human']:.2f} | "
                f"{sc['l1_loss_scene']:.4f} | {sc['ssim_loss_scene']:.4f} | {sc['psnr_scene']:.2f} | {sc['lpips_scene']:.2f} | "
                f"{sc['l1_loss_scene_green']:.4f} | {sc['ssim_loss_scene_green']:.4f} | {sc['psnr_scene_green']:.2f}| {sc['lpips_scene_green']:.2f}"
            )
