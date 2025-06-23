#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from random import randint
from time import time
import sys
import json
sys.path.append('.')
import numpy as np
import torch
import torchvision
import torchvision.utils as vutils
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from argparse import ArgumentParser, Namespace

from instantsplat.arguments import ModelParams, PipelineParams, OptimizationParams
from instantsplat.gaussian_renderer import render, network_gui
from instantsplat.scene import Scene, GaussianModel
from instantsplat.scene.cameras import Camera
from instantsplat.utils.camera_utils import generate_interpolated_path
from instantsplat.utils.general_utils import safe_state
from instantsplat.utils.graphics_utils import getWorld2View2_torch
from instantsplat.utils.image_utils import psnr
from instantsplat.utils.loss_utils import l1_loss, ssim
from instantsplat.utils.pose_utils import get_camera_from_tensor
from instantsplat.utils.sfm_utils import save_time
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

from instantsplat.utils.sh_utils import eval_sh

def save_pose(path, quat_pose, train_cams, llffhold=2):
    # Get camera IDs and convert quaternion poses to camera matrices
    camera_ids = [cam.colmap_id for cam in train_cams]
    world_to_camera = [get_camera_from_tensor(quat) for quat in quat_pose]
    
    # Reorder poses according to colmap IDs
    colmap_poses = []
    for i in range(len(camera_ids)):
        idx = camera_ids.index(i + 1)  # Find position of camera i+1
        pose = world_to_camera[idx]
        colmap_poses.append(pose)
    
    # Convert to numpy array and save
    colmap_poses = torch.stack(colmap_poses).detach().cpu().numpy()
    np.save(path, colmap_poses)


def load_and_prepare_confidence(confidence_path, device='cuda', scale=(0.1, 1.0)):
    """
    Loads, normalizes, inverts, and scales confidence values to obtain learning rate modifiers.
    
    Args:
        confidence_path (str): Path to the .npy confidence file.
        device (str): Device to load the tensor onto.
        scale (tuple): Desired range for the learning rate modifiers.
    
    Returns:
        torch.Tensor: Learning rate modifiers.
    """
    # Load and normalize
    confidence_np = np.load(confidence_path)
    confidence_tensor = torch.from_numpy(confidence_np).float().to(device)
    normalized_confidence = torch.sigmoid(confidence_tensor)

    # Invert confidence and scale to desired range
    inverted_confidence = 1.0 - normalized_confidence
    min_scale, max_scale = scale
    lr_modifiers = inverted_confidence * (max_scale - min_scale) + min_scale
    
    return lr_modifiers

import re
from pathlib import Path
def extract_lab_index(model_path):
    # Convert to Path object in case it's a string
    model_path = Path(model_path)

    # Search for "lab_<number>" in any part of the path
    match = re.search(r"lab_(\d+)", str(model_path))
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"'lab_i' not found in path: {model_path}")

def filter_and_export_final_gaussians_full(gaussians, model_path, depthmaps_path, intrinsics_path, extrinsics_path, mask_dir, output_name="filtered_gaussians.ply"):
    import numpy as np
    import torch
    import os
    import cv2
    from pathlib import Path
    from scipy.spatial import cKDTree
    from plyfile import PlyData, PlyElement

    depthmaps = np.load(depthmaps_path)
    intrinsics = np.load(intrinsics_path)
    extrinsics = np.load(extrinsics_path)

    all_pts = []
    selected_frame = extract_lab_index(model_path)

    for i in range(len(depthmaps)):
        depth = depthmaps[i].squeeze()
        mask = cv2.imread(str(Path(mask_dir) / f"{selected_frame:05d}.png"), cv2.IMREAD_GRAYSCALE)

        if mask.shape != depth.shape:
            mask = cv2.resize(mask, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = (mask < 127).astype(np.bool_)

        ys, xs = np.where(mask)
        z = depth[ys, xs]
        if z.ndim > 1:
            z = z[:, 0]

        if intrinsics.ndim == 3:
            intrinsics = intrinsics[0]

        x = (xs - intrinsics[0, 2]) * z / intrinsics[0, 0]
        y = (ys - intrinsics[1, 2]) * z / intrinsics[1, 1]
        cam_pts = np.stack((x, y, z, np.ones_like(z)), axis=-1).T

        c2w = np.eye(4)
        c2w[:3, :4] = extrinsics[i]
        world_pts = (c2w @ cam_pts)[:3].T

        all_pts.append(world_pts)

    human_pts = np.concatenate(all_pts, axis=0)
    tree = cKDTree(human_pts)
    xyz = gaussians.get_xyz.detach().cpu().numpy()
    _, idxs = tree.query(xyz, distance_upper_bound=0.01)
    mask = idxs != human_pts.shape[0]

    # Collect all fields
    filtered_xyz = gaussians.get_xyz[mask].detach().cpu().numpy()
    normals = np.zeros_like(filtered_xyz)

    f_dc = gaussians._features_dc[mask].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = gaussians._features_rest[mask].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()

    opacities = gaussians._opacity[mask].detach().cpu().numpy()
    scales = gaussians._scaling[mask].detach().cpu().numpy()
    rotations = gaussians._rotation[mask].detach().cpu().numpy()

    # Create structured array
    dtype_full = [(attribute, 'f4') for attribute in gaussians.construct_list_of_attributes()]
    elements = np.empty(filtered_xyz.shape[0], dtype=dtype_full)

    attributes = np.concatenate((filtered_xyz, normals, f_dc, f_rest, opacities, scales, rotations), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Save to PLY
    output_path = os.path.join(model_path, output_name)
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(output_path)

    print(f"[INFO] Saved filtered human-only FULL Gaussians to: {output_path}")


def filter_and_export_final_gaussians(gaussians, model_path, depthmaps_path, intrinsics_path, extrinsics_path, mask_dir, output_name="filtered_gaussians.ply"):
    import numpy as np
    import torch
    import os
    import cv2
    from pathlib import Path
    from scipy.spatial import cKDTree

    depthmaps = np.load(depthmaps_path)
    intrinsics = np.load(intrinsics_path)
    extrinsics = np.load(extrinsics_path)

    all_pts = []
    selected_frame = extract_lab_index(model_path)
    for i in range(len(depthmaps)):
        depth = depthmaps[i].squeeze()
        #print("trying to read:")
        #print(str(Path(mask_dir) / f"f{i:05d}.png"))
        mask = cv2.imread(str(Path(mask_dir) / f"{selected_frame:05d}.png"), cv2.IMREAD_GRAYSCALE)

        print(f"[DEBUG] Frame {i:05d} - Depth shape: {depth.shape}, Mask shape: {mask.shape}")
        if mask.shape != depth.shape:
            mask = cv2.resize(mask, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)
        print(f"[DEBUG] Frame {i:05d} - Depth shape: {depth.shape}, Mask shape: {mask.shape}")
        mask = (mask < 127).astype(np.bool_)



        ys, xs = np.where(mask)
        z = depth[ys, xs]
        print(f"[DEBUG] Nonzero mask pixels: {z}")
        if z.ndim > 1:
            z = z[:, 0] 
        
        print(f"[DEBUG] Intrinsics shape: {intrinsics.shape}")
        if intrinsics.ndim == 3:
            intrinsics = intrinsics[0]
            print(f"[DEBUG] Using first intrinsics matrix")
        x = (xs - intrinsics[0, 2]) * z / intrinsics[0, 0]
        y = (ys - intrinsics[1, 2]) * z / intrinsics[1, 1]
        cam_pts = np.stack((x, y, z, np.ones_like(z)), axis=-1).T

        c2w = np.eye(4)
        c2w[:3, :4] = extrinsics[i]
        world_pts = (c2w @ cam_pts)[:3].T

        all_pts.append(world_pts)

    human_pts = np.concatenate(all_pts, axis=0)
    tree = cKDTree(human_pts)
    xyz = gaussians.get_xyz.detach().cpu().numpy()
    _, idxs = tree.query(xyz, distance_upper_bound=0.01)
    mask = idxs != human_pts.shape[0]

    filtered_xyz = gaussians.get_xyz[mask].detach().cpu().numpy()
    shs_view = gaussians.get_features.transpose(1, 2).view(
    -1, 3, (gaussians.max_sh_degree + 1) ** 2
)       
    dir_dummy = torch.tensor([0.0, 0.0, 1.0], device=shs_view.device).repeat(shs_view.shape[0], 1)
    dir_dummy = dir_dummy / dir_dummy.norm(dim=1, keepdim=True)

    # Evaluate SH
    sh2rgb = eval_sh(gaussians.active_sh_degree, shs_view, dir_dummy)
    colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

    # Filter RGBs
    filtered_rgb = colors_precomp[mask].detach().cpu().numpy()
    #filtered_rgb = gaussians.get_colors[mask].detach().cpu().numpy()

    output_path = os.path.join(model_path, output_name)
    with open(output_path, "w") as f:
        f.write("ply\nformat ascii 1.0\nelement vertex {}\n".format(filtered_xyz.shape[0]))
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p, c in zip(filtered_xyz, filtered_rgb):
            f.write("{:.5f} {:.5f} {:.5f} {} {} {}\n".format(p[0], p[1], p[2], int(c[0]*255), int(c[1]*255), int(c[2]*255)))

    print(f"[INFO] Saved filtered human-only Gaussians to: {output_path}")

import rerun as rr
def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    #rr.init("debugging", recording_id="v0.1")
    #rr.connect_tcp("0.0.0.0:9876")


    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)

    # per-point-optimizer
    confidence_path = os.path.join(dataset.source_path, f"sparse_{dataset.n_views}/0", "confidence_dsp.npy")
    confidence_lr = load_and_prepare_confidence(confidence_path, device='cuda', scale=(1, 100))
    scene = Scene(dataset, gaussians)

    if opt.pp_optimizer:
        gaussians.training_setup_pp(opt, confidence_lr)                          
    else:
        gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    train_cams_init = scene.getTrainCameras().copy()
    for save_iter in saving_iterations:
        os.makedirs(scene.model_path + f'/pose/ours_{save_iter}', exist_ok=True)
        save_pose(scene.model_path + f'/pose/ours_{save_iter}/pose_org.npy', gaussians.P, train_cams_init)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")    
    first_iter += 1
    start = time()
    for iteration in range(first_iter, opt.iterations + 1):        
        # if network_gui.conn == None:
        #     network_gui.try_connect()
        # while network_gui.conn != None:
        #     try:
        #         net_image_bytes = None
        #         custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
        #         if custom_cam != None:
        #             net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
        #             net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
        #         network_gui.send(net_image_bytes, dataset.source_path)
        #         if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
        #             break
        #     except Exception as e:
        #         network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        if opt.optim_pose==False:
            gaussians.P.requires_grad_(False)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # SAVE ALL CAMERAS
        viewpoint_stack_train = scene.getTrainCameras().copy()
        viewpoint_indices_train = list(range(len(viewpoint_stack_train)))
        print("fdldsfs")
        # for i in range(len(viewpoint_stack_train)):
        #     viewpoint_cam = viewpoint_stack_train[i]
        #     print("uuid", viewpoint_cam.uid, "i", i)
        #     pose = gaussians.get_RT(viewpoint_cam.uid)
        #     torch.save(viewpoint_cam, scene.model_path + f"/viewpoint_cam_{viewpoint_cam.uid}.pth")
        #     #torch.save(pose, scene.model_path + f"/pose/viewpoint_cam_{i}.pth")
        
        # viewpoint_stack_test = scene.getTestCameras().copy()
        # viewpoint_indices_test = list(range(len(viewpoint_stack_test)))
        # for i in range(len(viewpoint_stack_test)):
        #     viewpoint_cam = viewpoint_stack_test[i]
        #     pose = gaussians.get_RT(viewpoint_cam.uid)
        #     print("uuid", viewpoint_cam.uid, "i", i)
        #     torch.save(viewpoint_cam, scene.model_path + f"/viewpoint_cam_test_{viewpoint_cam.uid}.pth")
        #     #torch.save(pose, scene.model_path + f"/pose/viewpoint_cam_{i}.pth")
            


        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))

        # for each cam take the render and save
        for i in range(len(viewpoint_stack)):
            viewpoint_cam = viewpoint_stack[i]
            pose = gaussians.get_RT(viewpoint_cam.uid)
            uidd = viewpoint_cam.uid
            torch.save(viewpoint_cam, scene.model_path + f"/viewpoint_cam_{viewpoint_cam.uid}.pth")

            bg = torch.rand((3), device="cuda") if opt.random_background else background
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg, camera_pose=pose)

            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            #depth = render_pkg["depth_map"]

            # Loss
            gt_image = viewpoint_cam.original_image.cuda()
            # save gt and rendered image for debugging
            debug_dir = f"{scene.model_path}/renders"
            os.makedirs(f"{scene.model_path}/renders", exist_ok=True)
            vutils.save_image(gt_image, os.path.join(debug_dir, f"{iteration:06d}_gt_{uidd}.png"))

            vutils.save_image(image, os.path.join(debug_dir, f"{iteration:06d}_rendered_cam_{uidd}.png"))

            #vutils.save_image(depth, os.path.join(debug_dir, f"{iteration:06d}_rendered_depth.png"))



            #torch.save(pose, scene.model_path + f"/pose/viewpoint_cam_{i}.pth")
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)
        pose = gaussians.get_RT(viewpoint_cam.uid)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        #torch.save(viewpoint_cam, scene.model_path + f"/viewpoint_cam_{viewpoint_cam.uid}.pth")
        #torch.save(gaussians.capture(), scene.model_path + "/gaussian_object.pth")

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, camera_pose=pose)
        print("pipe" , pipe)
        print("bg" , bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        depth = render_pkg["depth_map"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        # save gt and rendered image for debugging
        debug_dir = f"{scene.model_path}/renders"
        os.makedirs(f"{scene.model_path}/renders", exist_ok=True)
        vutils.save_image(gt_image, os.path.join(debug_dir, f"{iteration:06d}_gt.png"))

        vutils.save_image(image, os.path.join(debug_dir, f"{iteration:06d}_rendered.png"))

        vutils.save_image(depth, os.path.join(debug_dir, f"{iteration:06d}_rendered_depth.png"))

        try: 
            print("shape of depth", depth.shape)
            depth_np = depth.squeeze().detach().cpu().numpy()
            np.save(os.path.join(debug_dir, f"{iteration:06d}_rendered_depth.npy"), depth_np)
        except Exception as e:
            print("Error saving depth map in numpy format:", e)
            print("Error saving depth map in numpy format")

        
        if iteration % 50 == 1:
            shs_view = gaussians.get_features.transpose(1, 2).view(
                -1, 3, (gaussians.max_sh_degree + 1) ** 2
            )
            dir_pp = gaussians.get_xyz - viewpoint_cam.camera_center.repeat(
                gaussians.get_features.shape[0], 1
            )
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(gaussians.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)


            #rr.set_time_seconds("frame", iteration/50)
            #rr.log(f"world/pred_pc", rr.Points3D(positions=gaussians._xyz.data[visibility_filter].reshape(-1, 3).detach().cpu().numpy(), colors=colors_precomp[visibility_filter].detach().cpu().numpy()))

        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        loss.backward()
        iter_end.record()
        # for param_group in gaussians.optimizer.param_groups:
        #     for param in param_group['params']:
        #         if param is gaussians.P:
        #             print(viewpoint_cam.uid, param.grad)
        #             break
        # print("Gradient of self.P:", gaussians.P.grad)
        
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Densification
            # if iteration < opt.densify_until_iter:
                # # Keep track of max radii in image-space for pruning
                # gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                # gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                # if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                #     size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                #     gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                # if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                #     gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            # Log and save
            if iteration == opt.iterations:
                end = time()
                train_time_wo_log = end - start
                save_time(scene.model_path, '[2] train_joint_TrainTime', train_time_wo_log)
                training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                save_pose(scene.model_path + f'/pose/ours_{iteration}/pose_optimized.npy', gaussians.P, train_cams_init)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
            
            # if iteration  in [1, 50, 100, 200, 500, 999] or iteration in saving_iterations:
            #     print(dataset.source_path)
            #     filter_and_export_final_gaussians_full(
            #         gaussians,
            #         model_path=scene.model_path,
            #         depthmaps_path=os.path.join(scene.model_path, 'depthmaps.npy'),
            #         intrinsics_path=os.path.join(scene.model_path, 'intrinsics.npy'),
            #         extrinsics_path=os.path.join(scene.model_path, 'extrinsics.npy'),
            #         mask_dir=f"{dataset.source_path}/segmentations",
            #         output_name=os.path.join(scene.model_path, f"filtered_gaussians_iter{iteration}.ply"))
    end = time()
    train_time = end - start
    save_time(scene.model_path, '[2] train_joint', train_time)


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            # unique_str = str(uuid.uuid4())
            unique_str = "irem"
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations or iteration % 5000 == 0:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(len(scene.getTrainCameras()))]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    if config['name']=="train":
                        pose = scene.gaussians.get_RT(viewpoint.uid)
                    else:
                        pose = scene.gaussians.get_RT_test(viewpoint.uid)
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, camera_pose=pose)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=True)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    # os.makedirs(args.model_path, exist_ok=True)
    # os.makedirs("./mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth", exist_ok=True)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")




