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
import torch
import lpips
import trimesh
import imageio
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui

# from scene import Scene, GaussianModel
from datasets import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from PIL import Image
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from argparse import Namespace
from utils.depth_loss_utils import ScaleAndShiftInvariantLoss
from utils.camera_utils import Camera, SampleCamera
from colmap_utils import read_cameras_text, undistorted_camera, camera_to_intrinsic
from chamfer import chamfer_distance
import numpy as np
import torchvision.utils as vutils

## Extrapolation
from extrapolation.extrapolator import Extrapolator

from args import config_args
from extrapolation.outpaint.crop import OutpaintCrop
from extrapolation.outpaint.sparse import OutpaintSparse
from extrapolation.outpaint.rotation import OutpaintRotation
import cv2
import pickle
import matplotlib.pyplot as plt

import h5py
from pytorch3d.ops import knn_points
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def fnv_hash_vec(arr):
    """
    FNV64-1A
    """
    assert arr.ndim == 2
    # Floor first for negative coordinates
    arr = arr.copy()
    arr = arr.astype(np.uint64, copy=False)
    hashed_arr = np.uint64(14695981039346656037) * np.ones(arr.shape[0], dtype=np.uint64)
    for j in range(arr.shape[1]):
        hashed_arr *= np.uint64(1099511628211)
        hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
    return hashed_arr


def ravel_hash_vec(arr):
    """
    Ravel the coordinates after subtracting the min coordinates.
    """
    assert arr.ndim == 2
    arr = arr.copy()
    arr -= arr.min(0)
    arr = arr.astype(np.uint64, copy=False)
    arr_max = arr.max(0).astype(np.uint64) + 1

    keys = np.zeros(arr.shape[0], dtype=np.uint64)
    # Fortran style indexing
    for j in range(arr.shape[1] - 1):
        keys += arr[:, j]
        keys *= arr_max[j + 1]
    keys += arr[:, -1]
    return keys


def voxelize(coord, voxel_size=0.05, hash_type='fnv', mode='random'):
    '''
    Voxelization of the input coordinates
    Parameters:
        coord: input coordinates (N x D)
        voxel_size: Size of the voxels
        hash_type: Type of the hashing function, can be chosen from 'ravel' and 'fnv'
        mode: 'random', 'deterministic' or 'multiple' mode. In training mode one selects a random point within the voxel as the representation of the voxel.
              In deterministic model right now one always uses the first point. Usually random mode is preferred for training. In 'multiple' mode, we will return
              multiple sets of indices, so that each point will be covered in at least one of these sets
    Returns:
        idx_unique: the indices of the points so that there is at most one point for each voxel
    '''
    coord = coord - coord.min(0)
    discrete_coord = np.floor(coord / np.array(voxel_size))
    if hash_type == 'ravel':
        key = ravel_hash_vec(discrete_coord)
    else:
        key = fnv_hash_vec(discrete_coord)

    idx_sort = np.argsort(key)
    key_sort = key[idx_sort]
    _, count = np.unique(key_sort, return_counts=True)
    if mode == 'deterministic':
        # idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + torch.randint(count.max(), (count.size,)).numpy() % count
        idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + np.zeros((count.size,), dtype=np.int32)
        idx_unique = idx_sort[idx_select]
        return idx_unique
    elif mode == 'multiple':  # mode is 'multiple'
        idx_data = []
        for i in range(count.max()):
            idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
            idx_part = idx_sort[idx_select]
            idx_data.append(idx_part)
        return idx_data
    else:  # mode == 'random'
        # idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + np.random.randint(0, count.max(), count.size) % count
        idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + torch.randint(count.max(), (count.size,)).numpy() % count
        idx_unique = idx_sort[idx_select]
        return idx_unique

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def filter_complete_points(
    valid_points, 
    complete_points, 
    max_coord=25.0,
    min_coord=-25.0,
    voxel_size=1.0
):
    """
    Chunk complete_points into voxel chunks and only keep chunks that contain points from 
    either input points or gt_points.
    """    
    # Calculate number of voxels in each dimension
    coord_range = max_coord - min_coord
    num_voxels = int(np.ceil(coord_range / voxel_size))

    # get voxels occupied by complete points
    complete_points_shifted = complete_points - min_coord
    voxel_indices = np.floor(complete_points_shifted / voxel_size).astype(int)
    voxel_indices_1d = voxel_indices[:,0] + \
                    voxel_indices[:,1] * num_voxels + \
                    voxel_indices[:,2] * num_voxels * num_voxels
    complete_unique_voxels, chunk_indices = np.unique(voxel_indices_1d, return_inverse=True)
    complete_unique_voxels = complete_unique_voxels.tolist()

    # get voxels occupied by valid points
    valid_points_shifted = valid_points - min_coord
    voxel_indices = np.floor(valid_points_shifted / voxel_size).astype(int)
    voxel_indices_1d = voxel_indices[:,0] + \
                    voxel_indices[:,1] * num_voxels + \
                    voxel_indices[:,2] * num_voxels * num_voxels
    valid_unique_voxels, _ = np.unique(voxel_indices_1d, return_inverse=True)
    valid_unique_voxels = valid_unique_voxels.tolist()

    filtered_complete_points = []
    for voxel_idx in valid_unique_voxels:
        if voxel_idx in complete_unique_voxels:
            chunk_mask = chunk_indices == complete_unique_voxels.index(voxel_idx)
            filtered_complete_points.append(complete_points[chunk_mask])

    filtered_complete_points = np.concatenate(filtered_complete_points, axis=0)
    return filtered_complete_points


def backproject(depth, extrinsics, K):
    """
    Backproject depth map into 3D in world coordinate system.
    """
    u, v = np.meshgrid(range(depth.shape[1]), range(depth.shape[0]))
    u = u.reshape([1, -1])[0]
    v = v.reshape([1, -1])[0]

    z = depth[v, u]

    # remove invalid pixels (i.e., zero depth)
    non_inf_indices = np.argwhere(z > 0).T[0]
    z = z[non_inf_indices]
    u = u[non_inf_indices]
    v = v[non_inf_indices]

    # calculate coordinates
    x = (u - K[0][2]) * z / K[0][0]
    y = (v - K[1][2]) * z / K[1][1]

    point_cam = np.vstack([x, y, z]).T

    # camera to world coordinates
    R = extrinsics[:3, :3]
    T = extrinsics[:3, 3]
    point_canonical = point_cam.dot(R.T) + T

    return point_canonical


class TrainManager:
    def __init__(self, args):
        self.args = args
        self.tb_writer = self.init_logger()
        if self.args.repair or self.args.mono_depth:
            self.extrapolator = Extrapolator(self.args)
        else:
            self.extrapolator = None
        self.scene = self.init_scene()
        self.init_constants()
        self.init_depth_loss()
        self.init_lpips_loss()
        self.init_ema_log()

        print(self.print_args())

    def print_args(self):
        output = []
        output.append("\nArguments:")
        output.append("-" * 50)

        # Dataset args
        output.append("\nDataset Parameters:")
        for key, value in vars(self.args.dataset).items():
            output.append(f"{key}: {value}")

        # Model args
        output.append("\nModel Parameters:")
        for key, value in vars(self.args.model).items():
            output.append(f"{key}: {value}")

        # Optimization args
        output.append("\nOptimization Parameters:")
        for key, value in vars(self.args.opt).items():
            output.append(f"{key}: {value}")

        # Pipeline args
        output.append("\nPipeline Parameters:")
        for key, value in vars(self.args.pipe).items():
            output.append(f"{key}: {value}")

        # Other args
        output.append("\nOther Parameters:")
        for key, value in vars(self.args).items():
            if key not in ["dataset", "model", "opt", "pipe"]:
                output.append(f"{key}: {value}")

        output.append("-" * 50)

        print("\n".join(output))

    def init_logger(self):
        tb_writer = prepare_output_and_logger(self.args)
        return tb_writer

    def init_scene(self):
        if self.args.outpaint_type == "crop":
            gaussians = GaussianModel(self.args.model.sh_degree, sparse_aware=True)
        else:
            gaussians = GaussianModel(self.args.model.sh_degree)
        scene = Scene(
            self.args.model.model_path, self.args, gaussians, self.extrapolator
        )
        gaussians.training_setup(self.args.opt)
        if self.args.start_checkpoint:
            (model_params, first_iter) = torch.load(self.args.start_checkpoint)
            gaussians.restore(model_params, self.args.opt)
        return scene

    def init_constants(self):
        bg_color = [1, 1, 1] if self.args.model.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self.args.background = self.background

    def init_depth_loss(self):
        self.depth_loss_fun = ScaleAndShiftInvariantLoss()
        # self.depth_loss_fun = depth_ranking_loss

    def init_lpips_loss(self):
        self.lpips_loss_fun = lpips.LPIPS(net="vgg").cuda()

    def init_ema_log(self):
        self.ema_loss_for_log = 0.0
        self.ema_dist_for_log = 0.0
        self.ema_normal_for_log = 0.0

    # train from start_iter to end_iter (inclusive)
    def train(self, start_iter, end_iter):
        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)

        progress_bar = tqdm(
            range(start_iter, end_iter),
            initial=start_iter,
            total=end_iter,
            desc="Training progress",
        )
        start_iter += 1

        for iteration in range(start_iter, end_iter + 1):
            iter_start.record()
            self.scene.gaussians.update_learning_rate(iteration)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                self.scene.gaussians.oneupSHdegree()

            viewpoint = self.scene.getTrainInstant()
            cam = Camera(viewpoint["cam_info"])

            render_pkg = render(
                cam, self.scene.gaussians, self.args.pipe, self.background
            )
            image, viewspace_point_tensor, visibility_filter, radii = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
            )

            gt_image = viewpoint["image"][0].permute(2, 0, 1).cuda()
            Ll1 = l1_loss(image, gt_image)
            psnr_value = 20 * torch.log10(1.0 / torch.sqrt(Ll1**2))
            loss = (
                1.0 - self.args.opt.lambda_dssim
            ) * Ll1 + self.args.opt.lambda_dssim * (1.0 - ssim(image, gt_image))

            # regularization
            lambda_normal = self.args.opt.lambda_normal if iteration > 7000 else 0.0
            lambda_dist = (
                self.args.opt.lambda_dist
                if iteration > self.args.start_dist_iter
                else 0.0
            )

            render_depth = render_pkg["surf_depth"]
            if self.args.mono_depth:
                if render_depth.shape[1] < 540 or render_depth.shape[2] < 960:
                    render_depth = render_depth[
                        0,
                        : self.args.diffusion_crop_height // 2,
                        : self.args.diffusion_crop_width // 2,
                    ]
                else:
                    render_depth = render_depth[
                        0,
                        : self.args.diffusion_crop_height,
                        : self.args.diffusion_crop_width,
                    ]
                mono_depth = viewpoint["mono_depth"].float().cuda()
                mono_depth = mono_depth.squeeze(0).squeeze(0)
                mono_loss = 1.0 * self.depth_loss_fun(
                    render_depth, mono_depth
                )  # .mean()

            rend_dist = render_pkg["rend_dist"]
            rend_normal = render_pkg["rend_normal"]
            surf_normal = render_pkg["surf_normal"]
            normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
            normal_loss = lambda_normal * (normal_error).mean()
            dist_loss = lambda_dist * (rend_dist).mean()

            # loss
            total_loss = loss  # + dist_loss + normal_loss
            if self.args.outpaint_type == "sparse":
                total_loss += dist_loss + normal_loss
            if self.args.mono_depth:
                total_loss += mono_loss

            # try:
            if (
                self.args.repair
                and iteration > self.args.start_diffusion_iter
                and iteration < self.args.diffusion_until
            ):
                reg_data = next(self.scene.regset)
                reg_gt_image = reg_data["image"].permute(2, 0, 1).cuda()
                reg_gt_depth = reg_data["mono_depth"].permute(2, 0, 1).cuda()
                cam_info = reg_data["cam_info"]
                reg_cam = SampleCamera(cam_info)
                reg_render_pkg = render(
                    reg_cam, self.scene.gaussians, self.args.pipe, self.background
                )
                viewspace_point_tensor, visibility_filter, radii = (
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                )
                reg_image = reg_render_pkg["render"]
                reg_depth = reg_render_pkg["surf_depth"]
                reg_image = torch.nn.functional.interpolate(
                    reg_image.unsqueeze(0),
                    size=(reg_gt_image.shape[1], reg_gt_image.shape[2]),
                    mode="bilinear",
                ).squeeze(0)
                reg_depth = torch.nn.functional.interpolate(
                    reg_depth.unsqueeze(0),
                    size=(reg_gt_depth.shape[1], reg_gt_depth.shape[2]),
                    mode="nearest",
                ).squeeze(0)
                Ll1 = l1_loss(reg_image, reg_gt_image)
                psnr_value = 20 * torch.log10(1.0 / torch.sqrt(Ll1**2))
                reg_loss = (
                    1.0 - self.args.opt.lambda_dssim
                ) * Ll1 + self.args.opt.lambda_dssim * (
                    1.0 - ssim(reg_image, reg_gt_image)
                )
                reg_depth_loss = self.depth_loss_fun(reg_depth, reg_gt_depth)
                reg_weight = self.args.lambda_reg * np.sin(
                    (iteration - self.args.start_diffusion_iter)
                    / (self.args.diffusion_until - self.args.start_diffusion_iter)
                    * np.pi
                )
                total_loss += reg_weight * (reg_loss + reg_depth_loss)
                if self.args.outpaint_type == "sparse":
                    rend_dist = reg_render_pkg["rend_dist"]
                    rend_normal = reg_render_pkg["rend_normal"]
                    surf_normal = reg_render_pkg["surf_normal"]
                    normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
                    normal_loss = lambda_normal * (normal_error).mean()
                    dist_loss = lambda_dist * (rend_dist).mean()
                    total_loss += normal_loss + dist_loss

            total_loss.backward()
            iter_end.record()
            with torch.no_grad():
                self.ema_loss_for_log = 0.4 * loss.item() + 0.6 * self.ema_loss_for_log
                self.ema_dist_for_log = (
                    0.4 * dist_loss.item() + 0.6 * self.ema_dist_for_log
                )
                self.ema_normal_for_log = (
                    0.4 * normal_loss.item() + 0.6 * self.ema_normal_for_log
                )

                if iteration % 10 == 0:
                    loss_dict = {
                        "Loss": f"{self.ema_loss_for_log:.{5}f}",
                        "PSNR": f"{psnr_value:.{2}f}",
                        "distort": f"{self.ema_dist_for_log:.{5}f}",
                        "normal": f"{self.ema_normal_for_log:.{5}f}",
                        "Points": f"{len(self.scene.gaussians.get_xyz)}",
                    }
                    progress_bar.set_postfix(loss_dict)

                    progress_bar.update(10)
                if iteration == end_iter:
                    progress_bar.close()

                # Log and save
                if self.tb_writer is not None:
                    self.tb_writer.add_scalar(
                        "train_loss_patches/dist_loss", self.ema_dist_for_log, iteration
                    )
                    self.tb_writer.add_scalar(
                        "train_loss_patches/normal_loss",
                        self.ema_normal_for_log,
                        iteration,
                    )

                training_report(
                    self.tb_writer,
                    iteration,
                    Ll1,
                    loss,
                    l1_loss,
                    iter_start.elapsed_time(iter_end),
                    self.args.test_iterations,
                    self.scene,
                    render,
                    (self.args.pipe, self.background),
                )
                if iteration in self.args.save_iterations:
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    self.scene.save(iteration)

                # Densification
                if iteration < self.args.opt.densify_until_iter:
                    self.scene.gaussians.max_radii2D[visibility_filter] = torch.max(
                        self.scene.gaussians.max_radii2D[visibility_filter],
                        radii[visibility_filter],
                    )
                    self.scene.gaussians.add_densification_stats(
                        viewspace_point_tensor,
                        visibility_filter,
                        iteration,
                        self.args.opt.iterations,
                    )

                    if (
                        iteration > self.args.opt.densify_from_iter
                        and iteration % self.args.opt.densification_interval == 0
                    ):
                        size_threshold = (
                            20
                            if iteration > self.args.opt.opacity_reset_interval
                            else None
                        )
                        self.scene.gaussians.densify_and_prune(
                            self.args.opt.densify_grad_threshold,
                            self.args.opt.opacity_cull,
                            self.scene.cameras_extent,
                            size_threshold,
                        )

                    if iteration % self.args.opt.opacity_reset_interval == 0 or (
                        self.args.model.white_background
                        and iteration == self.args.opt.densify_from_iter
                    ):
                        self.scene.gaussians.reset_opacity()

                # Optimizer step
                if iteration < self.args.opt.iterations:
                    self.scene.gaussians.optimizer.step()
                    self.scene.gaussians.optimizer.zero_grad(set_to_none=True)

                if iteration in self.args.checkpoint_iterations:
                    print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    torch.save(
                        (self.scene.gaussians.capture(), iteration),
                        self.scene.model_path + "/chkpnt" + str(iteration) + ".pth",
                    )

            with torch.no_grad():
                if network_gui.conn is None:
                    network_gui.try_connect(self.args.model.render_items)
                while network_gui.conn is not None:
                    try:
                        net_image_bytes = None
                        (
                            custom_cam,
                            do_training,
                            keep_alive,
                            scaling_modifer,
                            render_mode,
                        ) = network_gui.receive()
                        if custom_cam is not None:
                            render_pkg = render(
                                custom_cam,
                                self.scene.gaussians,
                                self.args.pipe,
                                self.background,
                                scaling_modifer,
                            )
                            net_image = render_net_image(
                                render_pkg,
                                self.args.model.render_items,
                                render_mode,
                                custom_cam,
                            )
                            net_image_bytes = memoryview(
                                (torch.clamp(net_image, min=0, max=1.0) * 255)
                                .byte()
                                .permute(1, 2, 0)
                                .contiguous()
                                .cpu()
                                .numpy()
                            )
                        metrics_dict = {
                            "#": self.scene.gaussians.get_opacity.shape[0],
                            "loss": self.ema_loss_for_log,
                        }
                        # Send the data
                        network_gui.send(
                            net_image_bytes, self.args.model.source_path, metrics_dict
                        )
                        if do_training and (
                            (iteration < int(self.args.opt.iterations))
                            or not keep_alive
                        ):
                            break
                    except Exception:
                        # raise e
                        network_gui.conn = None

    def run_with_repair(self):
        # extrapolator = None
        sample_steps = [0] + list(
            range(
                self.args.start_diffusion_iter,
                self.args.diffusion_until + 1,
                self.args.diffusion_every,
            )
        )
        print("sample steps", sample_steps)
        print(f"Sample steps: {sample_steps}")
        # render_masks = []
        for i in range(len(sample_steps) - 1):
            start_iter = sample_steps[i]
            end_iter = sample_steps[i + 1]
            if start_iter >= self.args.diffusion_until:
                self.scene.save(start_iter)
                break

            self.train(start_iter, min(end_iter, self.args.iterations))
            print(f"Finished training from {start_iter} to {end_iter}")
            if end_iter >= self.args.iterations:
                self.scene.save(self.args.iterations)
                break
            self.scene.regset.clear()
            outpaint = None
            if self.args.outpaint_type == "crop":
                outpaint = OutpaintCrop(self.extrapolator, self.scene, self.args)
            elif self.args.outpaint_type == "sparse":
                outpaint = OutpaintSparse(self.extrapolator, self.scene, self.args)
            elif self.args.outpaint_type == "rotation":
                outpaint = OutpaintRotation(self.extrapolator, self.scene, self.args)

            outpaint.run(i)
        if sample_steps[-1] < self.args.iterations:
            self.train(sample_steps[-1], self.args.iterations)

        print("Saving final checkpoint.")
        torch.save(
            (self.scene.gaussians.capture(), self.args.iterations),
            self.scene.model_path + "/chkpnt" + str(self.args.iterations) + ".pth",
        )


    def run(self):
        if not self.args.repair:
            self.train(0, self.args.iterations)
        else:
            self.run_with_repair()

    def eval(self):
        all_configs = (
            {
                "name": "train",
                "cameras": [self.scene.getTrainInstant() for _ in range(len(self.scene.trainset))],
            },
            {
                "name": "test",
                "cameras": [self.scene.getTestInstant() for _ in range(len(self.scene.valset))],
            },
        )

        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
        lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).cuda()

        # load in ground truth data
        gt_dict = {}
        if "scannetplusplus" in args.data_dir:
            scene_id = os.path.basename(args.data_dir)

            # load intrinsics from COLMAP format
            intrinsics_file = f'{args.data_dir}/colmap/cameras.txt'
            colmap_camera = list(read_cameras_text(intrinsics_file).values())[0]
            pinhole_camera = undistorted_camera(colmap_camera)
            intrinsics = camera_to_intrinsic(pinhole_camera)

            # rescale intrinsics to match loaded image size
            height, width = pinhole_camera.height, pinhole_camera.width
            s_x = 480.0 / float(width)
            s_y = 360.0 / float(height)
            intrinsics[0, 0] *= s_x
            intrinsics[1, 1] *= s_y
            intrinsics[0, 2] = 480.0 / 2.
            intrinsics[1, 2] = 360.0 / 2.

            partial_point_cloud = []
            gt_backprojected_point_cloud = []
            with h5py.File(os.path.join(args.data_dir, "downsampled_data", f"{scene_id}.h5"), "r") as f:
                for config in all_configs:
                    if config["cameras"] and len(config["cameras"]) > 0:
                        for idx, viewpoint in enumerate(config["cameras"]):
                            cam = Camera(viewpoint["cam_info"])
                            print(cam.image_name)
                            image_idx = cam.image_name.split(".")[0]

                            # load pose
                            c2w = np.loadtxt(os.path.join(args.data_dir, "pose", f"{image_idx}.txt"))

                            # load image
                            gt_image = f['render_rgb'][image_idx][()].astype(np.float32) / 255.0

                            # load depth
                            gt_depth = f['render_depth'][image_idx][()].astype(np.float32) / 1000.0

                            gt_dict[cam.image_name] = {
                                "image": gt_image,
                                "depth": gt_depth,
                                "test_camera": True if config["name"] == "test" else False,
                            }

                            # backproject depth
                            view_points = backproject(gt_depth, c2w, intrinsics)
                            gt_backprojected_point_cloud.append(view_points)
                            if config["name"] == "train":
                                partial_point_cloud.append(view_points)
            gt_backprojected_point_cloud = np.concatenate(gt_backprojected_point_cloud, axis=0)
            partial_point_cloud = np.concatenate(partial_point_cloud, axis=0)

            # load ground truth complete point cloud
            gt_complete_point_cloud_data = np.load(os.path.join(args.data_dir, "complete_scene.npz"))
            gt_complete_point_cloud = gt_complete_point_cloud_data["points"]
            gt_complete_point_cloud[:, 2] *= -1
            # Rotate the pose -90 degrees about the x-axis (in radians)
            angle = np.pi / 2
            Rx = np.array([
                [1, 0, 0],
                [0, np.cos(angle), -np.sin(angle)],
                [0, np.sin(angle),  np.cos(angle)]
            ])
            gt_complete_point_cloud = gt_complete_point_cloud @ Rx.T
            gt_complete_point_cloud = filter_complete_points(gt_backprojected_point_cloud, gt_complete_point_cloud)

        elif "threedfront" in args.data_dir:
            # load intrinsics
            intrinsics = np.load(os.path.join(args.data_dir, "cam_K.npy"))

            partial_point_cloud = []
            gt_backprojected_point_cloud = []
            for config in all_configs:
                if config["cameras"] and len(config["cameras"]) > 0:
                    for idx, viewpoint in enumerate(config["cameras"]):
                        cam = Camera(viewpoint["cam_info"])
                
                        with h5py.File(os.path.join(args.data_dir, "data", cam.image_name), "r") as f:
                            # load pose
                            c2w = np.array(f["cam_Ts"])

                            # blender camera to opengl camera
                            y_z_swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
                            c2w = y_z_swap.dot(c2w)
                            c2w[:3, 0] *= -1

                            # opengl camera to opencv (COLMAP) camera
                            c2w[:3, 1] *= -1
                            c2w[:3, 2] *= -1

                            # Rotate the pose -90 degrees about the x-axis (in radians)
                            angle = np.pi / 2
                            Rx = np.array([
                                [1, 0, 0, 0],
                                [0, np.cos(angle), -np.sin(angle), 0],
                                [0, np.sin(angle),  np.cos(angle), 0],
                                [0, 0, 0, 1]
                            ])
                            c2w = Rx @ c2w

                            # load image
                            gt_image = np.array(f["colors"])[:,::-1].astype(np.float32) / 255.0

                            # load depth
                            gt_depth = np.array(f["depth"])[:, ::-1].astype(np.float32)
                            gt_depth[gt_depth >= 10000000000.0] = 0.0

                            gt_dict[cam.image_name] = {
                                "image": gt_image,
                                "depth": gt_depth,
                                "test_camera": True if config["name"] == "test" else False,
                            }

                            # backproject depth
                            view_points = backproject(gt_depth, c2w, intrinsics)
                            gt_backprojected_point_cloud.append(view_points)
                            if config["name"] == "train":
                                partial_point_cloud.append(view_points)
            gt_backprojected_point_cloud = np.concatenate(gt_backprojected_point_cloud, axis=0)
            partial_point_cloud = np.concatenate(partial_point_cloud, axis=0)

            # load in ground truth complete point cloud
            gt_complete_point_cloud_data = np.load(os.path.join(args.data_dir, "scene.npz"))
            gt_complete_point_cloud = gt_complete_point_cloud_data["points"]
            gt_complete_point_cloud[:, 2] *= -1
            # Rotate the pose -90 degrees about the x-axis (in radians)
            angle = np.pi / 2
            Rx = np.array([
                [1, 0, 0],
                [0, np.cos(angle), -np.sin(angle)],
                [0, np.sin(angle),  np.cos(angle)]
            ])
            gt_complete_point_cloud = gt_complete_point_cloud @ Rx.T
            gt_complete_point_cloud = filter_complete_points(gt_backprojected_point_cloud, gt_complete_point_cloud)

        save_dir = os.path.join(self.args.model.model_path, "renders")
        os.makedirs(save_dir, exist_ok=True)

        test_depths = []
        for cam in gt_dict.values():
            if cam["test_camera"]:
                test_depths.append(cam["depth"])
        test_depths = np.stack(test_depths, axis=0)
        max_depth = np.max(test_depths)
        min_depth = np.min(test_depths)

        # evaluate
        rendered_images = []
        rendered_depths = []
        gt_images = []
        gt_depths = []
        for config in all_configs:
            if config["name"] == "train":  # skip train views for evaluation
                continue

            if config["cameras"] and len(config["cameras"]) > 0:
                for idx, viewpoint in enumerate(config["cameras"]):
                    cam = Camera(viewpoint["cam_info"])
                    print(cam.image_name)

                    # render test view
                    render_pkg = render(cam, self.scene.gaussians, self.args.pipe, self.background)
                    image_tensor = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    depth_tensor = render_pkg["surf_depth"]
                    normal_tensor = render_pkg["rend_normal"]
                    depth_normal_tensor = render_pkg["surf_normal"]

                    # resize rendered views
                    image_tensor = torch.nn.functional.interpolate(
                        image_tensor.unsqueeze(0), size=(360, 480), mode="bicubic"
                    )
                    depth_tensor = torch.nn.functional.interpolate(
                        depth_tensor.unsqueeze(0), size=(360, 480), mode="nearest"
                    )
                    normal_tensor = torch.nn.functional.interpolate(
                        normal_tensor.unsqueeze(0), size=(360, 480), mode="nearest"
                    )
                    depth_normal_tensor = torch.nn.functional.interpolate(
                        depth_normal_tensor.unsqueeze(0), size=(360, 480), mode="nearest"
                    )

                    # process and save rendered image
                    image_tensor = image_tensor.clamp(0.0, 1.0)
                    image = image_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()
                    image = (image * 255).astype(np.uint8)

                    # process and save ground truth image
                    gt_image = gt_dict[cam.image_name]["image"]
                    gt_image_tensor = torch.tensor(gt_image).cuda().permute(2, 0, 1).unsqueeze(0)
                    gt_image = (gt_image * 255).astype(np.uint8)

                    # process and save rendered depth
                    depth = depth_tensor.squeeze().detach().cpu().numpy()
                    depth_image = (255 * (depth - min_depth) / (max_depth - min_depth)).astype(np.uint8)

                    # process and save ground truth depth
                    gt_depth = gt_dict[cam.image_name]["depth"]
                    gt_depth_image = (255 * (gt_depth - min_depth) / (max_depth - min_depth)).astype(np.uint8)

                    # process and save rendered normal
                    normal = normal_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()
                    normal_image = (255 * (normal + 1) / 2).astype(np.uint8)
                    depth_normal = depth_normal_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()
                    depth_normal_image = (255 * (depth_normal + 1) / 2).astype(np.uint8)

                    # save out views
                    image_idx = cam.image_name.split(".")[0]
                    # imageio.imwrite(os.path.join(save_dir, f"rendered_{image_idx}.png"), image)
                    # imageio.imwrite(os.path.join(save_dir, f"gt_{image_idx}.png"), gt_image)
                    # imageio.imwrite(os.path.join(save_dir, f"rendered_{image_idx}_depth.png"), depth_image)
                    # imageio.imwrite(os.path.join(save_dir, f"gt_{image_idx}_depth.png"), gt_depth_image)
                    # imageio.imwrite(os.path.join(save_dir, f"rendered_{image_idx}_normal.png"), normal_image)
                    # imageio.imwrite(os.path.join(save_dir, f"rendered_{image_idx}_depth_normal.png"), depth_normal_image)

                    rendered_images.append(image_tensor)
                    rendered_depths.append(depth)
                    gt_images.append(gt_image_tensor)
                    gt_depths.append(gt_depth)

    
        # prepare prediction
        predicted_point_cloud = self.scene.gaussians.get_xyz.detach().cpu().numpy()
        transform = np.linalg.inv(self.scene.valset.parser.transform)
        predicted_point_cloud = predicted_point_cloud @ transform[:3, :3].T + transform[:3, 3]
        # partial_point_cloud_mesh = trimesh.PointCloud(partial_point_cloud, np.tile([0, 0, 255], (partial_point_cloud.shape[0], 1)))
        # predicted_point_cloud_mesh = trimesh.PointCloud(predicted_point_cloud, np.tile([255, 0, 0], (predicted_point_cloud.shape[0], 1)))
        # trimesh.Scene([partial_point_cloud_mesh, predicted_point_cloud_mesh]).show()
        predicted_point_cloud = filter_complete_points(gt_backprojected_point_cloud, predicted_point_cloud)
        idx = voxelize(predicted_point_cloud, voxel_size=0.02)
        predicted_point_cloud = torch.tensor(predicted_point_cloud[idx]).to("cuda").float()
        predicted_point_cloud = predicted_point_cloud.view(1, -1, 3)

        # prepare ground truth
        gt_point_cloud = gt_complete_point_cloud
        idx = voxelize(gt_point_cloud, voxel_size=0.02)
        gt_point_cloud = torch.tensor(gt_point_cloud[idx]).to("cuda").float()
        gt_point_cloud = gt_point_cloud.view(1, -1, 3)

        # load partial point cloud
        idx = voxelize(partial_point_cloud, voxel_size=0.02)
        partial_point_cloud = torch.tensor(partial_point_cloud[idx]).to("cuda").float()
        partial_point_cloud = partial_point_cloud.view(1, -1, 3)

        # prepare missing
        dists, nn_idx, _ = knn_points(gt_point_cloud, partial_point_cloud, K=1)
        dists = dists.squeeze(-1) ** 0.5
        missing_point_cloud = gt_point_cloud[dists > 0.02].reshape(1, -1, 3)

        # Compute Chamfer Distance
        cd, _ = chamfer_distance(gt_point_cloud, predicted_point_cloud)
        cd = cd.item()

        # Compute Partial One-Sided Chamfer Distance
        partial_cd, _ = chamfer_distance(partial_point_cloud, predicted_point_cloud, single_directional=True)
        partial_cd = partial_cd.item()

        # Compute Complete One-Sided Chamfer Distance
        complete_cd, _ = chamfer_distance(missing_point_cloud, predicted_point_cloud, single_directional=True)
        complete_cd = complete_cd.item()

        # PSNR
        rendered_images = torch.concatenate(rendered_images, dim=0)
        gt_images = torch.concatenate(gt_images, dim=0)
        mse = F.mse_loss(rendered_images, gt_images, reduction='none')
        mse_per_image = mse.reshape(mse.shape[0], -1).mean(dim=1)
        psnr_per_image = -10 * torch.log10(mse_per_image + 1e-8)
        psnr = psnr_per_image.mean().item()

        # SSIM
        ssim_value = ssim(rendered_images, gt_images)
        ssim_value = ssim_value.item()

        # LPIPS
        lpips_value = lpips(rendered_images, gt_images)
        lpips_value = lpips_value.item()

        # L1 Depth Error
        rendered_depths = np.stack(rendered_depths, axis=0)
        gt_depths = np.stack(gt_depths, axis=0)
        l1_depth = np.mean(np.abs(rendered_depths[gt_depths > 0] - gt_depths[gt_depths > 0]))

        print(f"Evaluation Metrics:\n"
              f"PSNR: {psnr:.3f} dB\n"
              f"SSIM: {ssim_value:.4f}\n"
              f"LPIPS: {lpips_value:.4f}\n"
              f"L1 Depth: {l1_depth: .4f}\n"
              f"Chamfer Distance: {cd:.4f}\n"
              f"Partial One-Sided Chamfer Distance: {partial_cd:.4f}\n"
              f"Complete One-Sided Chamfer Distance: {complete_cd:.4f}")


def save_args_to_file(args, filepath):
    with open(filepath, "wb") as f:  # Note: using 'wb' for binary write mode
        pickle.dump(args, f)


def prepare_output_and_logger(args):
    if not args.model.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model.model_path = os.path.join("./output/", unique_str)

    # Set up output folder
    print("Output folder: {}".format(args.model.model_path))
    os.makedirs(args.model.model_path, exist_ok=True)
    print(args)
    save_args_to_file(args, os.path.join(args.model.model_path, "cfg_args"))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


@torch.no_grad()
def training_report(
    tb_writer,
    iteration,
    Ll1,
    loss,
    l1_loss,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    renderArgs,
):
    if tb_writer:
        tb_writer.add_scalar("train_loss_patches/reg_loss", Ll1.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/total_loss", loss.item(), iteration)
        tb_writer.add_scalar("iter_time", elapsed, iteration)
        tb_writer.add_scalar(
            "total_points", scene.gaussians.get_xyz.shape[0], iteration
        )

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {
                "name": "test",
                "cameras": [scene.getTestInstant() for _ in range(len(scene.valset))],
            },
            {
                "name": "train",
                "cameras": [scene.getTrainInstant() for _ in range(5, 30, 5)],
            },
        )

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config["cameras"]):
                    # print(config['name'], viewpoint['cam_info'].uid)
                    cam = Camera(viewpoint["cam_info"])
                    render_pkg = renderFunc(cam, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(
                        viewpoint["image"][0].permute(2, 0, 1).cuda(), 0.0, 1.0
                    )
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap

                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap="turbo")
                        tb_writer.add_images(
                            config["name"] + "_view_{}/depth".format(cam.image_name),
                            depth[None],
                            global_step=iteration,
                        )
                        tb_writer.add_images(
                            config["name"] + "_view_{}/render".format(cam.image_name),
                            image[None],
                            global_step=iteration,
                        )

                        try:
                            rend_alpha = render_pkg["rend_alpha"]
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(
                                config["name"]
                                + "_view_{}/rend_normal".format(cam.image_name),
                                rend_normal[None],
                                global_step=iteration,
                            )
                            tb_writer.add_images(
                                config["name"]
                                + "_view_{}/surf_normal".format(cam.image_name),
                                surf_normal[None],
                                global_step=iteration,
                            )
                            tb_writer.add_images(
                                config["name"]
                                + "_view_{}/rend_alpha".format(cam.image_name),
                                rend_alpha[None],
                                global_step=iteration,
                            )

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(
                                config["name"]
                                + "_view_{}/rend_dist".format(cam.image_name),
                                rend_dist[None],
                                global_step=iteration,
                            )
                        except Exception:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                config["name"]
                                + "_view_{}/ground_truth".format(cam.image_name),
                                gt_image[None],
                                global_step=iteration,
                            )

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config["cameras"])
                l1_test /= len(config["cameras"])
                print(
                    "\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(
                        iteration, config["name"], l1_test, psnr_test
                    )
                )
                if tb_writer:
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - l1_loss", l1_test, iteration
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - psnr", psnr_test, iteration
                    )

        torch.cuda.empty_cache()


if __name__ == "__main__":
    args = config_args()
    print("Optimizing " + args.model.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    train_manager = TrainManager(args)
    if args.render_test_views:
        train_manager.eval()
    else:
        train_manager.run()

    # All done
    print("\nTraining complete.")
