from argparse import ArgumentParser
from arguments import DataParams, ModelParams, PipelineParams, OptimizationParams
import sys


def config_args():
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    dp = DataParams(parser)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6012)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[10_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--repair", action="store_true")
    parser.add_argument("--mono_depth", action="store_true")
    parser.add_argument("--diffusion_ckpt", type=str, default=None)
    parser.add_argument("--diffusion_config", type=str, default=None)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--diffusion_resize_width", type=int, default=512)
    parser.add_argument("--diffusion_resize_height", type=int, default=320)
    parser.add_argument("--diffusion_crop_width", type=int, default=820)
    parser.add_argument("--diffusion_crop_height", type=int, default=512)
    parser.add_argument("--start_diffusion_iter", type=int, default=1000)
    parser.add_argument("--diffusion_every", type=int, default=1000)
    parser.add_argument("--diffusion_until", type=int, default=7000)
    parser.add_argument(
        "--outpaint_type",
        type=str,
        choices=["crop", "segment", "sparse", "rotation"],
        default="crop",
    )
    parser.add_argument("--add_indices", type=int, nargs="+", default=[])
    parser.add_argument("--initize_points", action="store_true")
    parser.add_argument("--downsample_factor", type=int, default=1)
    parser.add_argument("--wo_crop", action="store_true")
    parser.add_argument("--distance", type=float, default=1.8)
    parser.add_argument("--position_z_offset", type=float, default=0.0)
    parser.add_argument("--rotation_angle", type=float, default=16.0)
    parser.add_argument("--path_scale", type=float, default=0.7)
    parser.add_argument("--camera_path_file", type=str, default=None)
    parser.add_argument("--unconditional_guidance_scale", type=float, default=3.2)
    parser.add_argument("--lambda_reg", type=float, default=1.0)
    parser.add_argument("--start_dist_iter", type=int, default=5000)
    parser.add_argument("--render_test_views", action="store_true")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args.dataset = dp.extract(args)
    args.model = lp.extract(args)
    args.opt = op.extract(args)
    args.pipe = pp.extract(args)
    return args
