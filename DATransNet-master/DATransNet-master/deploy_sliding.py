"""
C++-style dynamic sliding-window deployment for DATransNet.

Per-patch pipeline: preprocess -> infer -> fuse -> postprocess -> draw.

Example (1280x1280, 512 patch, stride 256):
    python deploy_sliding.py ^
        --input_dir  ./data/my_defect/images ^
        --output_dir ./result/deploy ^
        --checkpoint ./log5/IRSTD-1K/DATransNet/400.pth.tar ^
        --patch_size 512 --stride 256 ^
        --preprocess_mode none

With defect enhancement (mixed point+line):
    python deploy_sliding.py ^
        --input_dir  M:/压印/images ^
        --output_dir M:/压印/deploy_out ^
        --checkpoint ./weights/point_512.pth.tar ^
        --preprocess_mode mixed ^
        --threshold 0.45 ^
        --edge_margin 35 --min_area 15
"""
import argparse
import os

from utils.deploy_pipeline import SlidingConfig, SlidingWindowEngine
from utils.utils import seed_pytorch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def parse_args():
    p = argparse.ArgumentParser(description='C++-style sliding-window deployment')
    p.add_argument('--checkpoint', required=True, help='.pth.tar model weights')
    p.add_argument('--input_dir', required=True, help='Folder with input images')
    p.add_argument('--output_dir', default='./result/deploy', help='Output folder')
    p.add_argument('--model_name', default='DATransNet')
    p.add_argument('--img_size', type=int, default=512, help='Model img_size (match training)')
    p.add_argument('--patch_size', type=int, default=512)
    p.add_argument('--stride', type=int, default=256,
                   help='Stride; 256 => 50%% overlap on 1280 images (16 windows)')
    p.add_argument('--blend', default='gaussian', choices=['gaussian', 'uniform'])
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--preprocess_mode', default='none',
                   choices=['none', 'point', 'line', 'mixed'],
                   help='Per-patch defect preprocess (like C++ ROI enhance)')
    p.add_argument('--input_channels', type=int, default=1,
                   help='1=grayscale (DATransNet default), 3=BGR (custom model)')
    p.add_argument('--batch_size', type=int, default=4, help='Patch batch size for GPU')
    p.add_argument('--edge_margin', type=int, default=0,
                   help='Drop defects within N px of image border')
    p.add_argument('--min_area', type=int, default=0, help='Min defect area in px^2')
    p.add_argument('--max_area', type=int, default=0, help='Max defect area in px^2')
    p.add_argument('--max_aspect', type=float, default=0.0,
                   help='Max aspect ratio filter (0=off, e.g. 3.0 for point-only)')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no_vis', action='store_true', help='Skip visualization output')
    return p.parse_args()


def main():
    args = parse_args()
    seed_pytorch(args.seed)

    if args.patch_size != args.img_size:
        print(f'[warn] patch_size={args.patch_size} != img_size={args.img_size}, '
              f'model expects img_size={args.img_size}')

    config = SlidingConfig(
        patch_size=args.patch_size,
        stride=args.stride,
        blend=args.blend,
        threshold=args.threshold,
        img_size=args.img_size,
        model_name=args.model_name,
        preprocess_mode=args.preprocess_mode,
        input_channels=args.input_channels,
        batch_size=args.batch_size,
        edge_margin=args.edge_margin,
        min_area=args.min_area,
        max_area=args.max_area,
        max_aspect=args.max_aspect,
    )

    print('=== Sliding Window Deploy ===')
    print(f'  model      : {config.model_name}')
    print(f'  patch/stride: {config.patch_size}/{config.stride}  blend={config.blend}')
    print(f'  preprocess : {config.preprocess_mode}  channels={config.input_channels}')
    print(f'  threshold  : {config.threshold}')
    if config.edge_margin or config.min_area or config.max_aspect:
        print(f'  post-filter: margin={config.edge_margin}  '
              f'min_area={config.min_area}  max_area={config.max_area}  '
              f'max_aspect={config.max_aspect}')

    engine = SlidingWindowEngine.from_checkpoint(
        checkpoint=args.checkpoint,
        config=config,
        device=args.device,
    )

    engine.infer_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        save_vis=not args.no_vis,
    )
    print(f'Done. Results in {args.output_dir}')


if __name__ == '__main__':
    main()
