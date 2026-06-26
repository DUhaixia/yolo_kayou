"""
1280x1280 (or arbitrary size) sliding-window inference with 512 patches.

Example:
    python inference_sliding.py ^
        --input_dir ./data/my_defect/images ^
        --output_dir ./result/sliding ^
        --checkpoint ./log5/IRSTD-1K/DATransNet/400.pth.tar ^
        --patch_size 512 --stride 256
"""
import argparse
import os
import time

import cv2
import numpy as np
import scipy.io as scio
import torch

from net import Net
from utils.images import load_image
from utils.sliding_window import count_windows, sliding_window_infer
from utils.utils import seed_pytorch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


def list_images(input_dir):
    names = []
    for fname in sorted(os.listdir(input_dir)):
        if fname.lower().endswith(IMG_EXTENSIONS):
            names.append(fname)
    return names


def main():
    parser = argparse.ArgumentParser(description='Sliding-window inference for large images')
    parser.add_argument('--model_name', default='DATransNet', type=str)
    parser.add_argument('--checkpoint', required=True, type=str, help='Path to .pth.tar checkpoint')
    parser.add_argument('--input_dir', required=True, type=str, help='Folder with input images')
    parser.add_argument('--output_dir', default='./result/sliding', type=str)
    parser.add_argument('--patch_size', type=int, default=512, help='Sliding window size')
    parser.add_argument('--stride', type=int, default=256,
                        help='Stride between windows. 256 => 50%% overlap on 1280 images')
    parser.add_argument('--blend', default='gaussian', choices=['gaussian', 'uniform'])
    parser.add_argument('--threshold', type=float, default=0.5, help='Binarize prob map for mask output')
    parser.add_argument('--img_size', type=int, default=512,
                        help='DATransNet img_size, must match training patch size')
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    seed_pytorch(args.seed)
    device = args.device if torch.cuda.is_available() else 'cpu'

    os.makedirs(os.path.join(args.output_dir, 'prob'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'mask'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'mat'), exist_ok=True)

    net = Net(model_name=args.model_name, mode='test', size=args.img_size).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    net.load_state_dict(ckpt['state_dict'], strict=False)
    net.eval()

    image_names = list_images(args.input_dir)
    if not image_names:
        raise FileNotFoundError(f'No images found in {args.input_dir}')

    print(f'model={args.model_name}, patch={args.patch_size}, stride={args.stride}, blend={args.blend}')
    print(f'found {len(image_names)} images in {args.input_dir}')

    for fname in image_names:
        src = os.path.join(args.input_dir, fname)
        stem = os.path.splitext(fname)[0]

        img = load_image(src)
        img = np.array(img, dtype=np.float32) / 255.0
        h, w = img.shape

        n_win = count_windows(h, w, args.patch_size, args.stride)
        print(f'[{fname}] {h}x{w} -> {n_win} windows', end=' ')

        t0 = time.time()
        pred = sliding_window_infer(
            net, img,
            patch_size=args.patch_size,
            stride=args.stride,
            blend=args.blend,
            device=device,
        )
        elapsed = time.time() - t0
        print(f'done in {elapsed:.2f}s')

        prob_png = (pred * 255).astype(np.uint8)
        mask_png = ((pred > args.threshold) * 255).astype(np.uint8)

        cv2.imwrite(os.path.join(args.output_dir, 'prob', stem + '.png'), prob_png)
        cv2.imwrite(os.path.join(args.output_dir, 'mask', stem + '.png'), mask_png)
        scio.savemat(os.path.join(args.output_dir, 'mat', stem + '.mat'), {'T': pred})


if __name__ == '__main__':
    main()
