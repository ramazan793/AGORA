"""
export_driving_sequence.py — Export FLAME driving sequence for browser inference.

Reads per-frame FLAME params from a smirk-processed video folder, applies
Savitzky-Golay smoothing, and writes a JSON file consumed by the browser viewer.

The smirk folder structure is:
  <video_path>/
    000/
      exp.npy        (50,)  expression codes
      jawpose.npy    (3,)   jaw pose
      eyelid.npy     (2,)   eyelid params
    001/
      ...

Each frame contributes a 55-dim vector: [exp(50), jawpose(3), eyelid(2)].

Usage:
  python export_driving_sequence.py \\
    --video-path /data2/ramazan.fazylov/media/dgghead_workspace/reenact_test_videos/obama_next3d/smirk/ \\
    --output-path scripts/deform_blendshapes_pca/web_inference/splat/data/driving_sequence.json

Obama video path (default):
  /data2/ramazan.fazylov/media/dgghead_workspace/reenact_test_videos/obama_next3d/smirk/
"""

import os
import sys
import json
import argparse
from glob import glob

import numpy as np
from scipy.signal import savgol_filter


def load_flame55_sequence(video_path):
    """
    Load per-frame 55-dim FLAME params from sorted smirk subfolders.
    Folder names are expected to be integers (e.g. '000', '001', ...).
    Returns np.ndarray of shape [T, 55].
    """
    frame_folders = sorted(
        glob(os.path.join(video_path, '*/')),
        key=lambda x: int(x.rstrip(os.sep).split(os.sep)[-1])
    )
    if not frame_folders:
        raise ValueError(
            f"No frame subfolders found under {video_path}.\n"
            f"Expected numbered subdirectories (e.g. 000/, 001/, ...) "
            f"each containing exp.npy, jawpose.npy, eyelid.npy."
        )

    flame55 = []
    missing = []
    for folder in frame_folders:
        exp_path    = os.path.join(folder, 'exp.npy')
        jaw_path    = os.path.join(folder, 'jawpose.npy')
        eyelid_path = os.path.join(folder, 'eyelid.npy')

        if not (os.path.exists(exp_path) and
                os.path.exists(jaw_path) and
                os.path.exists(eyelid_path)):
            missing.append(folder)
            continue

        exp    = np.load(exp_path).ravel()[:50]   # (50,)
        jaw    = np.load(jaw_path).ravel()[:3]    # (3,)
        eyelid = np.load(eyelid_path).ravel()[:2] # (2,)
        flame55.append(np.concatenate([exp, jaw, eyelid]).astype(np.float32))

    if missing:
        print(f"  WARNING: {len(missing)} folders missing params files (skipped)")
    if not flame55:
        raise ValueError("No valid frames found.")

    return np.array(flame55, dtype=np.float32)  # [T, 55]


def main():
    parser = argparse.ArgumentParser(
        description='Export FLAME driving sequence for browser 3DGS inference.'
    )
    parser.add_argument(
        '--video-path',
        default='/data2/ramazan.fazylov/media/dgghead_workspace/reenact_test_videos/obama_next3d/smirk/',
        help='Path to smirk-processed video folder (contains numbered subfolders)'
    )
    parser.add_argument(
        '--output-path',
        default='splat/data/driving_sequence.json',
        help='Output JSON file path (default: splat/data/driving_sequence.json)'
    )
    parser.add_argument(
        '--savgol-win', type=int, default=5,
        help='Savitzky-Golay filter window length (default: 5, must be odd and >= 4)'
    )
    parser.add_argument(
        '--fps', type=float, default=25.0,
        help='Playback FPS for browser animation (default: 25.0)'
    )
    parser.add_argument(
        '--max-frames', type=int, default=None,
        help='Truncate sequence to at most this many frames (optional)'
    )
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)

    # ── Load frames ───────────────────────────────────────────────────────────
    print(f"Loading FLAME params from  {args.video_path}")
    flame55 = load_flame55_sequence(args.video_path)  # [T, 55]
    T = flame55.shape[0]
    print(f"  Loaded {T} frames, 55-dim each")

    if args.max_frames is not None and T > args.max_frames:
        flame55 = flame55[:args.max_frames]
        T = args.max_frames
        print(f"  Truncated to {T} frames (--max-frames)")

    # ── Savitzky-Golay smoothing ──────────────────────────────────────────────
    win = args.savgol_win
    if win % 2 == 0:
        win += 1  # must be odd
    if T < win:
        win = max(3, T if T % 2 == 1 else T - 1)  # fallback for very short sequences
        print(f"  Adjusted savgol window to {win} (sequence too short)")

    polyorder = min(3, win - 1)
    flame55_smooth = savgol_filter(
        flame55, window_length=win, polyorder=polyorder, axis=0
    ).astype(np.float32)
    print(f"  Applied Savitzky-Golay smoothing (window={win}, polyorder={polyorder})")

    # ── Write JSON ────────────────────────────────────────────────────────────
    seq = {
        'num_frames': T,
        'params_dim': 55,
        'fps':        float(args.fps),
        'frames':     flame55_smooth.tolist(),   # list of T lists of 55 floats
    }
    with open(args.output_path, 'w') as f:
        json.dump(seq, f, separators=(',', ':'))  # compact encoding

    size_kb = os.path.getsize(args.output_path) / 1e3
    print(f"\nSaved driving_sequence.json: {T} frames × 55 params, {size_kb:.0f} KB")
    print(f"  Path: {args.output_path}")


if __name__ == '__main__':
    main()
