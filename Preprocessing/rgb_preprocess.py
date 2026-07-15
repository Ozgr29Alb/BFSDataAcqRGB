"""
rgb_preprocess.py

Preprocess a raw RGB video (AVI or MP4, any fps, any resolution) into
an HDF5 file whose sample structure matches fusion_preprocess.py output,
ready to be fused with the other modalities.

Protocol
--------
- All streams start at the hardware trigger.  fusion_preprocess.py bins
  data into consecutive windows of bin_length µs starting from t=0.
  Bin i covers  [i*bin_length, (i+1)*bin_length) µs.
- RGB frame 0 is at t=0, so the frame for bin i is the one whose
  timestamp is closest to the bin centre:
      fid = floor((i + 0.5) * bin_length / frame_interval_µs)
  floor() is used (not round) to avoid Python's banker's rounding,
  which causes duplicate frame selection when fps == output_hz.
- N is taken from an existing fusion H5 (--fusion_h5) or set directly
  (--n_samples).  Auto-computed from video length if neither is given.
- Video is read once sequentially and frames are written directly to H5
  as they arrive — no full-video frame cache.

Output H5 schema
----------------
  rgb   (N, seq_len, 3, out_H, out_W)  float32  [0.0, 1.0]  gzip-4
        out_H, out_W = --out_hw if given, else native video resolution.

Usage
-----
    # Existing session (matching fusion H5 available), resize to 720p:
    python rgb_preprocess.py \\
        --video      /Data/.../S2_D3_fast_val_rgb.avi \\
        --fusion_h5  /Data/.../fusion_ego_S2_D3_fast_val.h5 \\
        --output     /Data/.../rgb_S2_D3_fast_val.h5 \\
        --out_hw     720 1280 \\
        --save_mp4

    # New subject (no fusion H5 yet — derive N from video):
    python rgb_preprocess.py \\
        --video      /Data/.../S5_D2_arvr_test_rgb.mp4 \\
        --output     /Data/.../rgb_S5_D2_arvr_test.h5 \\
        --out_hw     720 1280

    # Quick smoke-test (first 100 samples only):
        ... --max_samples 100
"""

import argparse
import os
import sys
import cv2
import h5py
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime


def center_crop(frame_bgr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Center-crop (H, W, C) BGR frame to (out_h, out_w, C).
    Falls back to cv2.resize with INTER_AREA if source is smaller than target."""
    h, w = frame_bgr.shape[:2]
    if h < out_h or w < out_w:
        return cv2.resize(frame_bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)
    y0 = (h - out_h) // 2
    x0 = (w - out_w) // 2
    return frame_bgr[y0:y0 + out_h, x0:x0 + out_w]


def preprocess_rgb(
    video_path: str,
    output_h5_path: str,
    *,
    fusion_h5_path: str | None = None,
    n_samples_override: int | None = None,
    bin_length: int = 50_000,   # µs — must match fusion_preprocess.py
    seq_len: int = 1,
    max_samples: int | None = None,
    out_hw: tuple[int, int] | None = None,   # (H, W) after center crop; None = native
    save_mp4: bool = False,
    stretch_factor: float = 1.0,
    # stretch_factor > 1  →  fast-motion / time-lapse recording.
    # The video plays back FASTER than real time by this factor.
    # Example: video is 60s but represents 710s of real content → stretch_factor = 710/60 ≈ 11.85
    # Effect: frame_interval_us is multiplied by stretch_factor, so each bin maps
    # to a proportionally earlier frame. Pass --auto_stretch to compute from H5.
    auto_stretch: bool = False,
    # If True and a fusion_h5 is given, auto-compute stretch_factor as
    # n_h5_samples / video_samples_at_20hz when the ratio > 1.5.
    end_time: float | None = None,
    # Hard cut: ignore all video content after end_time seconds.
    # Use this to remove post-session relaxation / phone-checking at the end.
    # N is then derived from min(n_ref, floor(end_frame / frames_per_bin)).
) -> None:

    # ------------------------------------------------------------------
    # 1. Open video — must be first so we have fps/H/W for all decisions
    # ------------------------------------------------------------------
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: cannot open {video_path}", file=sys.stderr)
        sys.exit(1)

    src_fps           = cap.get(cv2.CAP_PROP_FPS)
    total_frames_raw  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_H             = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_W             = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_interval_us = 1_000_000.0 / src_fps
    out_hz            = 1_000_000.0 / bin_length
    frames_per_bin    = bin_length / frame_interval_us

    # Apply end_time cut before anything else
    if end_time is not None:
        end_frame    = min(total_frames_raw - 1, int(end_time * src_fps))
        total_frames = end_frame + 1
    else:
        total_frames = total_frames_raw

    out_H, out_W = out_hw if out_hw is not None else (src_H, src_W)

    print(f"\nVideo : {os.path.basename(video_path)}")
    print(f"  FPS           : {src_fps:.4f}  ({frame_interval_us:.2f} µs/frame)")
    if end_time is not None:
        print(f"  Total frames  : {total_frames_raw}  ({total_frames_raw/src_fps:.2f} s)  [full]")
        print(f"  Using frames  : {total_frames}  ({total_frames/src_fps:.2f} s)  [--end_time {end_time}s]")
    else:
        print(f"  Total frames  : {total_frames}  ({total_frames/src_fps:.2f} s)")
    print(f"  Source res    : {src_W} × {src_H}")
    if out_hw is not None:
        mode = "center crop" if src_H >= out_H and src_W >= out_W else "resize (INTER_AREA)"
        print(f"  Output res    : {out_W} × {out_H}  [{mode}]")
    else:
        print(f"  Output res    : {out_W} × {out_H}  [native]")
    print(f"  Output Hz     : {out_hz:.1f}  (bin_length={bin_length} µs)")
    print(f"  Frames/bin    : {frames_per_bin:.4f}")
    if abs(frames_per_bin - round(frames_per_bin)) > 0.05:
        print(f"  NOTE: frames/bin is not an integer ({frames_per_bin:.4f}). "
              f"Frame spacing will alternate between "
              f"{int(frames_per_bin)} and {int(frames_per_bin)+1} frames. "
              f"This is correct for non-integer ratios (e.g. 50fps → 20Hz = 2.5×).")

    # ------------------------------------------------------------------
    # 2. Determine N samples
    # ------------------------------------------------------------------
    if fusion_h5_path is not None:
        print(f"\nReference H5 (read-only): {os.path.basename(fusion_h5_path)}")
        with h5py.File(fusion_h5_path, "r") as f:
            ref_key = list(f.keys())[0]
            n_ref   = f[ref_key].shape[0]
        print(f"  Key '{ref_key}': {n_ref} samples  ({n_ref * bin_length / 1e6:.2f} s)")

    elif n_samples_override is not None:
        n_ref = n_samples_override
        print(f"\nN samples (--n_samples): {n_ref}  ({n_ref * bin_length / 1e6:.2f} s)")

    else:
        n_ref = int(total_frames / frames_per_bin)
        print(f"\nN samples (auto from video): {n_ref}  ({n_ref * bin_length / 1e6:.2f} s)")

    n_samples = min(n_ref, max_samples) if max_samples else n_ref

    duration_video_s = total_frames / src_fps
    duration_ref_s   = n_ref * bin_length / 1e6

    # ------------------------------------------------------------------
    # Auto-detect / apply stretch factor for fast-motion recordings.
    # A fast-motion (time-lapse) video is shorter than the session it covers:
    # the camera captured fewer frames per real second than its nominal fps.
    # stretch_factor = how many real seconds each playback second represents.
    # Corrected frame_interval: effective_interval = frame_interval_us * stretch_factor
    # → each bin maps to a proportionally earlier frame, spreading n_ref bins
    #   across total_frames instead of padding with the last frame.
    # ------------------------------------------------------------------
    video_samples_at_20hz = int(total_frames / frames_per_bin)
    if auto_stretch and n_ref > 0 and video_samples_at_20hz > 0:
        implied = n_ref / video_samples_at_20hz
        if implied > 1.5:
            stretch_factor = implied
            print(f"\n  AUTO-STRETCH: video has {video_samples_at_20hz} samples but H5 has "
                  f"{n_ref} → stretch_factor = {stretch_factor:.3f}")
            print(f"  Each video frame represents {stretch_factor:.2f}× real time.")
        else:
            print(f"\n  Auto-stretch: ratio {implied:.2f} < 1.5 — using stretch_factor=1.0")

    if stretch_factor != 1.0:
        effective_interval_us = frame_interval_us * stretch_factor
        print(f"\n  Stretch factor    : {stretch_factor:.4f}")
        print(f"  Effective fps     : {1e6/effective_interval_us:.4f} real fps")
        print(f"  Each frame covers : {effective_interval_us/1e3:.1f} ms of real time")
        print(f"  Bins per frame    : {effective_interval_us / bin_length:.2f}")
    else:
        effective_interval_us = frame_interval_us

    print(f"\n  Reference spans : {duration_ref_s:.2f} s  ({n_ref} samples)")
    print(f"  Video spans     : {duration_video_s:.2f} s  (playback)")
    print(f"  Will write      : {n_samples} samples")
    if stretch_factor == 1.0 and duration_video_s < duration_ref_s * 0.95:
        print(f"  WARNING: video is >5% shorter than the reference. "
              f"Last decoded frame will be repeated for missing bins.")

    # ------------------------------------------------------------------
    # 3. Pre-compute bin → frame mapping
    #
    #    fid = floor((i + 0.5) * bin_length / effective_interval_us)
    #    effective_interval_us = frame_interval_us * stretch_factor
    #    floor() avoids banker's rounding (Python round() rounds 0.5 → 0).
    #
    #    For normal video: effective_interval_us == frame_interval_us
    #    For fast-motion (stretch_factor > 1): frames are spread across all bins
    # ------------------------------------------------------------------
    frame_to_write: dict[int, list[tuple[int, int]]] = defaultdict(list)

    for sample_idx in range(n_samples):
        for s_i in range(seq_len):
            global_bin = sample_idx * seq_len + s_i
            centre_us  = (global_bin + 0.5) * bin_length
            fid        = int(centre_us / effective_interval_us)
            fid        = max(0, min(total_frames - 1, fid))
            frame_to_write[fid].append((sample_idx, s_i))

    last_needed_frame = max(frame_to_write.keys())
    print(f"\n  Unique frames needed : {len(frame_to_write)} / {total_frames}")
    print(f"  First / last frame   : {min(frame_to_write)} / {last_needed_frame}")

    # ------------------------------------------------------------------
    # 4. Create output H5 with pre-allocated dataset
    # ------------------------------------------------------------------
    if os.path.exists(output_h5_path):
        os.remove(output_h5_path)
    os.makedirs(os.path.dirname(os.path.abspath(output_h5_path)), exist_ok=True)

    out_h5 = h5py.File(output_h5_path, "w")
    out_h5.attrs["source_video"]   = os.path.basename(video_path)
    out_h5.attrs["reference_h5"]   = (os.path.basename(fusion_h5_path)
                                       if fusion_h5_path else "n/a")
    out_h5.attrs["source_fps"]     = float(src_fps)
    out_h5.attrs["bin_length_us"]  = bin_length
    out_h5.attrs["seq_len"]        = seq_len
    out_h5.attrs["output_hz"]      = float(out_hz)
    out_h5.attrs["channel_order"]  = "RGB"
    out_h5.attrs["value_range"]    = "[0.0, 1.0]"
    out_h5.attrs["out_resolution"] = f"{out_W}x{out_H}"
    out_h5.attrs["date_processed"] = datetime.now().isoformat()

    rgb_ds = out_h5.create_dataset(
        "rgb",
        shape=(n_samples, seq_len, 3, out_H, out_W),
        dtype=np.float32,
        chunks=(1, seq_len, 3, out_H, out_W),
        compression="gzip",
        compression_opts=4,
    )
    print(f"\n  H5 dataset pre-allocated: {rgb_ds.shape}  dtype=float32  gzip-4")

    # ------------------------------------------------------------------
    # 5. (Optional) Open MP4 writer
    #    Written to same folder as output H5, same basename with .mp4 ext.
    # ------------------------------------------------------------------
    mp4_writer = None
    if save_mp4:
        mp4_path = os.path.splitext(output_h5_path)[0] + ".mp4"
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        mp4_writer = cv2.VideoWriter(mp4_path, fourcc, float(out_hz), (out_W, out_H))
        if not mp4_writer.isOpened():
            print(f"  WARNING: could not open MP4 writer at {mp4_path}. "
                  f"MP4 will not be saved.", file=sys.stderr)
            mp4_writer = None
        else:
            print(f"  MP4 output     : {mp4_path}  ({out_hz:.1f} fps, {out_W}×{out_H})")

    # ------------------------------------------------------------------
    # 6. Read video once sequentially, write to H5 (and MP4) as frames arrive
    # ------------------------------------------------------------------
    print("\nReading and writing...")

    last_frame_chw  = None   # (3, out_H, out_W) float32 — fill fallback
    last_frame_bgr  = None   # (out_H, out_W, 3) uint8   — MP4 fill fallback
    cap_idx         = 0
    written_count   = 0
    partial: dict[int, dict[int, np.ndarray]] = {}

    with tqdm(total=len(frame_to_write), desc="Frames processed") as pbar:
        while cap_idx <= last_needed_frame:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            if cap_idx in frame_to_write:
                # Apply center crop / resize if requested
                if out_hw is not None:
                    frame_bgr = center_crop(frame_bgr, out_H, out_W)

                last_frame_bgr = frame_bgr
                last_frame_chw = (frame_bgr[:, :, ::-1]
                                  .transpose(2, 0, 1)
                                  .astype(np.float32) / 255.0)

                entries = frame_to_write[cap_idx]
                for sample_idx, s_i in entries:
                    if seq_len == 1:
                        rgb_ds[sample_idx, 0] = last_frame_chw
                        written_count += 1
                        if mp4_writer is not None:
                            mp4_writer.write(frame_bgr)
                    else:
                        if sample_idx not in partial:
                            partial[sample_idx] = {}
                        partial[sample_idx][s_i] = last_frame_chw
                        if len(partial[sample_idx]) == seq_len:
                            rgb_np = np.stack(
                                [partial[sample_idx][si] for si in range(seq_len)]
                            )
                            rgb_ds[sample_idx] = rgb_np
                            written_count += 1
                            del partial[sample_idx]
                            if mp4_writer is not None:
                                # Write the first sub-bin frame for the sample
                                mp4_writer.write(frame_bgr)

                pbar.update(1)

            cap_idx += 1

    cap.release()

    # ------------------------------------------------------------------
    # 7. Fill tail (video shorter than reference) and flush remaining partials
    # ------------------------------------------------------------------
    if last_frame_chw is not None:
        for sample_idx, sub in partial.items():
            for s_i in range(seq_len):
                rgb_ds[sample_idx, s_i] = sub.get(s_i, last_frame_chw)
            written_count += 1
            if mp4_writer is not None and last_frame_bgr is not None:
                mp4_writer.write(last_frame_bgr)

        actually_written = written_count
        if actually_written < n_samples:
            fill_count = n_samples - actually_written
            print(f"  NOTE: {fill_count} samples beyond video end — "
                  f"filling with last decoded frame.")
            for sample_idx in tqdm(range(actually_written, n_samples),
                                   desc="Filling tail"):
                for s_i in range(seq_len):
                    rgb_ds[sample_idx, s_i] = last_frame_chw
                if mp4_writer is not None and last_frame_bgr is not None:
                    mp4_writer.write(last_frame_bgr)
    else:
        if written_count < n_samples:
            print(f"  ERROR: video produced no frames — output is incomplete.",
                  file=sys.stderr)

    out_h5.close()
    if mp4_writer is not None:
        mp4_writer.release()

    size_gb = os.path.getsize(output_h5_path) / 1e9
    print(f"\nDone.")
    print(f"  Samples written : {n_samples}")
    print(f"  rgb shape       : ({n_samples}, {seq_len}, 3, {out_H}, {out_W})")
    print(f"  File size       : {size_gb:.2f} GB  (gzip-4)")
    if save_mp4 and mp4_writer is not None:
        mp4_path = os.path.splitext(output_h5_path)[0] + ".mp4"
        mp4_gb   = os.path.getsize(mp4_path) / 1e9
        print(f"  MP4 size        : {mp4_gb:.2f} GB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",       required=True,
                        help="Input video path (AVI or MP4, any fps, any resolution)")
    parser.add_argument("--fusion_h5",   default=None,
                        help="Existing fusion H5 (read-only) — source of N.")
    parser.add_argument("--n_samples",   type=int, default=None,
                        help="Override N directly. Omit to auto-derive from video length.")
    parser.add_argument("--output",      required=True,
                        help="Output H5 path")
    parser.add_argument("--bin_length",  type=int, default=50_000,
                        help="Bin length in µs (default 50000 = 50 ms = 20 Hz)")
    parser.add_argument("--seq_len",     type=int, default=1,
                        help="Frames per sample (default 1)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap number of samples for quick testing")
    parser.add_argument("--out_hw",        type=int, nargs=2, default=None,
                        metavar=("H", "W"),
                        help="Output resolution as H W (e.g. --out_hw 720 1280). "
                             "Center-crops from native res; falls back to INTER_AREA resize "
                             "if source is smaller. Omit to keep native resolution.")
    parser.add_argument("--save_mp4",      action="store_true",
                        help="Write a 20 Hz MP4 alongside the H5 (same folder, same basename).")
    parser.add_argument("--stretch_factor", type=float, default=1.0,
                        help="Fast-motion correction factor (default 1.0 = normal). "
                             "Set to N if the video plays N× faster than real time "
                             "(e.g. 11.85 if 60s video covers 710s of fusion H5 content). "
                             "Mutually exclusive with --auto_stretch.")
    parser.add_argument("--auto_stretch",   action="store_true",
                        help="Auto-compute stretch_factor from fusion H5 vs video duration "
                             "ratio when the ratio exceeds 1.5. Requires --fusion_h5.")
    parser.add_argument("--end_time",       type=float, default=None,
                        help="Hard cut: ignore video frames after this many seconds. "
                             "Use to remove post-session relaxation at the end of the recording.")
    args = parser.parse_args()

    preprocess_rgb(
        video_path         = args.video,
        output_h5_path     = args.output,
        fusion_h5_path     = args.fusion_h5,
        n_samples_override = args.n_samples,
        bin_length         = args.bin_length,
        seq_len            = args.seq_len,
        max_samples        = args.max_samples,
        out_hw             = tuple(args.out_hw) if args.out_hw else None,
        save_mp4           = args.save_mp4,
        stretch_factor     = args.stretch_factor,
        auto_stretch       = args.auto_stretch,
        end_time           = args.end_time,
    )
