# coding=utf-8
# =============================================================================
# export.py — Video and HDF5 export for captured RGB frames
#
# save_video()  — encodes frames with SpinVideo (H.264, MJPG, or uncompressed)
# save_hdf5()   — writes a gzip-compressed HDF5 dataset with per-frame timestamps
#
# Source: SaveToVideo.py (Spinnaker SDK example)
# =============================================================================

import os
import time

import numpy as np
import h5py
import PySpin

from config import OUTPUT_DIR, VIDEO_FORMAT, SAVE_HDF5, TRIGGER_TYPE, TriggerType


def ensure_output_dir():
    """Create the output directory if it does not exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# Video
# =============================================================================

def save_video(frames, nodemap, serial):
    """
    Encodes and saves captured frames to a video file using SpinVideo.

    SpinVideo is built into Spinnaker — no FFmpeg required.
    The video frame rate is read directly from the camera's
    AcquisitionFrameRate node so playback is in real-time.

    :param frames:  List of RGB8 PySpin.ImagePtr objects.
    :param nodemap: GenICam nodemap (for reading frame rate).
    :param serial:  Camera serial number string (for filename).
    :return:        True if successful.
    """
    if not frames:
        print("[VIDEO] No frames to save.")
        return False

    print(f"\n[VIDEO] Saving {len(frames)} frames as {VIDEO_FORMAT}...")
    try:
        timestamp  = time.strftime("%Y%m%d_%H%M%S")
        base_name  = os.path.join(OUTPUT_DIR, f"rgb_{serial}_{timestamp}")

        node_fps = PySpin.CFloatPtr(nodemap.GetNode('AcquisitionFrameRate'))
        fps      = node_fps.GetValue() if PySpin.IsReadable(node_fps) else 30.0
        print(f"[VIDEO] FPS for video metadata: {fps:.2f}")

        video_recorder = PySpin.SpinVideo()
        h = frames[0].GetHeight()
        w = frames[0].GetWidth()

        if VIDEO_FORMAT == "UNCOMPRESSED":
            option            = PySpin.AVIOption()
            option.frameRate  = fps
            option.height     = h
            option.width      = w

        elif VIDEO_FORMAT == "MJPG":
            option            = PySpin.MJPGOption()
            option.frameRate  = fps
            option.quality    = 85   # 0–100
            option.height     = h
            option.width      = w

        elif VIDEO_FORMAT in ("H264_AVI", "H264_MP4"):
            option            = PySpin.H264Option()
            option.frameRate  = fps
            option.bitrate    = 4_000_000   # bits/sec
            option.height     = h
            option.width      = w
            option.useMP4     = (VIDEO_FORMAT == "H264_MP4")
            option.crf        = 23          # lower = better quality

        else:
            print(f"[VIDEO] Unknown VIDEO_FORMAT '{VIDEO_FORMAT}'.")
            return False

        video_recorder.Open(base_name, option)

        for i, frame in enumerate(frames):
            video_recorder.Append(frame)
            if (i + 1) % 100 == 0:
                print(f"[VIDEO] Encoded {i + 1}/{len(frames)} frames...")

        video_recorder.Close()
        print(f"[VIDEO] Saved → {base_name}")

    except PySpin.SpinnakerException as ex:
        print(f"[VIDEO] Error: {ex}")
        return False

    return True


# =============================================================================
# HDF5
# =============================================================================

def save_hdf5(frames, serial, nodemap):
    """
    Exports frames to a gzip-compressed HDF5 dataset.

    Dataset layout
    --------------
    /frames       uint8 (N, H, W, 3)  — RGB pixel data, channel order RGB
    /timestamps   float64 (N,)        — camera hardware timestamps in seconds
    Attributes    serial, resolution, fps, trigger_type, n_frames, date, …

    Loading the dataset later
    -------------------------
        import h5py, numpy as np
        with h5py.File('...h5', 'r') as f:
            frames     = f['frames'][:]      # (N, H, W, 3) uint8
            timestamps = f['timestamps'][:]  # seconds, camera clock

    HDF5 advantages for ML
    ----------------------
    ✓ Single file, fast per-frame random access (one chunk = one frame)
    ✓ Gzip compression — much smaller than raw video
    ✓ Works natively with h5py, PyTorch, TensorFlow, MATLAB

    :param frames:  List of RGB8 PySpin.ImagePtr objects.
    :param serial:  Camera serial number (filename + metadata).
    :param nodemap: GenICam nodemap (for metadata).
    :return:        True if successful.
    """
    if not frames:
        print("[HDF5] No frames to export.")
        return False

    print(f"\n[HDF5] Exporting {len(frames)} frames to HDF5...")
    try:
        n = len(frames)
        h = frames[0].GetHeight()
        w = frames[0].GetWidth()

        data          = np.empty((n, h, w, 3), dtype=np.uint8)
        timestamps_ns = np.empty(n, dtype=np.int64)

        for i, frame in enumerate(frames):
            data[i]          = frame.GetNDArray()
            timestamps_ns[i] = frame.GetTimeStamp()  # nanoseconds, camera clock

        timestamps_s = timestamps_ns.astype(np.float64) / 1e9

        node_fps = PySpin.CFloatPtr(nodemap.GetNode('AcquisitionFrameRate'))
        fps      = node_fps.GetValue() if PySpin.IsReadable(node_fps) else 0.0

        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        hdf5_path     = os.path.join(OUTPUT_DIR, f"rgb_dataset_{serial}_{timestamp_str}.h5")

        with h5py.File(hdf5_path, 'w') as f:
            ds = f.create_dataset(
                'frames',
                data=data,
                dtype=np.uint8,
                compression='gzip',
                compression_opts=4,   # 1 (fast) – 9 (smallest)
                chunks=(1, h, w, 3)   # one chunk per frame → fast seeking
            )
            ds.attrs['channel_order'] = 'RGB'
            ds.attrs['description']   = 'Debayered RGB frames. Shape: (N, H, W, 3).'

            ts_ds = f.create_dataset('timestamps', data=timestamps_s, dtype=np.float64)
            ts_ds.attrs['unit']        = 'seconds'
            ts_ds.attrs['description'] = (
                'Camera hardware timestamps (camera internal clock, arbitrary epoch).'
            )

            f.attrs['camera_serial']  = serial
            f.attrs['n_frames']       = n
            f.attrs['height_px']      = h
            f.attrs['width_px']       = w
            f.attrs['fps_configured'] = fps
            f.attrs['trigger_type']   = (
                'HARDWARE_AcquisitionStart_Line0'
                if TRIGGER_TYPE == TriggerType.HARDWARE
                else 'SOFTWARE_FreeRun'
            )
            f.attrs['pixel_format']   = 'RGB8'
            f.attrs['color_algo']     = 'HQ_LINEAR'
            f.attrs['recorded_at']    = timestamp_str

        duration = timestamps_s[-1] - timestamps_s[0] if n > 1 else 0.0
        print(f"[HDF5] Saved → {hdf5_path}")
        print(f"       Shape    : ({n}, {h}, {w}, 3)  dtype=uint8")
        if n > 1:
            print(f"       Duration : {duration:.2f} s  |  "
                  f"Avg interval: {duration / (n - 1) * 1000:.1f} ms/frame")

    except Exception as ex:
        print(f"[HDF5] Error: {ex}")
        return False

    return True
