# coding=utf-8
# =============================================================================
# main.py — BFS RGB Video Dataset Acquisition
# =============================================================================
#
# Entry point — orchestrates the full per-camera pipeline:
#   1. Detect cameras via Spinnaker
#   2. Configure image format, frame rate, and trigger  (camera_io.py)
#   3. Open StreamWriter (video + optional HDF5)         (export.py)
#   4. Stream frames to disk until keypress/timeout     (acquisition.py)
#   5. Emit LSL markers at key events                   (lsl_markers.py)
#
# USER SETTINGS → edit config.py
#
# Usage:
#   python main.py
# =============================================================================

import os
import sys

import PySpin

from config      import (
    TRIGGER_TYPE, TriggerType, STOP_KEY,
    MAX_DURATION_S, QUEUE_MAXSIZE,
    OUTPUT_DIR, VIDEO_FORMAT, VIDEO_BITRATE, SAVE_HDF5,
    WIDTH, HEIGHT, ACQUISITION_FPS, ADC_BIT_DEPTH,
)
from camera_io   import configure_image_format, configure_trigger, reset_trigger
from acquisition import acquire_frames
from export      import ensure_output_dir, StreamWriter
from lsl_markers import create_lsl_outlet


# =============================================================================
# Helpers
# =============================================================================

def _disk_estimate(fps, height, width, duration_s, bitrate_bps):
    """Returns a human-readable disk usage estimate string for H264."""
    if duration_s is None:
        return "unlimited (keypress only)"
    total_bits  = bitrate_bps * duration_s
    total_gb    = total_bits / 8 / 1024**3
    total_frames = int(fps * duration_s)
    m, s = divmod(int(duration_s), 60)
    h, m2 = divmod(m, 60)
    dur_str = f"{h:02d}:{m2:02d}:{s:02d}"
    return (
        f"~{total_gb:.2f} GB  "
        f"({total_frames:,} frames at {fps:.0f} fps over {dur_str})"
    )


def _ram_estimate(queue_size, height, width):
    """Returns peak RAM estimate string for the writer queue."""
    frame_bytes = height * width * 3 if (height and width) else 1920 * 1080 * 3
    peak_mb = queue_size * frame_bytes / 1024**2
    return f"~{peak_mb:.0f} MB  (queue depth {queue_size} × {frame_bytes / 1024**2:.1f} MB/frame)"


# =============================================================================
# Per-camera pipeline
# =============================================================================

def run_single_camera(cam):
    """
    Runs the full streaming acquisition pipeline for one camera.

    Steps
    -----
    1. cam.Init()
    2. configure_image_format() — pixel format, resolution, FPS, ADC depth
    3. configure_trigger()      — AcquisitionStart on Line0 or free-run
    4. create_lsl_outlet()      — LSL marker stream
    5. StreamWriter.__enter__() — open video file (before BeginAcquisition)
    6. acquire_frames()         — stream directly to disk (producer–consumer)
    7. StreamWriter.__exit__()  — close video, save .npy timestamps, close HDF5
    8. reset_trigger()
    9. cam.DeInit()

    :param cam: PySpin camera object (not yet initialised).
    :return:    True if the pipeline completed without errors.
    """
    result = True
    try:
        nodemap_tl  = cam.GetTLDeviceNodeMap()
        node_serial = PySpin.CStringPtr(nodemap_tl.GetNode('DeviceSerialNumber'))
        serial      = node_serial.GetValue() if PySpin.IsReadable(node_serial) else "UNKNOWN"

        print(f"\n{'='*60}")
        print(f"[CAM] Serial: {serial}")
        print(f"{'='*60}")

        cam.Init()
        nodemap = cam.GetNodeMap()

        # ── Configure (must happen before BeginAcquisition) ─────────────────
        if not configure_image_format(nodemap):
            cam.DeInit()
            return False

        if not configure_trigger(nodemap):
            cam.DeInit()
            return False

        # ── Read actual resulting dimensions from camera after configuration ─
        import PySpin as _PySpin
        node_w = _PySpin.CIntegerPtr(nodemap.GetNode('Width'))
        node_h = _PySpin.CIntegerPtr(nodemap.GetNode('Height'))
        actual_w = node_w.GetValue() if _PySpin.IsReadable(node_w) else (WIDTH or 1920)
        actual_h = node_h.GetValue() if _PySpin.IsReadable(node_h) else (HEIGHT or 1080)

        # ── LSL outlet ───────────────────────────────────────────────────────
        lsl_outlet = create_lsl_outlet(serial)

        # ── Stream writer (opens video file BEFORE BeginAcquisition) ─────────
        ensure_output_dir()
        with StreamWriter(nodemap, serial, actual_h, actual_w) as writer:
            ok = acquire_frames(
                cam, nodemap, writer,
                lsl_outlet=lsl_outlet,
                serial=serial,
            )

        if not ok:
            print("[CAM] No frames captured.")
            result = False

        # ── Cleanup ──────────────────────────────────────────────────────────
        result &= reset_trigger(nodemap)
        cam.DeInit()

    except PySpin.SpinnakerException as ex:
        print(f"[CAM] Fatal error: {ex}")
        result = False

    return result


# =============================================================================
# Main
# =============================================================================

def main():
    # ── Startup summary ──────────────────────────────────────────────────────
    mode_str = ('SOFTWARE (free-run)'
                if TRIGGER_TYPE == TriggerType.SOFTWARE
                else 'HARDWARE (AcquisitionStart on Line0)')

    res_str  = (f"{WIDTH}×{HEIGHT}" if (WIDTH and HEIGHT) else "sensor max")
    adc_str  = ADC_BIT_DEPTH or "camera default"

    # Use rough estimate dimensions for RAM calc (actual values read from camera later)
    est_w = WIDTH  or 1920
    est_h = HEIGHT or 1080

    disk_str = _disk_estimate(
        ACQUISITION_FPS, est_h, est_w, MAX_DURATION_S, VIDEO_BITRATE
    )
    ram_str  = _ram_estimate(QUEUE_MAXSIZE, est_h, est_w)

    print("=" * 60)
    print(" BFS RGB Acquisition  —  Streaming Pipeline")
    print(f" Mode        : {mode_str}")
    print(f" Resolution  : {res_str}  @ {ACQUISITION_FPS:.1f} fps")
    print(f" ADC depth   : {adc_str}")
    print(f" Format      : {VIDEO_FORMAT}  |  HDF5: {SAVE_HDF5}")
    print(f" Output      : {os.path.abspath(OUTPUT_DIR)}")
    print(f" Stop        : press '{STOP_KEY}'")
    print(f" Disk est.   : {disk_str}")
    print(f" Peak RAM    : {ram_str}")
    print("=" * 60)

    system  = PySpin.System.GetInstance()
    ver     = system.GetLibraryVersion()
    print(f"[SYS] Spinnaker {ver.major}.{ver.minor}.{ver.type}.{ver.build}")

    cam_list = system.GetCameras()
    n_cams   = cam_list.GetSize()
    print(f"[SYS] Cameras detected: {n_cams}")

    if n_cams == 0:
        cam_list.Clear()
        system.ReleaseInstance()
        print("[SYS] No cameras found. Check connection and power.")
        return False

    result = True
    for i, cam in enumerate(cam_list):
        print(f"\n[SYS] Camera {i+1} of {n_cams}")
        result &= run_single_camera(cam)

    del cam

    cam_list.Clear()
    system.ReleaseInstance()
    print("\n[SYS] Done.")
    return result


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
