# coding=utf-8
# =============================================================================
# main.py — BFS RGB Video Dataset Acquisition
# =============================================================================
#
# Entry point — orchestrates the full per-camera pipeline:
#   1. Detect cameras via Spinnaker
#   2. Configure image format and trigger  (camera_io.py)
#   3. Acquire frames until keypress       (acquisition.py)
#   4. Save video + HDF5                   (export.py)
#   5. Emit LSL markers at key events      (lsl_markers.py)
#
# USER SETTINGS → edit config.py (trigger mode, output dir, format, etc.)
#
# Usage:
#   python main.py
# =============================================================================

import os
import sys

import PySpin

from config      import TRIGGER_TYPE, TriggerType, STOP_KEY, MAX_FRAMES, OUTPUT_DIR, VIDEO_FORMAT, SAVE_HDF5
from camera_io   import configure_image_format, configure_trigger, reset_trigger
from acquisition import acquire_frames
from export      import ensure_output_dir, save_video, save_hdf5
from lsl_markers import create_lsl_outlet


# =============================================================================
# Per-camera pipeline
# =============================================================================

def run_single_camera(cam):
    """
    Runs the full acquisition pipeline for one camera.

    Steps
    -----
    1. cam.Init()
    2. configure_image_format()  — pixel format, resolution          [camera_io]
    3. configure_trigger()       — AcquisitionStart on Line0 or free-run [camera_io]
    4. create_lsl_outlet()       — LSL marker stream for this camera [lsl_markers]
    5. acquire_frames()          — grab until STOP_KEY pressed        [acquisition]
    6. save_video()              — encode to video file               [export]
    7. save_hdf5()               — export to HDF5 (optional)          [export]
    8. reset_trigger()           — put camera back to no-trigger state [camera_io]
    9. cam.DeInit()

    :param cam: PySpin camera object (not yet initialised).
    :return:    True if the pipeline completed without errors.
    """
    result = True
    try:
        # Serial number is available from the transport-layer nodemap before Init()
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

        # ── LSL outlet — created after Init() so the serial is confirmed ────
        lsl_outlet = create_lsl_outlet(serial)

        # ── Acquire ──────────────────────────────────────────────────────────
        ok, frames = acquire_frames(cam, nodemap, lsl_outlet=lsl_outlet, serial=serial)

        if not ok or not frames:
            print("[CAM] No frames captured.")
            reset_trigger(nodemap)
            cam.DeInit()
            return False

        # ── Save ─────────────────────────────────────────────────────────────
        ensure_output_dir()
        result &= save_video(frames, nodemap, serial)
        if SAVE_HDF5:
            result &= save_hdf5(frames, serial, nodemap)

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
    print("=" * 60)
    print(" BFS RGB Acquisition")
    print(f" Mode    : {'SOFTWARE (free-run)' if TRIGGER_TYPE == TriggerType.SOFTWARE else 'HARDWARE (AcquisitionStart on Line0)'}")
    print(f" Stop    : press '{STOP_KEY}'")
    print(f" Max RAM : {MAX_FRAMES} frames")
    print(f" Output  : {os.path.abspath(OUTPUT_DIR)}")
    print(f" Format  : {VIDEO_FORMAT}  |  HDF5: {SAVE_HDF5}")
    print("=" * 60)

    # Spinnaker system singleton — one per process, manages all cameras
    system = PySpin.System.GetInstance()
    ver    = system.GetLibraryVersion()
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

    # PySpin requires explicit del before clearing — Python GC alone is not enough
    del cam

    cam_list.Clear()
    system.ReleaseInstance()
    print("\n[SYS] Done.")
    return result


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
