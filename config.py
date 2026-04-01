# coding=utf-8
# =============================================================================
# config.py — User configuration for BFS RGB acquisition
#
# Edit the constants in this file before running main.py.
# All other modules import from here; you should never need to touch them
# just to change a setting.
# =============================================================================

import PySpin


# ---------------------------------------------------------------------------
# Trigger mode
# ---------------------------------------------------------------------------
# SOFTWARE: camera free-runs immediately on BeginAcquisition() — use for testing.
# HARDWARE: camera waits for a rising-edge pulse on GPIO Line0 before starting.
#           Once that pulse arrives, it free-runs at AcquisitionFrameRate.

class TriggerType:
    SOFTWARE = 1   # no wiring needed; best for verifying the pipeline works
    HARDWARE = 2   # external device sends one pulse on GPIO Line0 to start recording


TRIGGER_TYPE = TriggerType.SOFTWARE   # ← change to TriggerType.HARDWARE when ready


# ---------------------------------------------------------------------------
# Stop key
# ---------------------------------------------------------------------------
# Press this key to stop recording and save output.
STOP_KEY = 'q'


# ---------------------------------------------------------------------------
# Safety limit
# ---------------------------------------------------------------------------
# Maximum frames to buffer in RAM. Recording also stops if this is reached.
# At 1920×1080 RGB8, each frame is ~6 MB; 1000 frames ≈ 6 GB in RAM.
# Adjust based on your available memory and session length.
MAX_FRAMES = 5000


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
OUTPUT_DIR   = "output"      # folder where video + HDF5 files are saved
VIDEO_FORMAT = "H264_MP4"    # "UNCOMPRESSED" | "MJPG" | "H264_AVI" | "H264_MP4"
SAVE_HDF5    = True          # write an HDF5 dataset alongside the video


# ---------------------------------------------------------------------------
# Grab timeout
# ---------------------------------------------------------------------------
# How long (ms) to wait for the next frame before retrying the loop.
# Keep this short in free-run mode so the STOP_KEY check stays responsive.
GRAB_TIMEOUT_MS = 500


# ---------------------------------------------------------------------------
# Color processing algorithm for Bayer → RGB debayering
# ---------------------------------------------------------------------------
# HQ_LINEAR = bilinear interpolation on the Bayer mosaic → good quality RGB.
# Use NEAREST_NEIGHBOR for faster (lower quality) debayering.
COLOR_ALGO = PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR
