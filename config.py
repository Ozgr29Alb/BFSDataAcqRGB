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
#           Once that pulse arrives, it free-runs at ACQUISITION_FPS.

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
# Resolution  (None = sensor maximum)
# ---------------------------------------------------------------------------
# WIDTH and HEIGHT set the camera's Region of Interest in pixels.
# If None, the full sensor area is used (e.g. 2048×1536 on BFS-U3-51S5C).
# Values are clamped to the sensor's WidthMax / HeightMax automatically.
# Note: some resolutions require Width/Height to be multiples of 8 or 16.
WIDTH  = None   # e.g. 1920  —  None → sensor WidthMax
HEIGHT = None   # e.g. 1080  —  None → sensor HeightMax


# ---------------------------------------------------------------------------
# Frame rate
# ---------------------------------------------------------------------------
# Target acquisition frame rate in frames per second.
# The camera must be able to sustain this at the chosen resolution.
# AcquisitionFrameRateEnable is set to True automatically before applying.
# Check AcquisitionResultingFrameRate after start if you suspect the exposure
# time is too long and is capping the actual rate below this target.
ACQUISITION_FPS = 30.0


# ---------------------------------------------------------------------------
# ADC bit depth
# ---------------------------------------------------------------------------
# Controls the analog-to-digital converter precision on the sensor.
#   None    → leave camera default (usually Bit8 or Bit10)
#   "Bit8"  → fastest readout, smallest bandwidth
#   "Bit10" → more dynamic range
#   "Bit12" → maximum precision (requires a matching PixelFormat, e.g. BayerRG12)
# Note: increasing ADC bit depth reduces the achievable maximum frame rate.
ADC_BIT_DEPTH = None   # None | "Bit8" | "Bit10" | "Bit12"


# ---------------------------------------------------------------------------
# Recording duration
# ---------------------------------------------------------------------------
# MAX_DURATION_S: hard stop after this many seconds even if STOP_KEY not pressed.
#   3600 = 1 hour.  Set to None to record until STOP_KEY only.
# QUEUE_MAXSIZE: depth of the internal writer queue (frames buffered in RAM).
#   At 1920×1080 RGB8 each frame is ~6 MB; 128 frames ≈ 768 MB peak RAM.
#   If your disk is fast enough the queue will stay near-empty in practice.
#   Bounded queue provides backpressure: producer blocks instead of OOM-ing.
MAX_DURATION_S = 3600   # seconds  (None = keypress only)
QUEUE_MAXSIZE  = 128    # frames   (tune to your available RAM)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
OUTPUT_DIR   = "output"       # folder where video + sidecar files are saved
VIDEO_FORMAT = "H264_MP4"     # "UNCOMPRESSED" | "MJPG" | "H264_AVI" | "H264_MP4"
VIDEO_BITRATE = 4_000_000     # H264 bitrate in bits/sec (4 Mbps ≈ 1.8 GB/hr)
VIDEO_CRF     = 23            # H264 quality: lower = better (0–51); ignored when
                              #   bitrate is the primary control on SpinVideo

# HDF5 export: disabled by default for long recordings.
# ~1.8 GB MP4 vs ~135-225 GB HDF5 for 1 hour at 1080p/30fps.
# Set True only for short sessions where per-frame random access is needed.
SAVE_HDF5         = False
HDF5_CHUNK_FRAMES = 32        # frames per HDF5 chunk (tune for your SSD)


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
