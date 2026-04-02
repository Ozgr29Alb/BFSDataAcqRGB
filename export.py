# coding=utf-8
# =============================================================================
# export.py — Streaming video and timestamp export for BFS RGB acquisition
#
# StreamWriter
# ============
# Context-manager class that opens SpinVideo (H264 MP4) before acquisition
# begins, accepts frames one-by-one from the writer thread via append(), and
# finalises everything (video file, timestamps .npy, optional HDF5) on close.
#
# This replaces the old batch save_video() / save_hdf5() functions that
# required all frames to be held in RAM until the end of the recording.
#
# Usage (inside a `with` block):
#
#     with StreamWriter(nodemap, serial, height, width) as writer:
#         acquire_frames(cam, nodemap, writer, ...)
#     # → output/rgb_<serial>_<timestamp>.mp4
#     # → output/timestamps_ns_<serial>_<timestamp>.npy
#     # → output/rgb_dataset_<serial>_<timestamp>.h5  (if SAVE_HDF5=True)
#
# Source: SaveToVideo.py (Spinnaker SDK example)
# =============================================================================

import os
import time

import numpy as np

try:
    import PySpin
    _PYSPIN_AVAILABLE = True
except ImportError:
    PySpin = None          # type: ignore
    _PYSPIN_AVAILABLE = False

try:
    import h5py
    _H5PY_AVAILABLE = True
except ImportError:
    _H5PY_AVAILABLE = False

from config import (
    OUTPUT_DIR, VIDEO_FORMAT, VIDEO_BITRATE, VIDEO_CRF,
    SAVE_HDF5, HDF5_COMPRESSION, HDF5_CHUNK_FRAMES,
    TRIGGER_TYPE, TriggerType, ACQUISITION_FPS,
)


def ensure_output_dir():
    """Create the output directory if it does not exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# StreamWriter
# =============================================================================

class StreamWriter:
    """
    Opens a video file (and optional HDF5 file) before acquisition starts,
    accepts one frame at a time via append(), and closes everything cleanly.

    Thread safety
    -------------
    append() is called exclusively from the writer thread in acquisition.py.
    open() and close() are called from the main thread. No locking is needed
    because the writer thread only runs between open() and close().

    Parameters
    ----------
    nodemap : PySpin nodemap
        Used to read AcquisitionFrameRate for video metadata.
    serial : str
        Camera serial number for filenames and metadata.
    height : int
        Frame height in pixels (must be known before opening).
    width : int
        Frame width in pixels.
    """

    def __init__(self, nodemap, serial, height, width):
        self._nodemap  = nodemap
        self._serial   = serial
        self._height   = height
        self._width    = width

        self._video_recorder = None
        self._hdf5_file      = None
        self._hdf5_ds        = None
        self._hdf5_ts_ds     = None

        self._timestamps_ns  = []   # list of int64, flushed to .npy on close
        self._frame_count    = 0
        self._base_name      = None
        self._timestamp_str  = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False   # do not suppress exceptions

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self):
        """
        Opens the video recorder (and optional HDF5 file).
        Must be called before BeginAcquisition().
        """
        ensure_output_dir()

        self._timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        self._base_name     = os.path.join(
            OUTPUT_DIR, f"rgb_{self._serial}_{self._timestamp_str}"
        )

        # --- Read configured frame rate from camera (or fall back to config value) ---
        if _PYSPIN_AVAILABLE and self._nodemap is not None:
            node_fps = PySpin.CFloatPtr(self._nodemap.GetNode('AcquisitionFrameRate'))
            fps = node_fps.GetValue() if PySpin.IsReadable(node_fps) else ACQUISITION_FPS
        else:
            fps = ACQUISITION_FPS
        print(f"[WRITER] Video FPS: {fps:.2f}")

        # --- SpinVideo recorder (skipped in test/no-camera mode or if "NONE") ---
        h, w = self._height, self._width
        if _PYSPIN_AVAILABLE and VIDEO_FORMAT != "NONE":
            self._video_recorder = PySpin.SpinVideo()

            if VIDEO_FORMAT == "UNCOMPRESSED":
                option           = PySpin.AVIOption()
                option.frameRate = fps
                option.height    = h
                option.width     = w

            elif VIDEO_FORMAT == "MJPG":
                option           = PySpin.MJPGOption()
                option.frameRate = fps
                option.quality   = 85
                option.height    = h
                option.width     = w

            elif VIDEO_FORMAT in ("H264_AVI", "H264_MP4"):
                option           = PySpin.H264Option()
                option.frameRate = fps
                option.bitrate   = VIDEO_BITRATE
                option.height    = h
                option.width     = w
                option.useMP4    = (VIDEO_FORMAT == "H264_MP4")
                option.crf       = VIDEO_CRF

            else:
                raise ValueError(f"[WRITER] Unknown VIDEO_FORMAT '{VIDEO_FORMAT}'.")

            self._video_recorder.Open(self._base_name, option)
            ext = ".mp4" if VIDEO_FORMAT == "H264_MP4" else ".avi"
            print(f"[WRITER] Video opened → {self._base_name}{ext}")
        elif VIDEO_FORMAT == "NONE":
            print("[WRITER] VIDEO_FORMAT is 'NONE' — video encoder skipped.")
        else:
            print("[WRITER] PySpin not available — video encoder skipped (timestamps + HDF5 only).")

        # --- Optional HDF5 ---
        if SAVE_HDF5:
            self._open_hdf5(h, w, fps)

    def _open_hdf5(self, h, w, fps):
        """Opens a resizable HDF5 dataset for streaming frame-by-frame writes."""
        if not _H5PY_AVAILABLE:
            print("[WRITER] WARNING: h5py not installed — SAVE_HDF5 disabled.")
            return

        hdf5_path = os.path.join(
            OUTPUT_DIR,
            f"rgb_dataset_{self._serial}_{self._timestamp_str}.h5"
        )
        self._hdf5_file = h5py.File(hdf5_path, 'w')

        # Resizable dataset: starts at 0 frames, expands per chunk.
        # No compression — uncompressed writes have zero CPU overhead per frame,
        # which is important when streaming high-res RGB at 30+ fps.
        # Chunking is still required for resizable datasets and gives fast
        # per-frame seeking (one chunk = HDF5_CHUNK_FRAMES frames on disk).
        chunk_shape = (HDF5_CHUNK_FRAMES, h, w, 3)
        kwargs = {
            'name': 'frames',
            'shape': (0, h, w, 3),
            'maxshape': (None, h, w, 3),
            'dtype': np.uint8,
            'chunks': chunk_shape,
        }
        
        if HDF5_COMPRESSION:
            kwargs['compression'] = 'gzip'
            kwargs['compression_opts'] = 4
            
        self._hdf5_ds = self._hdf5_file.create_dataset(**kwargs)
        
        self._hdf5_ds.attrs['channel_order'] = 'RGB'
        self._hdf5_ds.attrs['description']   = 'Debayered RGB frames. Shape: (N, H, W, 3).'

        self._hdf5_ts_ds = self._hdf5_file.create_dataset(
            'timestamps',
            shape=(0,),
            maxshape=(None,),
            dtype=np.float64,
        )
        self._hdf5_ts_ds.attrs['unit']        = 'seconds'
        self._hdf5_ts_ds.attrs['description'] = (
            'Camera hardware timestamps (camera internal clock, arbitrary epoch).'
        )

        self._hdf5_file.attrs['camera_serial']  = self._serial
        self._hdf5_file.attrs['height_px']      = h
        self._hdf5_file.attrs['width_px']       = w
        self._hdf5_file.attrs['fps_configured'] = fps
        self._hdf5_file.attrs['trigger_type']   = (
            'HARDWARE_AcquisitionStart_Line0'
            if TRIGGER_TYPE == TriggerType.HARDWARE
            else 'SOFTWARE_FreeRun'
        )
        self._hdf5_file.attrs['pixel_format']   = 'RGB8'
        self._hdf5_file.attrs['recorded_at']    = self._timestamp_str

        print(f"[WRITER] HDF5 opened   → {hdf5_path}")

    def append(self, image_rgb, ndarray, timestamp_ns):
        """
        Appends one frame to the video (and optional HDF5).

        Called from the writer thread for every captured frame.

        Parameters
        ----------
        image_rgb : PySpin.ImagePtr
            Converted RGB8 image pointer (for SpinVideo.Append).
        ndarray : np.ndarray  shape (H, W, 3)  dtype uint8
            NumPy view of the frame data (for HDF5 + timestamp sidecar).
        timestamp_ns : int
            Camera hardware timestamp in nanoseconds.
        """
        # --- Video (skipped if no PySpin / test mode) ---
        if self._video_recorder is not None and image_rgb is not None:
            self._video_recorder.Append(image_rgb)

        # --- Timestamp accumulation (small int64 list, negligible RAM) ---
        self._timestamps_ns.append(int(timestamp_ns))

        # --- Optional HDF5 (resizable, appended per frame) ---
        if self._hdf5_ds is not None:
            n = self._frame_count
            self._hdf5_ds.resize(n + 1, axis=0)
            self._hdf5_ds[n] = ndarray
            ts_s = timestamp_ns / 1e9
            self._hdf5_ts_ds.resize(n + 1, axis=0)
            self._hdf5_ts_ds[n] = ts_s

        self._frame_count += 1

    def close(self):
        """
        Finalises the video file, saves the timestamps .npy sidecar,
        and closes any open HDF5 file.
        """
        n = self._frame_count

        # --- Close video ---
        if self._video_recorder is not None:
            try:
                self._video_recorder.Close()
                print(f"[WRITER] Video closed  ({n} frames).")
            except Exception as ex:
                print(f"[WRITER] Video close error: {ex}")
            self._video_recorder = None

        # --- Save timestamps sidecar ---
        if self._timestamps_ns:
            ts_arr  = np.array(self._timestamps_ns, dtype=np.int64)
            npy_path = os.path.join(
                OUTPUT_DIR,
                f"timestamps_ns_{self._serial}_{self._timestamp_str}.npy"
            )
            np.save(npy_path, ts_arr)
            duration = (ts_arr[-1] - ts_arr[0]) / 1e9 if n > 1 else 0.0
            print(f"[WRITER] Timestamps    → {npy_path}")
            print(f"         {n} frames | {duration:.2f} s duration | "
                  f"{duration / (n - 1) * 1000:.2f} ms/frame avg" if n > 1
                  else f"         {n} frame(s)")

        # --- Close HDF5 ---
        if self._hdf5_file is not None:
            try:
                self._hdf5_file.attrs['n_frames'] = n
                self._hdf5_file.close()
                print(f"[WRITER] HDF5 closed   ({n} frames).")
            except Exception as ex:
                print(f"[WRITER] HDF5 close error: {ex}")
            self._hdf5_file = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def frame_count(self):
        """Number of frames appended so far."""
        return self._frame_count

    @property
    def height(self):
        return self._height

    @property
    def width(self):
        return self._width
