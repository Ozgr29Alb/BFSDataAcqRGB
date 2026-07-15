# coding=utf-8
# =============================================================================
# acquisition.py — Streaming frame grab loop for the BFS RGB camera
#
# Producer–Consumer Architecture
# ================================
# This module implements a two-thread pipeline to decouple frame capture from
# disk I/O, allowing arbitrarily long recordings without RAM accumulation.
#
#   Producer (main acquire loop):
#     - Calls cam.GetNextImage() — blocks up to GRAB_TIMEOUT_MS
#     - Converts Bayer → RGB8 via ImageProcessor
#     - Copies pixel data to a numpy array (GetNDArray)
#     - Releases the camera buffer slot immediately (image_raw.Release())
#     - Pushes (image_rgb, ndarray, timestamp_ns) onto a bounded queue
#
#   Consumer (writer thread):
#     - Drains the queue and calls writer.append() for each frame
#     - writer.append() calls SpinVideo.Append() + optional HDF5 write
#
#   Bounded queue (QUEUE_MAXSIZE):
#     - If disk is slower than the camera, the producer blocks on put()
#       instead of filling RAM — pure backpressure, no OOM risk.
#
#   Stop conditions (any one triggers graceful shutdown):
#     - User presses STOP_KEY (keyboard watcher thread)
#     - MAX_DURATION_S elapsed (timer thread; None = disabled)
#
# Source: Acquisition.py + Trigger.py (Spinnaker SDK examples)
# =============================================================================

import queue
import time
import threading

import numpy as np
import PySpin
import keyboard

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

from config import (
    TRIGGER_TYPE, TriggerType,
    STOP_KEY, MAX_DURATION_S, QUEUE_MAXSIZE,
    GRAB_TIMEOUT_MS, COLOR_ALGO,
    SHOW_PREVIEW, PREVIEW_DOWNSCALE,
)
from lsl_markers import lsl_push


# ---------------------------------------------------------------------------
# Sentinel: placed on the queue to signal the writer thread to exit cleanly.
# ---------------------------------------------------------------------------
_STOP_SENTINEL = None


def acquire_frames(cam, nodemap, writer, lsl_outlet=None, serial="UNKNOWN"):
    """
    Streams frames from the camera to *writer* until stopped.

    The function runs a producer loop (this thread) that feeds a consumer
    (writer thread). No frames are held in RAM beyond the queue depth.

    Stop conditions
    ---------------
    1. User presses STOP_KEY.
    2. MAX_DURATION_S seconds elapsed (if not None).
    Either condition sets stop_event, which drains the queue and allows the
    writer thread to finish before this function returns.

    In HARDWARE mode BeginAcquisition() arms the camera.  After a single
    rising-edge pulse on Line0 the camera free-runs and frames start flowing.
    The 'first_frame' LSL marker is the most precise start timestamp.

    Parameters
    ----------
    cam : PySpin camera object (initialised, not yet streaming)
    nodemap : GenICam nodemap from cam.GetNodeMap()
    writer : export.StreamWriter  (already open, context-managed in main.py)
    lsl_outlet : pylsl.StreamOutlet or None
    serial : str  camera serial number (for log messages and LSL markers)

    Returns
    -------
    bool  True if at least one frame was captured without fatal error.
    """

    # --- AcquisitionMode → Continuous ---
    node_acq_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
    if not PySpin.IsReadable(node_acq_mode) or not PySpin.IsWritable(node_acq_mode):
        print("[ACQ] ERROR: AcquisitionMode not accessible.")
        return False
    node_acq_mode.SetIntValue(node_acq_mode.GetEntryByName('Continuous').GetValue())
    print("[ACQ] AcquisitionMode → Continuous")

    # --- ImageProcessor for Bayer → RGB8 (created once, reused per frame) ---
    processor = PySpin.ImageProcessor()
    processor.SetColorProcessing(COLOR_ALGO)

    # --- Shared stop flag ---
    stop_event = threading.Event()

    # --- Thread 1: key watcher — sets stop_event on STOP_KEY press ---
    def _watch_for_stop():
        keyboard.wait(STOP_KEY)
        print(f"\n[ACQ] '{STOP_KEY}' pressed — stopping...")
        stop_event.set()

    # --- Thread 2: optional duration timer ---
    def _watch_duration():
        time.sleep(MAX_DURATION_S)
        if not stop_event.is_set():
            print(f"\n[ACQ] MAX_DURATION_S ({MAX_DURATION_S} s) reached — stopping.")
            stop_event.set()

    threading.Thread(target=_watch_for_stop, daemon=True).start()
    if MAX_DURATION_S is not None:
        threading.Thread(target=_watch_duration, daemon=True).start()

    # --- Bounded frame queue ---
    frame_queue = queue.Queue(maxsize=QUEUE_MAXSIZE)

    # -----------------------------------------------------------------------
    # Writer thread (consumer)
    # -----------------------------------------------------------------------
    writer_error  = threading.Event()
    frames_written = [0]   # mutable container so the inner function can update it

    def _writer_loop():
        while True:
            item = frame_queue.get()
            if item is _STOP_SENTINEL:
                frame_queue.task_done()
                break
            image_rgb, ndarray, timestamp_ns = item
            try:
                writer.append(image_rgb, ndarray, timestamp_ns)
                frames_written[0] += 1
            except Exception as ex:
                print(f"[WRITER] Error appending frame: {ex}")
                writer_error.set()
            finally:
                frame_queue.task_done()

    writer_thread = threading.Thread(target=_writer_loop, daemon=False)
    writer_thread.start()

    # -----------------------------------------------------------------------
    # Start acquisition
    # -----------------------------------------------------------------------
    cam.BeginAcquisition()

    lsl_push(lsl_outlet, f"RGBCamera[{serial}]: start")

    if TRIGGER_TYPE == TriggerType.HARDWARE:
        print(f"\n[ACQ] Camera armed — waiting for hardware pulse on Line0...")
        print(f"      (send your trigger signal now; camera will start streaming)\n")
    else:
        print(f"\n[ACQ] Streaming started in free-run mode.")

    if MAX_DURATION_S is not None:
        m, s = divmod(int(MAX_DURATION_S), 60)
        print(f"[ACQ] Recording... press '{STOP_KEY}' to stop  "
              f"(auto-stop in {m:02d}:{s:02d}).\n")
    else:
        print(f"[ACQ] Recording... press '{STOP_KEY}' to stop.\n")

    # -----------------------------------------------------------------------
    # Producer loop
    # -----------------------------------------------------------------------
    frame_count         = 0
    first_frame_stamped = False
    t_start             = time.time()

    while not stop_event.is_set():
        try:
            image_raw = cam.GetNextImage(GRAB_TIMEOUT_MS)
        except PySpin.SpinnakerException:
            # Timeout: normal while waiting for trigger in HARDWARE mode.
            continue

        if image_raw.IsIncomplete():
            print(f"[ACQ] Frame {frame_count}: INCOMPLETE "
                  f"(status={image_raw.GetImageStatus()}), skipping.")
            image_raw.Release()
            continue

        # Convert Bayer → RGB8 (allocates a new buffer, decoupled from ring buffer)
        image_rgb    = processor.Convert(image_raw, PySpin.PixelFormat_RGB8)
        timestamp_ns = image_raw.GetTimeStamp()   # nanoseconds, camera clock

        # CRITICAL: release camera buffer slot immediately after Convert().
        # The BFS ring buffer has ~10 slots; holding them causes overflow drops.
        image_raw.Release()

        # Copy pixel data to numpy before handing off to writer thread.
        # GetNDArray() returns a view into the ImagePtr buffer, which may be
        # freed by SpinVideo; a copy ensures the writer thread sees valid data.
        ndarray = image_rgb.GetNDArray().copy()

        # LSL: stamp the first good frame (most precise start time in HARDWARE mode)
        if not first_frame_stamped:
            lsl_push(lsl_outlet, f"RGBCamera[{serial}]: first_frame")
            first_frame_stamped = True

        # Push to writer queue (blocks if queue is full → backpressure)
        frame_queue.put((image_rgb, ndarray, timestamp_ns))
        frame_count += 1

        # Progress log + queue depth health check
        if frame_count % 50 == 0:
            elapsed  = time.time() - t_start
            fps_act  = frame_count / elapsed if elapsed > 0 else 0
            q_depth  = frame_queue.qsize()
            q_warn   = " ⚠ DISK SLOW" if q_depth > QUEUE_MAXSIZE * 0.75 else ""
            print(f"[ACQ] {frame_count} frames | {fps_act:.1f} fps | "
                  f"queue {q_depth}/{QUEUE_MAXSIZE}{q_warn}")

        if writer_error.is_set():
            print("[ACQ] Writer thread error — stopping acquisition.")
            stop_event.set()

        # -------------------------------------------------------------------
        # OpenCV Live Preview (Rendered at ~10 FPS to preserve CPU/RAM bandwith)
        # -------------------------------------------------------------------
        if SHOW_PREVIEW and _CV2_AVAILABLE and (frame_count % 3 == 0):
            try:
                # ndarray is RGB. OpenCV expects BGR.
                bgr = cv2.cvtColor(ndarray, cv2.COLOR_RGB2BGR)
                if PREVIEW_DOWNSCALE > 1:
                    h, w = bgr.shape[:2]
                    bgr = cv2.resize(bgr, (w // PREVIEW_DOWNSCALE, h // PREVIEW_DOWNSCALE))
                
                window_name = f"Preview - {serial}"
                cv2.imshow(window_name, bgr)
                
                # waitKey processes window events. 1ms is enough.
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n[ACQ] 'q' pressed in preview window — stopping...")
                    stop_event.set()
            except Exception as e:
                print(f"[ACQ] Preview error: {e}")

    # -----------------------------------------------------------------------
    # Graceful shutdown
    # -----------------------------------------------------------------------
    cam.EndAcquisition()
    lsl_push(lsl_outlet, f"RGBCamera[{serial}]: stop")

    # Signal writer thread to finish draining, then wait for it
    frame_queue.put(_STOP_SENTINEL)
    writer_thread.join()

    elapsed   = time.time() - t_start
    fps_actual = frame_count / elapsed if elapsed > 0 else 0
    print(f"\n[ACQ] Stopped. {frame_count} frames captured | "
          f"{frames_written[0]} frames written | "
          f"{elapsed:.1f} s | {fps_actual:.1f} fps actual.")

    if SHOW_PREVIEW and _CV2_AVAILABLE:
        cv2.destroyAllWindows()

    return frames_written[0] > 0
