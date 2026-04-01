# coding=utf-8
# =============================================================================
# acquisition.py — Frame grab loop for the BFS RGB camera
#
# Runs until the user presses STOP_KEY (from config.py) or MAX_FRAMES is hit.
# Emits LSL markers at acquisition start, first frame, and stop.
#
# Source: Acquisition.py + Trigger.py (Spinnaker SDK examples)
# =============================================================================

import time
import threading

import PySpin
import keyboard

from config import (
    TRIGGER_TYPE, TriggerType,
    STOP_KEY, MAX_FRAMES,
    GRAB_TIMEOUT_MS, COLOR_ALGO,
)
from lsl_markers import lsl_push


def acquire_frames(cam, nodemap, lsl_outlet=None, serial="UNKNOWN"):
    """
    Records frames until the user presses STOP_KEY or MAX_FRAMES is reached.

    How the loop works
    ------------------
    - GetNextImage(GRAB_TIMEOUT_MS) blocks for up to GRAB_TIMEOUT_MS ms.
    - Frame arrives  → convert Bayer→RGB8, stamp LSL on first frame, buffer it.
    - Timeout        → loop back and re-check the stop flag.
    - STOP_KEY press → stop_event is set; loop exits cleanly after current frame.

    In HARDWARE mode BeginAcquisition() arms the camera (→ idle, waiting for
    Line0 pulse). After that single pulse the camera free-runs and frames start
    flowing. The 'first_frame' LSL marker is the most precise start timestamp
    because it reflects when the pulse actually arrived.

    :param cam:        Initialized PySpin camera object.
    :param nodemap:    GenICam nodemap from cam.GetNodeMap().
    :param lsl_outlet: pylsl StreamOutlet (or None) for marker emission.
    :param serial:     Camera serial number string (for marker text).
    :return:           (success: bool, frames: list[PySpin.ImagePtr RGB8])
    """
    # --- AcquisitionMode → Continuous ---
    node_acq_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
    if not PySpin.IsReadable(node_acq_mode) or not PySpin.IsWritable(node_acq_mode):
        print("[ACQ] ERROR: AcquisitionMode not accessible.")
        return False, []
    node_acq_mode.SetIntValue(node_acq_mode.GetEntryByName('Continuous').GetValue())
    print("[ACQ] AcquisitionMode → Continuous")

    # --- ImageProcessor for Bayer → RGB8 debayering (created once, reused) ---
    processor = PySpin.ImageProcessor()
    processor.SetColorProcessing(COLOR_ALGO)

    # --- Stop flag shared between grab loop and key-watcher thread ---
    stop_event = threading.Event()

    def _watch_for_stop():
        """Background thread: blocks until STOP_KEY, then sets the flag."""
        keyboard.wait(STOP_KEY)
        stop_event.set()

    stop_watcher = threading.Thread(target=_watch_for_stop, daemon=True)
    stop_watcher.start()

    # --- Start acquisition ---
    cam.BeginAcquisition()

    # LSL: 'start' fires the moment BeginAcquisition() returns.
    # In SOFTWARE mode this is t=0 of the recording.
    # In HARDWARE mode the camera is armed but not yet streaming —
    # 'first_frame' below will stamp the actual trigger moment.
    lsl_push(lsl_outlet, f"RGBCamera[{serial}]: start")

    if TRIGGER_TYPE == TriggerType.HARDWARE:
        print(f"\n[ACQ] Camera armed — waiting for hardware pulse on Line0...")
        print(f"      (send your trigger signal now; camera will start streaming)\n")
    else:
        print(f"\n[ACQ] Streaming started in free-run mode.")

    print(f"[ACQ] Recording... press '{STOP_KEY}' to stop.\n")

    frames             = []
    frame_count        = 0
    first_frame_stamped = False
    t_start            = time.time()

    while not stop_event.is_set():
        try:
            image_raw = cam.GetNextImage(GRAB_TIMEOUT_MS)
        except PySpin.SpinnakerException:
            # Timeout: normal while waiting for trigger in HARDWARE mode.
            # In free-run mode a persistent timeout indicates a real problem.
            continue

        if image_raw.IsIncomplete():
            print(f"[ACQ] Frame {frame_count}: INCOMPLETE "
                  f"(status={image_raw.GetImageStatus()}), skipping.")
            image_raw.Release()
            continue

        # Convert raw Bayer → RGB8.
        # Convert() allocates a NEW buffer unlinked from the camera ring buffer,
        # so Release() below is safe and immediate.
        image_rgb = processor.Convert(image_raw, PySpin.PixelFormat_RGB8)

        # LSL: 'first_frame' — emitted exactly once on the first good frame.
        # In HARDWARE mode this is the most precise recording-start timestamp.
        if not first_frame_stamped:
            lsl_push(lsl_outlet, f"RGBCamera[{serial}]: first_frame")
            first_frame_stamped = True

        frames.append(image_rgb)
        frame_count += 1

        # CRITICAL: release the camera buffer slot immediately.
        # The BFS has ~10 slots by default; not releasing causes buffer overflow.
        image_raw.Release()

        if frame_count % 50 == 0:
            elapsed = time.time() - t_start
            fps_actual = frame_count / elapsed if elapsed > 0 else 0
            print(f"[ACQ] Captured {frame_count} frames | {fps_actual:.1f} fps actual")

        if frame_count >= MAX_FRAMES:
            print(f"[ACQ] MAX_FRAMES ({MAX_FRAMES}) reached — stopping.")
            break

    # --- Stop streaming ---
    cam.EndAcquisition()
    lsl_push(lsl_outlet, f"RGBCamera[{serial}]: stop")

    elapsed   = time.time() - t_start
    fps_actual = frame_count / elapsed if elapsed > 0 else 0
    print(f"\n[ACQ] Stopped. {frame_count} frames in {elapsed:.1f}s "
          f"({fps_actual:.1f} fps actual).")

    return True, frames
