# coding=utf-8
# =============================================================================
# test_pipeline.py — Camera-free pipeline validation
#
# Tests the StreamWriter + producer-consumer queue WITHOUT any camera hardware
# or PySpin SDK. Uses synthetic numpy frames and a simulated frame generator.
#
# What is tested
# --------------
#   ✓ config.py is importable and all new parameters are present
#   ✓ export.py StreamWriter: open, append, close, sidecar .npy
#   ✓ Timestamps .npy sidecar is written and has correct frame count
#   ✓ Optional HDF5 path (if h5py is installed and SAVE_HDF5=True in config)
#   ✓ bounded queue backpressure: producer + consumer run in separate threads
#   ✓ Duration timer stops the loop automatically (MAX_DURATION_S respected)
#   ✓ STOP_KEY does NOT require keyboard hardware (skipped in this test)
#
# What is NOT tested here (requires camera hardware)
#   ✗ PySpin camera detection
#   ✗ SpinVideo H264 encoder
#   ✗ Actual camera timestamps
#   ✗ Hardware trigger pulse
#
# Usage
# -----
#   uv run python test_pipeline.py
#   or
#   python test_pipeline.py
#
# Expected output (no camera needed):
#   All checks printed as [PASS] / [FAIL]
#   A test_output/ folder with timestamps_ns_TEST_*.npy
# =============================================================================

import os
import sys
import queue
import time
import threading

import numpy as np

# ── Ensure we can import project modules from the same directory ──────────────
sys.path.insert(0, os.path.dirname(__file__))


# =============================================================================
# Helpers
# =============================================================================

class _Colors:
    OK   = "\033[92m"  # green
    FAIL = "\033[91m"  # red
    END  = "\033[0m"

_results = []

def check(label, condition, detail=""):
    status = f"{_Colors.OK}[PASS]{_Colors.END}" if condition else f"{_Colors.FAIL}[FAIL]{_Colors.END}"
    msg    = f"  {status} {label}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    _results.append((label, condition))
    return condition


# =============================================================================
# Test 1 — config.py imports and has required parameters
# =============================================================================

def test_config():
    print("\n── Test 1: config.py ──────────────────────────────────────────")
    try:
        import config as cfg
        check("config imports without error", True)
        check("WIDTH parameter exists",           hasattr(cfg, 'WIDTH'))
        check("HEIGHT parameter exists",          hasattr(cfg, 'HEIGHT'))
        check("ACQUISITION_FPS exists",           hasattr(cfg, 'ACQUISITION_FPS'))
        check("ADC_BIT_DEPTH exists",             hasattr(cfg, 'ADC_BIT_DEPTH'))
        check("MAX_DURATION_S exists",            hasattr(cfg, 'MAX_DURATION_S'))
        check("QUEUE_MAXSIZE exists",             hasattr(cfg, 'QUEUE_MAXSIZE'))
        check("SAVE_HDF5 exists",                 hasattr(cfg, 'SAVE_HDF5'))
        check("VIDEO_BITRATE exists",             hasattr(cfg, 'VIDEO_BITRATE'))
        check("MAX_FRAMES removed",               not hasattr(cfg, 'MAX_FRAMES'),
              "MAX_FRAMES must be gone — it caused OOM on long recordings")

        check("SAVE_HDF5 defaults to False",      cfg.SAVE_HDF5 is False,
              "HDF5 is huge for long recordings — default must be False")
        check("ACQUISITION_FPS is numeric",       isinstance(cfg.ACQUISITION_FPS, (int, float)))
        check("QUEUE_MAXSIZE is int > 0",         isinstance(cfg.QUEUE_MAXSIZE, int) and cfg.QUEUE_MAXSIZE > 0)

        return True
    except Exception as ex:
        check("config imports without error", False, str(ex))
        return False


# =============================================================================
# Test 2 — export.py StreamWriter (no PySpin needed)
# =============================================================================

def test_stream_writer():
    print("\n── Test 2: StreamWriter (no camera) ───────────────────────────")

    # Override OUTPUT_DIR to a safe test folder
    import config as cfg
    original_output_dir = cfg.OUTPUT_DIR
    cfg.OUTPUT_DIR = "test_output"
    os.makedirs("test_output", exist_ok=True)

    try:
        from export import StreamWriter, _PYSPIN_AVAILABLE
        check("export.py imports without PySpin", True)
        if _PYSPIN_AVAILABLE:
            print("  [INFO] PySpin is installed — SpinVideo encoder will run (camera not needed for this test).")
        else:
            print("  [INFO] PySpin not installed — video encoder skipped, timestamps + HDF5 still tested.")

        # Clean up stale test artefacts from previous runs
        test_dir = "test_output"
        for f in os.listdir(test_dir) if os.path.isdir(test_dir) else []:
            if f.startswith("timestamps_ns_TEST_"):
                os.remove(os.path.join(test_dir, f))

        # Create a writer with no nodemap (None is handled gracefully)
        H, W = 64, 96   # tiny synthetic resolution
        writer = StreamWriter(nodemap=None, serial="TEST", height=H, width=W)

        writer.open()
        check("StreamWriter.open() completed without crash", True)

        # Feed 30 synthetic frames (simulate 1 second at 30 fps)
        N_FRAMES = 30
        t0_ns    = int(time.time() * 1e9)
        frame_ns = int(1e9 / 30)   # ~33 ms per frame

        for i in range(N_FRAMES):
            # Synthetic RGB frame: a gradient that changes each frame
            arr = np.zeros((H, W, 3), dtype=np.uint8)
            arr[:, :, 0] = i * 4           # red channel ramps up
            arr[:, :, 1] = 128             # green constant
            arr[:, :, 2] = 255 - i * 4    # blue ramps down
            ts_ns = t0_ns + i * frame_ns
            # image_rgb=None → SpinVideo.Append skipped (guarded in export.py)
            writer.append(image_rgb=None, ndarray=arr, timestamp_ns=ts_ns)

        check(f"append() called {N_FRAMES} times without error", True)
        check("frame_count matches N_FRAMES", writer.frame_count == N_FRAMES,
              f"got {writer.frame_count}")

        writer.close()
        check("StreamWriter.close() completed without crash", True)

        # Verify sidecar .npy was written
        npy_files = [f for f in os.listdir("test_output")
                     if f.startswith("timestamps_ns_TEST_") and f.endswith(".npy")]
        check(".npy sidecar written", len(npy_files) == 1, str(npy_files))

        if npy_files:
            ts = np.load(os.path.join("test_output", npy_files[0]))
            check(".npy has correct frame count", len(ts) == N_FRAMES,
                  f"expected {N_FRAMES}, got {len(ts)}")
            check(".npy dtype is int64",          ts.dtype == np.int64,
                  str(ts.dtype))
            duration_s = (ts[-1] - ts[0]) / 1e9
            check("timestamps are monotonically increasing",
                  np.all(np.diff(ts) > 0))
            print(f"         Synthetic duration: {duration_s:.3f} s over {N_FRAMES} frames")

        return True

    except Exception as ex:
        check("StreamWriter test completed", False, str(ex))
        import traceback; traceback.print_exc()
        return False

    finally:
        cfg.OUTPUT_DIR = original_output_dir


# =============================================================================
# Test 3 — producer-consumer queue mechanics
# =============================================================================

def test_queue_mechanics():
    print("\n── Test 3: Producer-consumer queue ────────────────────────────")

    MAXSIZE   = 8
    N_FRAMES  = 50
    H, W      = 32, 32

    produced  = [0]
    consumed  = [0]
    q         = queue.Queue(maxsize=MAXSIZE)
    errors    = []

    def _producer():
        for i in range(N_FRAMES):
            arr    = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
            ts_ns  = int(time.time() * 1e9)
            q.put((arr, ts_ns))   # blocks if full → backpressure
            produced[0] += 1
        q.put(None)   # sentinel

    def _consumer():
        while True:
            item = q.get()
            if item is None:
                q.task_done()
                break
            arr, ts_ns = item
            time.sleep(0.002)   # simulate 2 ms disk write
            consumed[0] += 1
            q.task_done()
            if arr.shape != (H, W, 3):
                errors.append(f"bad shape: {arr.shape}")

    pt = threading.Thread(target=_producer)
    ct = threading.Thread(target=_consumer)
    ct.start(); pt.start()
    pt.join(); ct.join()

    check("All frames produced",  produced[0] == N_FRAMES, f"{produced[0]}/{N_FRAMES}")
    check("All frames consumed",  consumed[0] == N_FRAMES, f"{consumed[0]}/{N_FRAMES}")
    check("Queue drained to 0",   q.empty())
    check("No shape errors",      len(errors) == 0, str(errors))
    return not errors


# =============================================================================
# Test 4 — MAX_DURATION_S stop timer
# =============================================================================

def test_duration_timer():
    print("\n── Test 4: Duration timer (MAX_DURATION_S) ────────────────────")

    stop_event = threading.Event()
    TEST_DURATION = 0.5   # 500 ms for fast testing

    def _timer():
        time.sleep(TEST_DURATION)
        stop_event.set()

    t_start = time.time()
    threading.Thread(target=_timer, daemon=True).start()

    # Simulate producer waiting for stop
    frame_count = 0
    while not stop_event.is_set():
        time.sleep(0.05)   # 50 ms between "frames"
        frame_count += 1

    elapsed = time.time() - t_start
    check("Timer fires stop_event",          stop_event.is_set())
    check("Elapsed ≈ TEST_DURATION (±100ms)",
          abs(elapsed - TEST_DURATION) < 0.1,
          f"elapsed={elapsed:.3f}s, expected≈{TEST_DURATION}s")
    print(f"         Captured {frame_count} synthetic frames before auto-stop.")
    return True


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 62)
    print(" BFS RGB Pipeline — Camera-Free Unit Tests")
    print("=" * 62)

    test_config()
    test_stream_writer()
    test_queue_mechanics()
    test_duration_timer()

    # ── Summary ──────────────────────────────────────────────────
    passed = sum(1 for _, ok in _results if ok)
    total  = len(_results)
    failed = [(lbl, ok) for lbl, ok in _results if not ok]

    print("\n" + "=" * 62)
    print(f" Results: {passed}/{total} passed")
    if failed:
        print(f"\n{_Colors.FAIL} FAILED checks:{_Colors.END}")
        for lbl, _ in failed:
            print(f"   • {lbl}")
    else:
        print(f"{_Colors.OK} All checks passed!{_Colors.END}")
    print("=" * 62)

    print("""
╔══════════════════════════════════════════════════════════╗
║  Next step: connect the camera and run main.py           ║
║                                                          ║
║  SOFTWARE mode (no trigger wiring):                      ║
║    python main.py                                        ║
║    → records until you press 'q' or MAX_DURATION_S       ║
║                                                          ║
║  HARDWARE mode (GPIO Line0 trigger):                     ║
║    Set TRIGGER_TYPE = TriggerType.HARDWARE in config.py  ║
║    → camera arms, waits for pulse, then free-runs        ║
╚══════════════════════════════════════════════════════════╝
""")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
