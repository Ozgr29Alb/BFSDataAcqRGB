# coding=utf-8
# =============================================================================
# camera_io.py — Camera format and trigger configuration (Spinnaker SDK)
#
# All functions in this module MUST be called before cam.BeginAcquisition().
# Settings become read-only once acquisition starts.
#
# Sources: ImageFormatControl.py, Trigger.py (Spinnaker SDK examples)
# =============================================================================

import PySpin
from config import TriggerType, TRIGGER_TYPE, WIDTH, HEIGHT, ACQUISITION_FPS, ADC_BIT_DEPTH


# =============================================================================
# Image format
# =============================================================================

def configure_image_format(nodemap):
    """
    Configures pixel format, resolution (ROI), frame rate, and ADC bit depth.

    Pixel format
    ------------
    The BFS sensor outputs raw Bayer-pattern data. PixelFormat is set to
    BayerRG8 (raw mosaic); software debayering to RGB8 happens in ImageProcessor.
    This keeps USB/GigE bandwidth low while still producing full-colour output.

    Resolution
    ----------
    WIDTH / HEIGHT from config.py control the Region of Interest.
    None → sensor maximum (WidthMax / HeightMax).
    Values are clamped and snapped to the camera's increment (usually 1 or 8 px).

    Frame rate
    ----------
    AcquisitionFrameRateEnable must be True before AcquisitionFrameRate is
    writable. The resulting rate may be lower than requested if the exposure
    time is too long; check AcquisitionResultingFrameRate after BeginAcquisition.

    ADC bit depth
    -------------
    AdcBitDepth controls the sensor's A/D converter precision.
    None → leave at camera default.

    :param nodemap: GenICam nodemap from cam.GetNodeMap()
    :return: True if successful
    """
    print("\n[FORMAT] Configuring image format...")
    try:
        # --- Pixel format: BayerRG8 (or closest Bayer variant available) ---
        node_pixel_format = PySpin.CEnumerationPtr(nodemap.GetNode('PixelFormat'))
        if not PySpin.IsReadable(node_pixel_format) or not PySpin.IsWritable(node_pixel_format):
            print("[FORMAT] ERROR: PixelFormat node not accessible.")
            return False

        for bayer_name in ('BayerRG8', 'BayerGB8', 'BayerGR8', 'BayerBG8'):
            node_entry = PySpin.CEnumEntryPtr(node_pixel_format.GetEntryByName(bayer_name))
            if PySpin.IsReadable(node_entry):
                node_pixel_format.SetIntValue(node_entry.GetValue())
                print(f"[FORMAT] PixelFormat → {bayer_name}")
                break
        else:
            print("[FORMAT] ERROR: No 8-bit Bayer format available.")
            return False

        # --- Width ---
        node_width = PySpin.CIntegerPtr(nodemap.GetNode('Width'))
        if PySpin.IsReadable(node_width) and PySpin.IsWritable(node_width):
            w_max = node_width.GetMax()
            w_min = node_width.GetMin()
            w_inc = node_width.GetInc()
            if WIDTH is None:
                target_w = w_max
            else:
                target_w = max(w_min, min(w_max, int(WIDTH)))
                # snap to nearest valid increment
                target_w = w_min + ((target_w - w_min) // w_inc) * w_inc
            node_width.SetValue(target_w)
            print(f"[FORMAT] Width  → {node_width.GetValue()} px  (max={w_max})")
        else:
            print("[FORMAT] WARNING: Width node not writable — using current value.")

        # --- Height ---
        node_height = PySpin.CIntegerPtr(nodemap.GetNode('Height'))
        if PySpin.IsReadable(node_height) and PySpin.IsWritable(node_height):
            h_max = node_height.GetMax()
            h_min = node_height.GetMin()
            h_inc = node_height.GetInc()
            if HEIGHT is None:
                target_h = h_max
            else:
                target_h = max(h_min, min(h_max, int(HEIGHT)))
                target_h = h_min + ((target_h - h_min) // h_inc) * h_inc
            node_height.SetValue(target_h)
            print(f"[FORMAT] Height → {node_height.GetValue()} px  (max={h_max})")
        else:
            print("[FORMAT] WARNING: Height node not writable — using current value.")

        # --- Offsets: zero so we capture the full configured area ---
        for name in ('OffsetX', 'OffsetY'):
            node = PySpin.CIntegerPtr(nodemap.GetNode(name))
            if PySpin.IsReadable(node) and PySpin.IsWritable(node):
                node.SetValue(node.GetMin())

        # --- ADC bit depth ---
        if ADC_BIT_DEPTH is not None:
            node_adc = PySpin.CEnumerationPtr(nodemap.GetNode('AdcBitDepth'))
            if PySpin.IsReadable(node_adc) and PySpin.IsWritable(node_adc):
                entry = PySpin.CEnumEntryPtr(node_adc.GetEntryByName(ADC_BIT_DEPTH))
                if PySpin.IsReadable(entry):
                    node_adc.SetIntValue(entry.GetValue())
                    print(f"[FORMAT] AdcBitDepth → {ADC_BIT_DEPTH}")
                else:
                    print(f"[FORMAT] WARNING: AdcBitDepth '{ADC_BIT_DEPTH}' not available on this camera.")
            else:
                print("[FORMAT] WARNING: AdcBitDepth node not writable.")

        # --- Frame rate: enable control, then set target ---
        node_fps_enable = PySpin.CBooleanPtr(nodemap.GetNode('AcquisitionFrameRateEnable'))
        if PySpin.IsReadable(node_fps_enable) and PySpin.IsWritable(node_fps_enable):
            node_fps_enable.SetValue(True)
            print("[FORMAT] AcquisitionFrameRateEnable → True")
        else:
            print("[FORMAT] WARNING: AcquisitionFrameRateEnable not accessible — "
                  "frame rate may be exposure-limited.")

        node_fps = PySpin.CFloatPtr(nodemap.GetNode('AcquisitionFrameRate'))
        if PySpin.IsReadable(node_fps) and PySpin.IsWritable(node_fps):
            fps_max = node_fps.GetMax()
            fps_min = node_fps.GetMin()
            target_fps = max(fps_min, min(fps_max, float(ACQUISITION_FPS)))
            node_fps.SetValue(target_fps)
            print(f"[FORMAT] AcquisitionFrameRate → {target_fps:.2f} fps  "
                  f"(range: {fps_min:.1f}–{fps_max:.1f})")
        else:
            print("[FORMAT] WARNING: AcquisitionFrameRate not writable — "
                  "camera will use its current frame rate.")

    except PySpin.SpinnakerException as ex:
        print(f"[FORMAT] Error: {ex}")
        return False

    return True


# =============================================================================
# Trigger
# =============================================================================

def configure_trigger(nodemap):
    """
    Configures the camera trigger for the chosen TRIGGER_TYPE (from config.py).

    SOFTWARE mode: trigger stays Off; camera free-runs on BeginAcquisition().
    HARDWARE mode: TriggerSelector=AcquisitionStart, TriggerSource=Line0.
                   One rising-edge pulse on GPIO Line0 starts the acquisition;
                   the camera then free-runs at ACQUISITION_FPS.

    :param nodemap: GenICam nodemap from cam.GetNodeMap()
    :return: True if successful
    """
    if TRIGGER_TYPE == TriggerType.SOFTWARE:
        print("\n[TRIGGER] Software mode: trigger OFF — camera will free-run on BeginAcquisition().")
        try:
            node_trigger_mode = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerMode'))
            if PySpin.IsReadable(node_trigger_mode) and PySpin.IsWritable(node_trigger_mode):
                node_trigger_mode.SetIntValue(
                    node_trigger_mode.GetEntryByName('Off').GetValue()
                )
        except PySpin.SpinnakerException as ex:
            print(f"[TRIGGER] Warning: {ex}")
        return True

    # --- HARDWARE trigger ---
    print("\n[TRIGGER] Configuring HARDWARE AcquisitionStart trigger on Line0...")
    try:
        # Step 1: TriggerMode must be Off before changing source/selector
        node_trigger_mode = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerMode'))
        if not PySpin.IsReadable(node_trigger_mode) or not PySpin.IsWritable(node_trigger_mode):
            print("[TRIGGER] ERROR: TriggerMode not accessible.")
            return False
        node_trigger_mode.SetIntValue(node_trigger_mode.GetEntryByName('Off').GetValue())
        print("[TRIGGER] TriggerMode → Off (required to change settings)")

        # Step 2: TriggerSelector = AcquisitionStart
        # The trigger fires once to begin the whole session (not per-frame).
        node_trigger_selector = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerSelector'))
        if not PySpin.IsReadable(node_trigger_selector) or not PySpin.IsWritable(node_trigger_selector):
            print("[TRIGGER] ERROR: TriggerSelector not accessible.")
            return False
        node_trigger_selector.SetIntValue(
            node_trigger_selector.GetEntryByName('AcquisitionStart').GetValue()
        )
        print("[TRIGGER] TriggerSelector → AcquisitionStart")

        # Step 3: TriggerSource = Line0 (opto-isolated BFS GPIO input)
        node_trigger_source = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerSource'))
        if not PySpin.IsReadable(node_trigger_source) or not PySpin.IsWritable(node_trigger_source):
            print("[TRIGGER] ERROR: TriggerSource not accessible.")
            return False
        node_trigger_source.SetIntValue(
            node_trigger_source.GetEntryByName('Line0').GetValue()
        )
        print("[TRIGGER] TriggerSource → Line0 (rising edge)")

        # Step 4: Enable trigger — camera is now armed and waiting for Line0 pulse
        node_trigger_mode.SetIntValue(node_trigger_mode.GetEntryByName('On').GetValue())
        print("[TRIGGER] TriggerMode → On — waiting for pulse on Line0...")

    except PySpin.SpinnakerException as ex:
        print(f"[TRIGGER] Error: {ex}")
        return False

    return True


def reset_trigger(nodemap):
    """
    Turns TriggerMode back to Off after acquisition ends.

    IMPORTANT: forgetting this leaves the camera armed on Line0. On the next
    run it will appear frozen, waiting for a hardware pulse that never comes.

    :param nodemap: GenICam nodemap from cam.GetNodeMap()
    :return: True if successful
    """
    try:
        node_trigger_mode = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerMode'))
        if PySpin.IsReadable(node_trigger_mode) and PySpin.IsWritable(node_trigger_mode):
            node_trigger_mode.SetIntValue(
                node_trigger_mode.GetEntryByName('Off').GetValue()
            )
            print("[TRIGGER] TriggerMode reset → Off")
    except PySpin.SpinnakerException as ex:
        print(f"[TRIGGER] Reset warning: {ex}")
        return False
    return True
