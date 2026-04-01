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
from config import TriggerType, TRIGGER_TYPE


# =============================================================================
# Image format
# =============================================================================

def configure_image_format(nodemap):
    """
    Sets the camera pixel format, width, and height to their sensor maximums.

    The BFS sensor outputs raw Bayer-pattern data. We set PixelFormat to
    BayerRG8 (raw mosaic) and debayer to RGB8 in software via ImageProcessor.
    This keeps USB/GigE bandwidth low while still producing full-colour output.

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

        # --- Width and Height: sensor maximum ---
        node_width = PySpin.CIntegerPtr(nodemap.GetNode('Width'))
        if PySpin.IsReadable(node_width) and PySpin.IsWritable(node_width):
            node_width.SetValue(node_width.GetMax())
            print(f"[FORMAT] Width  → {node_width.GetValue()} px")

        node_height = PySpin.CIntegerPtr(nodemap.GetNode('Height'))
        if PySpin.IsReadable(node_height) and PySpin.IsWritable(node_height):
            node_height.SetValue(node_height.GetMax())
            print(f"[FORMAT] Height → {node_height.GetValue()} px")

        # --- Offsets: zero so we capture the full sensor area ---
        for name in ('OffsetX', 'OffsetY'):
            node = PySpin.CIntegerPtr(nodemap.GetNode(name))
            if PySpin.IsReadable(node) and PySpin.IsWritable(node):
                node.SetValue(node.GetMin())

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
                   the camera then free-runs at AcquisitionFrameRate.

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
