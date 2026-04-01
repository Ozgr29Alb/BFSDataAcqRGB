# coding=utf-8
# =============================================================================
# lsl_markers.py — Lab Streaming Layer marker outlet for the RGB camera
#
# Publishes timestamped text markers that any LSL-compatible recorder on the
# same network (Acquisition_Interface, OpenSignals, BrainVision, etc.) can
# subscribe to and time-align with other modalities.
#
# Three markers are emitted per session:
#   RGBCamera[<serial>]: start       — BeginAcquisition() returned
#   RGBCamera[<serial>]: first_frame — very first complete frame received
#                                      (= real trigger moment in HARDWARE mode)
#   RGBCamera[<serial>]: stop        — EndAcquisition() called
# =============================================================================

try:
    from pylsl import StreamInfo, StreamOutlet
    _LSL_AVAILABLE = True
except ImportError:
    _LSL_AVAILABLE = False
    print("[LSL] pylsl not installed — markers will be printed to console only.")
    print("      Install with: pip install pylsl")


def create_lsl_outlet(serial: str):
    """
    Creates a pylsl StreamOutlet for sending text markers.

    The stream is named 'RGBCamera_<serial>' so other LSL listeners can
    identify which physical camera the markers come from.

    :param serial: Camera serial number string.
    :return: StreamOutlet if pylsl is available, else None.
    """
    if not _LSL_AVAILABLE:
        return None

    info = StreamInfo(
        name=f'RGBCamera_{serial}',  # unique per physical camera
        type='Markers',
        channel_count=1,
        nominal_srate=0,             # irregular rate — event-based stream
        channel_format='string',
        source_id=f'bfs_rgb_{serial}'
    )
    outlet = StreamOutlet(info)
    print(f"[LSL] Outlet created: stream='RGBCamera_{serial}' (visible on the network)")
    return outlet


def lsl_push(outlet, marker: str):
    """
    Pushes a text marker to the LSL outlet and always prints to console.

    :param outlet: StreamOutlet or None.
    :param marker: Marker string, e.g. 'RGBCamera[12345678]: start'.
    """
    print(f"[LSL] Marker → '{marker}'")
    if outlet is not None:
        outlet.push_sample([marker])
