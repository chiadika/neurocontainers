#!/usr/bin/env python3
"""Reusable OpenRecon / ISMRMRD plumbing for non-Cartesian MRI reconstruction.

This module is the part of a reconstruction app that is *not* the algorithm:
the MRD streaming loop, trajectory loading, k-space assembly, the standard
coil-data preparation steps, patient-space geometry, and DICOM display scaling.
It was carved out of ``recipes/sodiumgridding/sodiumgridding.py`` so that a new
reconstruction only has to supply one function.

You should not need to edit this file. Write your algorithm in the app module
(``openrecontemplate.py``) and hand it to :func:`run`::

    import mrdrecon

    mrdrecon.configure(name="myrecon", defaults=MY_DEFAULTS,
                       trajectories=MY_TRAJECTORIES)

    def reconstruct(recon):
        ...                       # <- your algorithm
        return volume             # real float32, (matrix, matrix, matrix)

    def process(connection, config, metadata):
        return mrdrecon.run(connection, config, metadata, reconstruct)

The volume you return is indexed ``[z, y, x]`` in the acquisition frame that the
trajectory defines; ``run`` takes care of orienting it into the DICOM display
frame, scaling it to the scanner's 0-4096 range, and emitting the images.
"""

from concurrent.futures import ThreadPoolExecutor  # noqa: F401  (for algorithms)
import ctypes
from dataclasses import dataclass
from itertools import permutations
import logging
import os
from pathlib import Path
import time
from time import perf_counter
import traceback
from typing import Any
import uuid

import h5py
import ismrmrd
import numpy as np
import scipy.ndimage as ndi

import constants
import mrdhelper


debugFolder = "/tmp/share/debug"

# ---------------------------------------------------------------------------
# Identity of the app using this framework. Override with configure().
# ---------------------------------------------------------------------------
RECON_NAME = "openrecontemplate"
OUTPUT_SERIES_DESCRIPTION = RECON_NAME
OUTPUT_IMAGE_COMMENT = "OpenRecon template reconstruction"
META_PREFIX = "OpenReconTemplate"
META_IMAGE_TYPE = "OPENRECONTEMPLATE"

# Bundled trajectory presets, keyed by the id used in OpenReconLabel.json.
BUNDLED_TRAJECTORIES = {}

# Every key the app accepts from the OpenRecon UI, with its fallback value.
# configure(defaults=...) merges the app's own keys over these.
OPENRECON_DEFAULTS = {
    "config": RECON_NAME,
    "matrixsize": 128,
    "fovcm": 22.0,
    "trajectorypreset": "",
    "trajectoryfile": "",
    "trajectorydataset": "k",
    "trajectorysampleoffset": 0,
    "rejectbadreadouts": True,
    "badreadoutsigma": 3.0,
    "centerwindow": 5,
    "applyfermifilter": True,
    "fermiwidth": 0.05,
    "fermicutoff": 0.98,
    "maxcoils": 16,
    "maxworkers": 8,
    "compresscoils": True,
    "coilvarianceretention": 0.9,
    "coilcombinemode": "AC",
    "applyn4biascorrection": False,
    "orientation": "zyx_fy",
    "orientationflipslice": False,
    "orientationdebugseries": False,
}


def configure(name=None, defaults=None, trajectories=None,
              series_description=None, image_comment=None, meta_prefix=None):
    """Point the framework at the app that is using it.

    Call this once, at import time, from your app module.
    """
    global RECON_NAME, OUTPUT_SERIES_DESCRIPTION, OUTPUT_IMAGE_COMMENT
    global META_PREFIX, META_IMAGE_TYPE, OUTPUT_FRAME_ORDER_REVERSED_ATTRIBUTE

    if name is not None:
        RECON_NAME = str(name)
        OUTPUT_SERIES_DESCRIPTION = RECON_NAME
        META_PREFIX = RECON_NAME[:1].upper() + RECON_NAME[1:]
        META_IMAGE_TYPE = RECON_NAME.upper()
        OPENRECON_DEFAULTS["config"] = RECON_NAME
    if series_description is not None:
        OUTPUT_SERIES_DESCRIPTION = str(series_description)
    if image_comment is not None:
        OUTPUT_IMAGE_COMMENT = str(image_comment)
    if meta_prefix is not None:
        META_PREFIX = str(meta_prefix)
    if defaults:
        OPENRECON_DEFAULTS.update(defaults)
    if trajectories is not None:
        BUNDLED_TRAJECTORIES.clear()
        BUNDLED_TRAJECTORIES.update(trajectories)

    OUTPUT_FRAME_ORDER_REVERSED_ATTRIBUTE = META_PREFIX + "IceFrameOrderReversed"


OUTPUT_IMAGE_SERIES_INDEX = 1
# Emit the reconstructed volume as one 3D MRD image (True) or as one 2D image
# per slice (False). The single 3D image matches the native ICE contract under
# TPI_3D_online, but IceProgramStandard's IceImageReconFunctors::ComputeImage
# crashes on it, so per-slice is the default. Flip this to revert.
EMIT_VOLUME_AS_SINGLE_IMAGE = False
AUTO_TRAJECTORY = "auto"
# FIRE mounts the scanner's `fire\share` folder here. Trajectory files that are
# too large to bake into the image can be dropped there; `auto` searches these
# directories after the bundled set. Matching stays by (samples, readouts).
SHARE_DIR = "/tmp/share"
TRAJECTORY_SEARCH_DIRS = [SHARE_DIR]
TRAJECTORY_FILE_GLOB = "*_trajectory.h5"
# Path of the trajectory file the last load resolved to (None if embedded).
LAST_TRAJECTORY_SOURCE = None

# Trajectory-to-acquisition orientation is the first geometry stage.
#
# The gridding kernel writes grid[iz, iy, ix] with ix taken from trajectory
# component 0, so the reconstructed volume is (component 2, component 1,
# component 0). Which trajectory component maps to the acquisition's read and
# phase axes is not encoded by the trajectory file. The orientation setting
# selects that mapping and any component-sign corrections.
ORIENTATION_IN_PLANE_TRANSFORMS = {
    # key: (transpose_in_plane, reverse_rows, reverse_columns)
    "zyx": (False, False, False),
    "zyx_fx": (False, False, True),
    "zyx_fy": (False, True, False),
    "zyx_fxy": (False, True, True),
    "zxy": (True, False, False),
    "zxy_fx": (True, False, True),
    "zxy_fy": (True, True, False),
    "zxy_fxy": (True, True, True),
}
# Trajectory component 1 runs opposite to the acquisition's phase_dir, so the
# rows have to be reversed to make the gridded pixels match the header. Measured
# on the scanner from the 0.1.5 run: with 'zyx' the anterior-posterior axis is
# flipped, the anatomy appearing mirrored top to bottom while the markers stay
# correct. It shows in the earlier figures too, where the app's row centroid is
# the negation of the native reference's: A9.3 against P9.3 in
# sodiumgridding_v0.1.3.PNG and A7.9 against P12.6 in the 0.1.4 pair.
#
# Unlike the stage 3 slice compensation, this is an honest correction rather
# than a workaround. The header always described the acquisition's true axes; it
# was the trajectory-to-axis mapping that had the sign wrong, so reversing the
# rows brings the pixels into agreement with phase_dir instead of away from it.
#
# The left-right sign remains unverified: a laterally symmetric phantom cannot
# reveal it. Only the row sign is corrected here.
DEFAULT_ORIENTATION = "zyx_fy"
ORIENTATION_DEBUG_SELECTION = "debug"
ORIENTATION_DEBUG_ORDER = tuple(ORIENTATION_IN_PLANE_TRANSFORMS)
#
# ISMRMRD direction vectors are in the DICOM/Siemens patient coordinate system
# (+x left, +y posterior, +z head). Labels below are (negative, positive).
PATIENT_AXIS_LABELS = (("R", "L"), ("A", "P"), ("F", "H"))

# The FIRE Configurator disables NormOrientation, so nothing downstream rotates
# our images into the DICOM standard display view. The native ICE reconstruction
# is emitted in that view, so this app has to produce it itself, otherwise the
# two series are mirrored relative to each other on screen.
#
# The standard view is the right-handed frame
#   columns increase toward the patient's Left      (+x)
#   rows    increase toward the patient's Posterior (+y)
#   slices  increase toward the patient's Head      (+z)
#
# The slice sign is measured from the native reference series in
# sodiumgridding_v0.1.3.PNG. That volume is centred at isocenter -- the scanner
# logfile records "dSag, dCor, Tra = 0; 0; 0" for the native reconstruction --
# and holds 128 slices over 220 mm, so frame 48 of 128 lies at
#   (47 - 63.5) * 220/128 = -28.36 mm
# along the slice axis. The reference displays that frame at SP F28.4, that is
# at z = -28.4, which is only possible if the slice axis points toward the Head.
#
# Version 0.1.3 targeted the Feet instead and displayed the same frame at H29.
# Because reversing one axis of a right-handed frame also swaps the side the
# viewer looks from, that single sign error is what produced the apparent 180
# degree rotation about the anterior-posterior axis: R/L exchanged on the left
# edge, the view-from marker flipped H/F, and slice 48 showing what the
# reference shows at slice 81.
#
# Targets are ordered (slices, rows, columns) to match the emitted array.
DISPLAY_FRAME_TARGETS = (
    (0.0, 0.0, 1.0),
    (0.0, 1.0, 0.0),
    (1.0, 0.0, 0.0),
)
DISPLAY_FRAME_AXIS_NAMES = ("slices", "rows", "columns")


def _validate_display_frame_targets(targets):
    """Return the handedness of the display targets, refusing a left-handed set.

    The targets describe the emitted DICOM frame, which the native
    reconstruction builds right-handed under the rule columns x rows = normal. A
    left-handed target set can therefore only be a sign mistake: it reverses one
    axis too many and puts the boxed view-from marker on the opposite side from
    the native series. That is exactly how 0.1.3 shipped.

    This says nothing about the incoming acquisition frame, which is
    DICOM-left-handed by convention because Siemens builds its PRS frame as
    phase x read = slice. _log_reference_geometry checks that one against the
    Siemens rule instead.
    """
    slices, rows, columns = (np.asarray(target, dtype=float) for target in targets)
    handedness = float(np.dot(np.cross(columns, rows), slices))
    if handedness <= 0.0:
        raise ValueError(
            "DISPLAY_FRAME_TARGETS must form a right-handed (columns, rows, "
            f"slices) frame, but columns x rows . slices = {handedness:+.3f}"
        )
    return handedness


DISPLAY_FRAME_HANDEDNESS = _validate_display_frame_targets(DISPLAY_FRAME_TARGETS)

# ICE stacks the frames of an emitted 3D volume against slice_dir, so the
# emitted frame order has to be reversed to compensate.
#
# Measured twice on the scanner with the same emitted slice_dir of F->H:
#   0.1.3, frame 48 of 128 over 220 mm: header says F28.4, scanner shows H29
#   0.1.4, frame 41 of 128 over 220 mm: header says F40.4, scanner shows H41
# Same magnitude, opposite sign, both times.
#
# The pixels themselves are in the right place. In the 0.1.4 run the emitted
# volume's intensity centroid is at F37.0, and the native 64-slice reference
# shows a full-width cross-section at F43.0. Were our content reversed, its bulk
# would sit at H37 and that reference slice would be nearly empty.
#
# Only the pixels may be reversed here, never slice_dir along with them.
# Transforming both is a change of storage convention and therefore a no-op: ICE
# derives the frame positions from slice_dir too, so the reversal would cancel
# and the displayed anatomy would not move by a single slice. That is why the
# 0.1.4 display-frame rotation, which is honest by construction, corrected the
# in-plane view but could not correct this.
#
# Set to False to emit the frame order unchanged, which is the right thing to do
# if a future FIRE release stops inverting the stacking. The log reports the
# predicted frame positions either way, so one screenshot settles it.
ICE_STACKS_FRAMES_AGAINST_SLICE_DIR = True

# Set on every emitted image to declare whether stage 3 reversed its frames.
# The compensation makes the pixels disagree with the emitted slice_dir on
# purpose, so this attribute is the contract that lets a consumer reading the
# header undo it rather than silently mirroring the volume.
OUTPUT_FRAME_ORDER_REVERSED_ATTRIBUTE = META_PREFIX + "IceFrameOrderReversed"
SCANNER_DISPLAY_MIN = 0
SCANNER_DISPLAY_MAX = 4096
N4_SHRINK_FACTOR = 2
N4_MAX_ITERATIONS = [50, 50, 50, 50]






def _get_config_value(config, key, default, value_type):
    try:
        return mrdhelper.get_json_config_param(config, key, default=default, type=value_type)
    except Exception:
        return default


def _config_bool(config, key, default):
    value = _get_config_value(config, key, default, "bool")
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _config_int(config, key, default):
    value = _get_config_value(config, key, default, "int")
    try:
        return int(value)
    except Exception:
        return int(default)


def _config_float(config, key, default):
    value = _get_config_value(config, key, default, "float")
    try:
        return float(value)
    except Exception:
        return float(default)


def _config_str(config, key, default):
    value = _get_config_value(config, key, default, "str")
    if value is None:
        return default
    return str(value)


def _ensure_debug_folder():
    os.makedirs(debugFolder, exist_ok=True)


def _read_runtime_file(*paths):
    for path_text in paths:
        try:
            value = Path(path_text).read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if value:
            return value
    return "unavailable"


def _cgroup_cpu_limit():
    cpu_max = _read_runtime_file("/sys/fs/cgroup/cpu.max")
    if cpu_max != "unavailable":
        return cpu_max

    quota = _read_runtime_file("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    period = _read_runtime_file("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if quota == "unavailable" and period == "unavailable":
        return "unavailable"
    return f"quota={quota} period={period}"


def _cgroup_cpuset():
    return _read_runtime_file(
        "/sys/fs/cgroup/cpuset.cpus.effective",
        "/sys/fs/cgroup/cpuset/cpuset.cpus",
    )


def _log_cpu_resources(configured_max_workers, effective_coil_workers):
    logical_cpu_count = os.cpu_count()
    try:
        affinity = sorted(os.sched_getaffinity(0))
        affinity_count = len(affinity)
        affinity_cpus = ",".join(str(cpu) for cpu in affinity)
    except (AttributeError, OSError):
        affinity_count = "unavailable"
        affinity_cpus = "unavailable"

    logging.info(
        "FIRE CPU resources: os_cpu_count=%s affinity_count=%s "
        "affinity_cpus=%s cgroup_cpu_limit='%s' cgroup_cpuset='%s' "
        "configured_maxworkers=%d effective_coil_workers=%d",
        logical_cpu_count,
        affinity_count,
        affinity_cpus,
        _cgroup_cpu_limit(),
        _cgroup_cpuset(),
        configured_max_workers,
        effective_coil_workers,
    )


def _safe_protocol_name(metadata):
    try:
        protocol_name = getattr(metadata.measurementInformation, "protocolName", "")
        if protocol_name:
            return str(protocol_name)
    except Exception:
        pass
    return OUTPUT_SERIES_DESCRIPTION


















def _normalize_trajectory_array(traj):
    array = np.asarray(traj)
    if array.size == 0:
        raise ValueError("Trajectory array is empty")

    if array.ndim == 2:
        if array.shape[-1] in (2, 3):
            array = array[None, :, :]
        elif array.shape[0] in (2, 3):
            array = array.T[None, :, :]
        else:
            raise ValueError(f"Unsupported 2D trajectory shape: {array.shape}")
    elif array.ndim == 3:
        if array.shape[-1] in (2, 3):
            pass
        elif array.shape[0] in (2, 3):
            array = np.moveaxis(array, 0, -1)
        elif array.shape[1] in (2, 3):
            array = np.moveaxis(array, 1, -1)
        else:
            raise ValueError(f"Unsupported 3D trajectory shape: {array.shape}")
    else:
        raise ValueError(f"Unsupported trajectory shape: {array.shape}")

    if array.shape[-1] == 2:
        array = np.concatenate(
            [array, np.zeros(array.shape[:-1] + (1,), dtype=array.dtype)],
            axis=-1,
        )

    return np.asarray(array, dtype=np.float32)


def _load_trajectory_from_file(path_text, dataset_name):
    path = Path(path_text).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Trajectory file does not exist: {path}")

    with h5py.File(path, "r") as h5_file:
        if dataset_name not in h5_file:
            raise KeyError(
                f"Trajectory dataset '{dataset_name}' not found in {path}. "
                f"Available datasets: {list(h5_file.keys())}"
            )
        trajectory = h5_file[dataset_name][...]

    logging.info("Loaded trajectory from %s[%s] with shape %s", path, dataset_name, trajectory.shape)
    return _normalize_trajectory_array(trajectory)


def _trajectory_dataset_name(config):
    return (_config_str(config, "trajectorydataset",
                        OPENRECON_DEFAULTS["trajectorydataset"]).strip()
            or OPENRECON_DEFAULTS["trajectorydataset"])


# Largest fraction of readouts that may be missing before an acquisition stops
# being recognisable as the sequence a bundled trajectory describes. The bundled
# trajectories differ in readout count by far more than this, so widening the
# match this much cannot make two of them ambiguous.
TRAJECTORY_SHORTFALL_TOLERANCE = 0.02


def _acquisition_dimensions(acquisitions):
    """(samples, readouts) actually present in the incoming acquisitions."""
    readouts = len(acquisitions)
    samples = min(int(a.data.shape[1]) for a in acquisitions) if readouts else 0
    return samples, readouts


def _acquisition_scan_counters(acquisitions):
    """MDH scan counters for the incoming readouts, or None if unusable.

    ICE numbers every readout it emits, so the counter identifies which spoke of
    the designed sequence each acquisition is. The Pulseq interpreter leaves some
    header fields at zero, so treat an all-zero or non-monotonic set as absent
    rather than trusting it.
    """
    try:
        counters = np.array(
            [int(a.getHead().scan_counter) for a in acquisitions], dtype=np.int64
        )
    except Exception as error:
        logging.warning("Could not read scan_counter from the acquisitions (%s)", error)
        return None

    if counters.size == 0 or not np.any(counters):
        return None
    if np.any(np.diff(counters) <= 0):
        logging.warning(
            "scan_counter is not strictly increasing (first=%d last=%d); "
            "falling back to acquisition order",
            counters[0],
            counters[-1],
        )
        return None
    return counters


def _log_acquisition_ordering(acquisitions, counters):
    """Report the received readout range and any gap in it.

    A short acquisition is only safe to reconstruct if the missing readouts are
    known: a gap in the middle shifts every later spoke onto the wrong
    trajectory row, which produces a plausible but wrong image.
    """
    samples, readouts = _acquisition_dimensions(acquisitions)
    try:
        lin = [int(a.getHead().idx.kspace_encode_step_1) for a in acquisitions]
        par = [int(a.getHead().idx.kspace_encode_step_2) for a in acquisitions]
        label_text = (f" LIN {min(lin)}..{max(lin)} ({len(set(lin))} distinct)"
                      f" PAR {min(par)}..{max(par)} ({len(set(par))} distinct)")
    except Exception:
        label_text = ""

    if counters is None:
        logging.warning(
            "Received %d readouts of %d samples; scan_counter unavailable, so "
            "readouts are matched to the trajectory by arrival order.%s",
            readouts,
            samples,
            label_text,
        )
        return

    span = int(counters[-1] - counters[0]) + 1
    missing = span - int(counters.size)
    logging.info(
        "Received %d readouts of %d samples; scan_counter %d..%d (span %d).%s",
        readouts,
        samples,
        int(counters[0]),
        int(counters[-1]),
        span,
        label_text,
    )
    if missing:
        gaps = np.nonzero(np.diff(counters) > 1)[0]
        largest = int(np.max(np.diff(counters))) - 1 if gaps.size else 0
        logging.warning(
            "%d readout(s) missing inside the received range: %d gap(s), largest %d. "
            "Alignment is by scan_counter, so the reconstruction stays correct.",
            missing,
            int(gaps.size),
            largest,
        )


def _readout_row_indices(acquisitions):
    """Trajectory row for each acquisition, or None to keep arrival order.

    The bound against the trajectory is checked by _clip_data_to_trajectory,
    which is the first place the readout axis is unambiguous -- a standalone
    trajectory may arrive with its samples and readouts transposed.
    """
    counters = _acquisition_scan_counters(acquisitions)
    _log_acquisition_ordering(acquisitions, counters)
    if counters is None:
        return None

    rows = counters - counters[0]
    if int(rows[-1]) + 1 == int(rows.size):
        return None                      # contiguous: positional order is identical
    logging.info(
        "Mapping %d readouts onto trajectory rows %d..%d by scan_counter",
        int(rows.size),
        int(rows[0]),
        int(rows[-1]),
    )
    return rows


def _trajectory_dimensions(path, dataset_name):
    """(samples, readouts) a trajectory file describes, or None.

    Reads only the dataset shape: auto-selection probes every candidate, and
    the larger files are hundreds of MB, so loading them all just to compare
    shapes would cost more than the reconstruction. The axis rule mirrors
    _normalize_trajectory_array; anything it does not cover falls back to a
    full load.
    """
    try:
        with h5py.File(Path(path).expanduser(), "r") as h5_file:
            if dataset_name not in h5_file:
                raise KeyError(f"dataset '{dataset_name}' not in {path}")
            shape = tuple(int(n) for n in h5_file[dataset_name].shape)
        if len(shape) == 3:
            if shape[-1] in (2, 3):
                return shape[0], shape[1]
            if shape[0] in (2, 3):
                return shape[1], shape[2]
            if shape[1] in (2, 3):
                return shape[0], shape[2]
        array = _normalize_trajectory_array(_load_trajectory_from_file(path, dataset_name))
        return int(array.shape[0]), int(array.shape[1])
    except Exception as exc:
        logging.warning("Could not read trajectory %s: %s", path, exc)
        return None


def _trajectory_candidates():
    """Bundled trajectories plus any *_trajectory.h5 in the search directories.

    Keyed by name; a bundled entry wins over a share-folder file of the same
    name, and the search directories are optional (absent on a workstation).
    """
    candidates = dict(BUNDLED_TRAJECTORIES)
    for directory in TRAJECTORY_SEARCH_DIRS:
        try:
            found = sorted(Path(directory).glob(TRAJECTORY_FILE_GLOB))
        except Exception:
            found = []
        for path in found:
            key = path.name[: -len("_trajectory.h5")]
            if key not in candidates:
                candidates[key] = str(path)
    return candidates


def _autoselect_trajectory(acquisitions, dataset_name):
    """Pick the bundled trajectory whose shape matches the acquired data.

    Selecting the wrong trajectory is not a benign mistake here. Spiral
    phyllotaxis directions depend on the *total* spoke count, so the first N
    spokes of a denser sequence are not the spokes of a sparser one -- measured
    on the bundled pair, 91.7% of directions differ by more than 10 degrees.
    And because _clip_data_to_trajectory crops to the shorter of the two, a
    mismatch would otherwise reconstruct silently and wrongly.

    The bundled trajectories have distinct (samples, readouts), so the acquired
    shape identifies the sequence unambiguously.
    """
    samples, readouts = _acquisition_dimensions(acquisitions)
    logging.info("Auto-selecting trajectory for acquired data: samples=%d readouts=%d",
                 samples, readouts)

    matches, short_matches, catalogue = [], [], []
    for key, path in sorted(_trajectory_candidates().items()):
        dims = _trajectory_dimensions(path, dataset_name)
        if dims is None:
            continue
        catalogue.append(f"{key} (samples={dims[0]}, readouts={dims[1]})")
        if dims == (samples, readouts):
            matches.append((key, path))
        elif dims[0] == samples and 0 < readouts < dims[1]:
            # A truncated acquisition -- ICE has been seen to close the raw
            # channel a few readouts before the end -- should still match the
            # sequence it came from. Keep the window tight enough that the
            # bundled trajectories stay mutually exclusive, and rely on
            # scan_counter alignment for correctness.
            if readouts >= dims[1] * (1.0 - TRAJECTORY_SHORTFALL_TOLERANCE):
                short_matches.append((key, path, dims[1]))

    unique = {path for _, path in matches}
    if len(unique) == 1:
        key, path = matches[0]
        logging.info("Auto-selected trajectory '%s': %s", key, path)
        return path

    available = "; ".join(catalogue) if catalogue else "none readable"

    if not matches and len({path for _, path, _ in short_matches}) == 1:
        key, path, full = short_matches[0]
        logging.warning(
            "Auto-selected trajectory '%s' for a short acquisition: %d of %d "
            "readouts received (%.2f%% missing): %s",
            key,
            readouts,
            full,
            100.0 * (full - readouts) / full,
            path,
        )
        return path

    if not matches:
        raise ValueError(
            f"No bundled trajectory matches the acquired data (samples={samples}, "
            f"readouts={readouts}). Available: {available}. The acquired sequence "
            "does not correspond to any bundled trajectory -- set 'trajectoryfile' "
            "to an explicit HDF5 path for this sequence."
        )
    raise ValueError(
        f"Acquired data (samples={samples}, readouts={readouts}) matches more than "
        f"one bundled trajectory: {available}. Set 'trajectoryfile' explicitly."
    )


def _resolve_trajectory_file(config, acquisitions=None):
    explicit_file = _config_str(
        config,
        "trajectoryfile",
        OPENRECON_DEFAULTS["trajectoryfile"],
    ).strip()
    if explicit_file.lower() == AUTO_TRAJECTORY and acquisitions:
        return _autoselect_trajectory(acquisitions, _trajectory_dataset_name(config))
    if explicit_file:
        if explicit_file in BUNDLED_TRAJECTORIES:
            trajectory_file = BUNDLED_TRAJECTORIES[explicit_file]
            logging.info("Using bundled trajectory %s: %s", explicit_file, trajectory_file)
            return trajectory_file
        logging.info("Using trajectory file override: %s", explicit_file)
        return explicit_file

    preset = _config_str(
        config,
        "trajectorypreset",
        OPENRECON_DEFAULTS["trajectorypreset"],
    ).strip() or OPENRECON_DEFAULTS["trajectorypreset"]

    if preset.lower() == AUTO_TRAJECTORY and acquisitions:
        return _autoselect_trajectory(acquisitions, _trajectory_dataset_name(config))

    if preset in BUNDLED_TRAJECTORIES:
        trajectory_file = BUNDLED_TRAJECTORIES[preset]
        logging.info("Using bundled trajectory preset %s: %s", preset, trajectory_file)
        return trajectory_file

    valid_presets = ", ".join(sorted(BUNDLED_TRAJECTORIES))
    raise ValueError(
        f"Unknown trajectory preset '{preset}'. "
        f"Expected one of: {valid_presets}. "
        "Use 'trajectoryfile' to provide an explicit external HDF5 path."
    )


def _load_trajectory(acquisitions, config):
    embedded_trajectory = []
    for acquisition in acquisitions:
        traj = getattr(acquisition, "traj", None)
        if traj is None:
            continue
        traj_array = np.asarray(traj)
        if traj_array.size == 0:
            continue
        embedded_trajectory.append(_normalize_trajectory_array(traj_array)[0])

    global LAST_TRAJECTORY_SOURCE
    if embedded_trajectory:
        trajectory = np.stack(embedded_trajectory, axis=0)
        logging.info("Using embedded ISMRMRD trajectory with shape %s", trajectory.shape)
        LAST_TRAJECTORY_SOURCE = None
        return trajectory

    trajectory_file = _resolve_trajectory_file(config, acquisitions)
    if not trajectory_file:
        raise ValueError(
            "No embedded trajectory found in the MRD acquisitions and no "
            "'trajectorypreset' or 'trajectoryfile' parameter was provided."
        )

    trajectory_dataset = _config_str(
        config,
        "trajectorydataset",
        OPENRECON_DEFAULTS["trajectorydataset"],
    ).strip() or OPENRECON_DEFAULTS["trajectorydataset"]
    trajectory = _load_trajectory_from_file(trajectory_file, trajectory_dataset)

    # Refuse a trajectory that does not describe this acquisition. Without this
    # check _clip_data_to_trajectory would quietly crop to the shorter of the
    # two and reconstruct a confidently wrong image.
    samples, readouts = _acquisition_dimensions(acquisitions)
    shape = _normalize_trajectory_array(trajectory).shape
    traj_samples, traj_readouts = int(shape[0]), int(shape[1])
    if readouts and (traj_samples, traj_readouts) != (samples, readouts):
        # A truncated acquisition still belongs to this trajectory: ICE has been
        # seen to close the raw channel a few readouts short of the end. Accept
        # the same shortfall the auto-selector does, and let scan_counter
        # alignment place the readouts that did arrive. Anything else -- a
        # different sample count, more readouts than the trajectory has, or a
        # shortfall beyond the tolerance -- is a different sequence.
        shortfall_ok = (
            traj_samples == samples
            and 0 < readouts < traj_readouts
            and readouts >= traj_readouts * (1.0 - TRAJECTORY_SHORTFALL_TOLERANCE)
        )
        if not shortfall_ok:
            raise ValueError(
                f"Trajectory {trajectory_file} describes samples={traj_samples} "
                f"readouts={traj_readouts}, but the acquisition has samples={samples} "
                f"readouts={readouts}. These must match exactly -- reconstructing a "
                "sequence with another sequence's trajectory produces a wrong image, "
                "not a degraded one. Set 'trajectoryfile' to 'auto' to select by shape."
            )
        logging.warning(
            "Acquisition is short by %d of %d readouts (%.2f%%); keeping trajectory "
            "%s and aligning the readouts that arrived.",
            traj_readouts - readouts,
            traj_readouts,
            100.0 * (traj_readouts - readouts) / traj_readouts,
            trajectory_file,
        )
    LAST_TRAJECTORY_SOURCE = trajectory_file
    return trajectory


def _build_data_array(acquisitions):
    num_readouts = len(acquisitions)
    num_coils = int(acquisitions[0].data.shape[0])
    num_samples = min(int(acq.data.shape[1]) for acq in acquisitions)

    data = np.zeros((num_coils, num_readouts, num_samples), dtype=np.complex64)
    for readout_index, acquisition in enumerate(acquisitions):
        acquisition_data = np.asarray(acquisition.data, dtype=np.complex64)
        if acquisition_data.shape[0] != num_coils:
            raise ValueError(
                "All acquisitions must contain the same number of coils. "
                f"Expected {num_coils}, got {acquisition_data.shape[0]}"
            )
        data[:, readout_index, :] = acquisition_data[:, :num_samples]

    return data


def _compute_default_fov_cm(metadata):
    try:
        fov_cm = float(metadata.encoding[0].reconSpace.fieldOfView_mm.x) / 10.0
        if fov_cm <= 0:
            raise ValueError(f"Invalid reconSpace FOV from metadata: {fov_cm}")
        return fov_cm
    except Exception:
        return OPENRECON_DEFAULTS["fovcm"]


def _compute_default_matrix_size(metadata):
    try:
        matrix_size = int(metadata.encoding[0].reconSpace.matrixSize.x)
        if matrix_size <= 1:
            raise ValueError(f"Invalid reconSpace matrix size from metadata: {matrix_size}")
        return matrix_size
    except Exception:
        return OPENRECON_DEFAULTS["matrixsize"]


def _clip_data_to_trajectory(data, trajectory, sample_offset, row_indices=None):
    data_readouts = data.shape[1]
    data_samples = data.shape[2]

    if trajectory.ndim == 3:
        direct_score = abs(trajectory.shape[0] - data_samples) + abs(trajectory.shape[1] - data_readouts)
        swapped_score = abs(trajectory.shape[1] - data_samples) + abs(trajectory.shape[0] - data_readouts)

        if swapped_score < direct_score:
            logging.warning(
                "Swapping trajectory axes to match standalone dimensions: trajectory=%s data=(samples=%d, readouts=%d)",
                trajectory.shape,
                data_samples,
                data_readouts,
            )
            trajectory = np.swapaxes(trajectory, 0, 1)

    available_sample_offset = max(0, trajectory.shape[0] - data_samples)
    applied_sample_offset = max(0, min(sample_offset, available_sample_offset))
    if applied_sample_offset > 0:
        logging.info(
            "Applying trajectory sample offset %d to align %d trajectory samples with %d raw samples",
            applied_sample_offset,
            trajectory.shape[0],
            data_samples,
        )

    if row_indices is not None:
        if int(np.max(row_indices)) < trajectory.shape[1]:
            # Each readout keeps the trajectory row it was acquired on, so a gap
            # in the stream drops rows instead of shifting every later spoke.
            trajectory = trajectory[:, np.asarray(row_indices, dtype=np.int64), :]
        else:
            logging.warning(
                "Ignoring scan_counter row mapping: row %d is outside the %d "
                "trajectory readouts",
                int(np.max(row_indices)),
                trajectory.shape[1],
            )

    samples = min(data.shape[2], trajectory.shape[0] - applied_sample_offset)
    readouts = min(data.shape[1], trajectory.shape[1])
    if readouts != data.shape[1] or samples != data.shape[2]:
        logging.warning(
            "Cropping raw data to match trajectory dimensions: data=%s trajectory=%s offset=%d -> (%d, %d)",
            data.shape,
            trajectory.shape,
            applied_sample_offset,
            readouts,
            samples,
        )
    return (
        data[:, :readouts, :samples],
        trajectory[applied_sample_offset:applied_sample_offset + samples, :readouts, :],
    )


def _compute_sample_mask_and_scale(coil_data, sigma):
    column_max = np.abs(coil_data).max(axis=0)
    smoothed = ndi.gaussian_filter1d(column_max.astype(np.float32), 5)
    histogram, edges = np.histogram(smoothed, bins=40)
    modal_edge = float(edges[int(np.argmax(histogram))])

    nearby = column_max[column_max >= 0.95 * modal_edge]
    if nearby.size == 0:
        bad_columns = np.array([], dtype=np.int64)
    else:
        threshold = modal_edge - sigma * float(nearby.std())
        bad_columns = np.where(column_max < threshold)[0]

    filtered = np.array(coil_data, copy=True)
    if bad_columns.size > 0:
        filtered[:, bad_columns] = 0

    return filtered, bad_columns


def _build_fermi_filter(
    trajectory,
    apply_fermi_filter,
    fermi_width,
    fermi_cutoff,
):
    if not apply_fermi_filter:
        return np.ones(trajectory.shape[:2], dtype=np.float32)

    abs_k = np.linalg.norm(trajectory, axis=-1)
    abs_k_max = float(np.max(abs_k)) if abs_k.size else 0.0
    if abs_k_max <= 0.0:
        return np.ones(trajectory.shape[:2], dtype=np.float32)

    normalized_radius = abs_k / abs_k_max
    fermi_filter = 1.0 / (
        1.0
        + np.exp(
            (normalized_radius - float(fermi_cutoff))
            / max(float(fermi_width), 1e-6)
        )
    )
    return np.asarray(fermi_filter, dtype=np.float32)


def _prepare_single_coil_data(
    coil_index,
    coil_data,
    reject_bad_readouts,
    bad_readout_sigma,
    center_window,
    fermi_filter,
):
    working_data = np.asarray(coil_data.T, dtype=np.complex64)

    if reject_bad_readouts:
        working_data, bad_columns = _compute_sample_mask_and_scale(
            working_data,
            sigma=bad_readout_sigma,
        )
        logging.info(
            "Coil %d: rejected %d low-signal sample columns",
            coil_index,
            bad_columns.size,
        )
    else:
        bad_columns = np.array([], dtype=np.int64)
        working_data = np.array(working_data, copy=True)

    center = working_data.shape[1] // 2
    start = max(0, center - center_window)
    stop = min(working_data.shape[1], center + center_window)
    reference_mean = float(np.abs(working_data[:, start:stop]).mean())
    if reference_mean > 0:
        working_data *= 2.0 / reference_mean

    return np.asarray(working_data * fermi_filter, dtype=np.complex64)


def _compress_coils_by_variance(coil_data, variance_retention=0.9, epsilon=1e-12):
    num_input_coils = int(coil_data.shape[0])
    if num_input_coils <= 1:
        return (
            np.asarray(coil_data, dtype=np.complex64),
            np.array([1.0], dtype=np.float32),
            np.eye(num_input_coils, dtype=np.complex64),
        )

    data_2d = np.asarray(coil_data).reshape(num_input_coils, -1)
    covariance = data_2d @ data_2d.conj().T
    covariance /= max(data_2d.shape[1], 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order].real, 0.0)
    eigenvectors = eigenvectors[:, order]

    total_variance = float(eigenvalues.sum())
    if total_variance <= epsilon:
        logging.warning(
            "Coil compression skipped because covariance has near-zero variance"
        )
        return (
            np.asarray(coil_data, dtype=np.complex64),
            np.ones(num_input_coils, dtype=np.float32),
            np.eye(num_input_coils, dtype=np.complex64),
        )

    cumulative_variance = np.cumsum(eigenvalues) / total_variance
    retention = float(np.clip(variance_retention, 0.0, 1.0))
    num_virtual_coils = min(
        num_input_coils,
        int(np.searchsorted(cumulative_variance, retention) + 1),
    )
    compression_matrix = np.asarray(
        eigenvectors[:, :num_virtual_coils],
        dtype=np.complex64,
    )
    compressed_data = np.einsum(
        "cv,c...->v...",
        compression_matrix.conj(),
        coil_data,
        optimize=True,
    )

    logging.info(
        "Coil compression: %d physical coils -> %d virtual coils "
        "(%.2f%% variance retained)",
        num_input_coils,
        num_virtual_coils,
        100.0 * cumulative_variance[num_virtual_coils - 1],
    )
    logging.info(
        "Cumulative coil variance: %s",
        np.array2string(cumulative_variance[:num_virtual_coils], precision=4),
    )
    return (
        np.asarray(compressed_data, dtype=np.complex64),
        np.asarray(cumulative_variance, dtype=np.float32),
        compression_matrix,
    )




def _estimate_sensitivities(coil_images, smooth_sigma=None, epsilon=1e-8):
    num_coils, matrix_x, _, _ = coil_images.shape
    if smooth_sigma is None:
        smooth_sigma = matrix_x / 32.0
    logging.info(
        "Estimating sensitivities for %d virtual coils with sigma %.2f voxels",
        num_coils,
        smooth_sigma,
    )

    smoothed = np.zeros_like(coil_images, dtype=np.complex64)
    for coil_index in range(num_coils):
        real = ndi.gaussian_filter(coil_images[coil_index].real, smooth_sigma)
        imaginary = ndi.gaussian_filter(coil_images[coil_index].imag, smooth_sigma)
        smoothed[coil_index] = real + 1j * imaginary

    root_sum_of_squares = np.sqrt(np.sum(np.abs(smoothed) ** 2, axis=0)) + epsilon
    smoothed /= root_sum_of_squares[None, ...]
    return np.asarray(smoothed, dtype=np.complex64)


def _combine_coils(coil_images, mode="AC"):
    normalized_mode = str(mode).strip().upper()
    if coil_images.shape[0] == 1:
        logging.info("Single virtual coil: skipping coil combination")
        return np.asarray(np.abs(coil_images[0]), dtype=np.float32)

    if normalized_mode == "SOS":
        logging.info("Combining virtual coils with root-sum-of-squares")
        return np.asarray(
            np.sqrt(np.sum(np.abs(coil_images) ** 2, axis=0)),
            dtype=np.float32,
        )

    if normalized_mode == "AC":
        sensitivity_maps = _estimate_sensitivities(coil_images)
        logging.info("Combining virtual coils adaptively")
        combined = np.sum(np.conj(sensitivity_maps) * coil_images, axis=0)
        return np.asarray(np.abs(combined), dtype=np.float32)

    raise ValueError("coilcombinemode must be 'AC' or 'SoS'")


def _n4_bias_field_correct(volume):
    try:
        import SimpleITK as sitk
    except ImportError as error:
        raise RuntimeError(
            "N4 bias correction requires the SimpleITK runtime dependency"
        ) from error

    values = np.asarray(volume, dtype=np.float32)
    if not values.size or float(np.max(values)) <= 0.0:
        logging.warning("Skipping N4 bias correction for an empty image")
        return values

    image_zyx = values.swapaxes(0, 2)
    sitk_image = sitk.GetImageFromArray(image_zyx)
    mask = sitk.OtsuThreshold(sitk_image, 0, 1, 512)
    mask = sitk.BinaryMorphologicalClosing(mask, [9, 9, 9])
    mask = sitk.BinaryFillhole(mask)

    if N4_SHRINK_FACTOR > 1:
        shrink = [int(N4_SHRINK_FACTOR)] * sitk_image.GetDimension()
        correction_image = sitk.Shrink(sitk_image, shrink)
        correction_mask = sitk.Shrink(mask, shrink)
    else:
        correction_image = sitk_image
        correction_mask = mask

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations(N4_MAX_ITERATIONS)
    corrector.SetConvergenceThreshold(0.001)
    corrector.SetSplineOrder(3)
    corrector.SetWienerFilterNoise(0.1)
    corrector.SetBiasFieldFullWidthAtHalfMaximum(0.15)
    corrector.Execute(correction_image, correction_mask)
    log_bias_field = corrector.GetLogBiasFieldAsImage(sitk_image)
    corrected = sitk_image / sitk.Exp(log_bias_field)
    return np.asarray(
        sitk.GetArrayFromImage(corrected).swapaxes(0, 2),
        dtype=np.float32,
    )


def _format_display_number(value):
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:.6g}"


def _scale_volume_to_display_range(volume):
    values = np.asarray(volume, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        display = np.zeros(values.shape, dtype=np.uint16)
        return display, {
            "input_min": 0.0,
            "input_max": 0.0,
            "scale": 1.0,
            "display_min": 0,
            "display_max": 0,
            "formula": "value = display",
        }

    input_min = float(np.min(finite))
    input_max = float(np.max(finite))
    input_range = input_max - input_min
    if input_range <= 0.0 or not np.isfinite(input_range):
        display = np.zeros(values.shape, dtype=np.uint16)
        return display, {
            "input_min": input_min,
            "input_max": input_max,
            "scale": 1.0,
            "display_min": 0,
            "display_max": 0,
            "formula": f"value = display + {_format_display_number(input_min)}",
        }

    scale = float(SCANNER_DISPLAY_MAX - SCANNER_DISPLAY_MIN) / input_range
    cleaned = np.nan_to_num(values, nan=input_min, posinf=input_max, neginf=input_min)
    display = np.rint((cleaned - input_min) * scale + SCANNER_DISPLAY_MIN)
    display = np.clip(display, SCANNER_DISPLAY_MIN, SCANNER_DISPLAY_MAX)
    display = display.astype(np.uint16, copy=False)
    scale_text = _format_display_number(scale)
    min_text = _format_display_number(input_min)
    return display, {
        "input_min": input_min,
        "input_max": input_max,
        "scale": scale,
        "display_min": int(np.min(display)) if display.size else 0,
        "display_max": int(np.max(display)) if display.size else 0,
        "formula": f"value = display / {scale_text} + {min_text}",
    }


def _scanner_display_comment(display_meta):
    return (
        f"{OUTPUT_IMAGE_COMMENT}; scanner display uint16 "
        f"{SCANNER_DISPLAY_MIN}-{SCANNER_DISPLAY_MAX}; {display_meta['formula']}"
    )


def _new_dicom_uid():
    return f"2.25.{uuid.uuid4().int}"


def _log_trajectory_extent(trajectory):
    """Log per-component k-space extent.

    An asymmetric or unequal extent between components is the main offline clue
    that the trajectory component order is not (read, phase, slice).
    """
    coordinates = np.asarray(trajectory, dtype=np.float64).reshape(-1, 3)
    if not coordinates.size:
        logging.warning("Trajectory is empty; cannot report k-space extent")
        return

    for component in range(3):
        values = coordinates[:, component]
        logging.info(
            "Trajectory component %d: min=%+.6f max=%+.6f mean=%+.6f "
            "abs_max=%.6f center_offset=%+.6f",
            component,
            float(values.min()),
            float(values.max()),
            float(values.mean()),
            float(np.abs(values).max()),
            float(values.min() + values.max()),
        )


def _direction_letters(vector):
    """Return the anatomical letters at the start and end of a direction vector.

    ``(-x, +x)`` yields ``("L", "R")`` because the vector begins on the
    patient's left and ends on the right. The scanner labels the edges of a
    displayed image with exactly these two letters.
    """
    values = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(values))
    if norm < 1e-6:
        return ("?", "?")
    unit = values / norm
    axis = int(np.argmax(np.abs(unit)))
    start, end = PATIENT_AXIS_LABELS[axis]
    if unit[axis] < 0:
        start, end = end, start
    return (start, end)


def _direction_label(vector):
    """Describe a patient-space direction vector as, for example, 'R->L'."""
    start, end = _direction_letters(vector)
    if start == "?":
        return "undefined"
    return f"{start}->{end}"


def _format_direction(vector):
    values = np.asarray(vector, dtype=float)
    components = ",".join(f"{float(value):+.4f}" for value in values)
    obliquity = float(np.max(np.abs(values))) if values.size else 0.0
    return f"{_direction_label(values)} [{components}] alignment={obliquity:.4f}"


def _position_label(position):
    """Format a patient-space position the way the scanner prints one, 'R6.3 P2.3 H35.4'."""
    values = np.asarray(position, dtype=float)
    parts = []
    for axis in range(3):
        value = float(values[axis])
        negative, positive = PATIENT_AXIS_LABELS[axis]
        parts.append(f"{positive if value >= 0.0 else negative}{abs(value):.1f}")
    return " ".join(parts)


def _slice_position_label(position):
    """Format the dominant component of a position, matching the scanner 'SP' field."""
    values = np.asarray(position, dtype=float)
    axis = int(np.argmax(np.abs(values)))
    value = float(values[axis])
    negative, positive = PATIENT_AXIS_LABELS[axis]
    return f"{positive if value >= 0.0 else negative}{abs(value):.1f}"


def _unit_vector(vector, fallback=(0.0, 0.0, 1.0)):
    values = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(values))
    if norm < 1e-6:
        return np.asarray(fallback, dtype=float)
    return values / norm


def _frame_position(center_position, slice_dir, frame_index, slice_count, slice_spacing_mm):
    """Patient-space centre of one emitted frame.

    ``center_position`` is the centre of the whole volume, which is what the MRD
    header carries, so frame ``frame_index`` of ``slice_count`` sits half a
    volume away from it minus its own offset. ``frame_index`` is zero based;
    the scanner numbers the same frame ``frame_index + 1``.
    """
    offset = (float(frame_index) - (float(slice_count) - 1.0) / 2.0) * float(
        slice_spacing_mm
    )
    return np.asarray(center_position, dtype=float) + offset * _unit_vector(slice_dir)


def _resolve_orientation(orientation):
    key = str(orientation).strip().lower()
    if key not in ORIENTATION_IN_PLANE_TRANSFORMS:
        logging.warning(
            "Unknown orientation '%s'; falling back to '%s'. Valid values: %s",
            orientation,
            DEFAULT_ORIENTATION,
            ", ".join(sorted(ORIENTATION_IN_PLANE_TRANSFORMS)),
        )
        key = DEFAULT_ORIENTATION
    return key


def _resolve_orientation_selection(selection):
    """Split a UI orientation value into (orientation, emit_debug_series).

    ``ORIENTATION_DEBUG_SELECTION`` is one of the values the scanner UI offers,
    so every entry point that accepts an orientation has to understand it. This
    is the single place that mapping is made.
    """
    if str(selection).strip().lower() == ORIENTATION_DEBUG_SELECTION:
        return DEFAULT_ORIENTATION, True
    return _resolve_orientation(selection), False


def _resolve_orientation_config(config):
    selection = _config_str(config, "orientation", OPENRECON_DEFAULTS["orientation"])
    orientation, emit_debug_series = _resolve_orientation_selection(selection)
    # Continue accepting the older boolean in manually supplied JSON configs.
    if _config_bool(
        config,
        "orientationdebugseries",
        OPENRECON_DEFAULTS["orientationdebugseries"],
    ):
        orientation, emit_debug_series = DEFAULT_ORIENTATION, True
    return orientation, emit_debug_series


# Optional volume post-processing supplied by the app module. Each hook is
# called as hook(volume, context) after the algorithm output (and N4) and
# before display scaling, orientation and emission, and returns the volume.
# `context` carries reference_head, metadata, params, output_fov_mm,
# orientation and flip_slice. An app that needs patient-space geometry can
# call to_acquisition_frame()/from_acquisition_frame() to work in the frame
# whose axes are slice_dir, phase_dir and read_dir with the header's centre.
OUTPUT_VOLUME_HOOKS = []


def to_acquisition_frame(volume, orientation, flip_slice=False):
    """(slices, rows, columns) along slice_dir, phase_dir, read_dir; returns (volume, key)."""
    return _orient_volume(volume, orientation, flip_slice)


def from_acquisition_frame(volume, orientation_key, flip_slice=False):
    """Inverse of to_acquisition_frame for the same key (all steps are involutions)."""
    transpose_in_plane, reverse_rows, reverse_columns = (
        ORIENTATION_IN_PLANE_TRANSFORMS[orientation_key]
    )
    oriented = np.asarray(volume)
    if flip_slice:
        oriented = oriented[::-1, :, :]
    if reverse_columns:
        oriented = oriented[:, :, ::-1]
    if reverse_rows:
        oriented = oriented[:, ::-1, :]
    if transpose_in_plane:
        oriented = oriented.transpose(0, 2, 1)
    return np.ascontiguousarray(oriented)


def _orient_volume(volume, orientation, flip_slice=False):
    """Map trajectory components into acquisition (slice, phase, read) axes."""
    key = _resolve_orientation(orientation)
    transpose_in_plane, reverse_rows, reverse_columns = (
        ORIENTATION_IN_PLANE_TRANSFORMS[key]
    )

    oriented = np.asarray(volume)
    if transpose_in_plane:
        oriented = oriented.transpose(0, 2, 1)
    if reverse_rows:
        oriented = oriented[:, ::-1, :]
    if reverse_columns:
        oriented = oriented[:, :, ::-1]
    if flip_slice:
        oriented = oriented[::-1, :, :]

    return np.ascontiguousarray(oriented), key


def _canonicalize_to_display_frame(volume, axis_directions):
    """Rotate a (slices, rows, columns) volume into the DICOM standard display view.

    ``axis_directions`` are the patient-space directions of the volume's own
    axes, in the same (slices, rows, columns) order. The returned directions
    describe the returned volume, so the header stays exactly as honest as it
    was on input: this is a change of display convention, not a correction
    layered on top of one. That also means it cannot move the anatomy, which is
    why stage 3 exists and why the emitted image is not self-consistent by the
    time it leaves _build_single_output_image.

    The permutation and signs are derived from the acquisition's own vectors
    rather than hardcoded, so obliquity and a non-HFS patient position are
    handled without a second code path.
    """
    directions = [np.asarray(vector, dtype=float) for vector in axis_directions]
    unit_directions = []
    for vector in directions:
        norm = float(np.linalg.norm(vector))
        unit_directions.append(vector / norm if norm > 1e-6 else vector)

    targets = [np.asarray(target, dtype=float) for target in DISPLAY_FRAME_TARGETS]

    # Choose all three axes as one assignment. Greedy target-by-target matching
    # can consume the second-best axis early and force a wrong mapping for a
    # valid oblique rotation.
    permutation = max(
        permutations(range(3)),
        key=lambda candidate: sum(
            abs(float(np.dot(unit_directions[candidate[index]], targets[index])))
            for index in range(3)
        ),
    )
    signs = [
        -1.0
        if float(np.dot(unit_directions[permutation[index]], targets[index])) < 0.0
        else 1.0
        for index in range(3)
    ]

    canonical = np.asarray(volume).transpose(*permutation)
    for output_axis, sign in enumerate(signs):
        if sign < 0.0:
            canonical = np.flip(canonical, axis=output_axis)

    canonical_directions = [
        signs[output_axis] * directions[permutation[output_axis]]
        for output_axis in range(3)
    ]
    return np.ascontiguousarray(canonical), canonical_directions, permutation, signs


def _log_display_frame(permutation, signs, canonical_directions, shape):
    logging.info(
        "DICOM display frame: NormOrientation is disabled on the scanner, so the "
        "volume is rotated into the standard view here (columns to L, rows to P, "
        "slices to H). Direction metadata is transformed with the pixels."
    )
    for output_axis, name in enumerate(DISPLAY_FRAME_AXIS_NAMES):
        logging.info(
            "Display frame axis %d (%s): source axis %d %s -> %s",
            output_axis,
            name,
            permutation[output_axis],
            "reversed" if signs[output_axis] < 0.0 else "kept",
            _format_direction(canonical_directions[output_axis]),
        )
    logging.info("Display frame shape: %s", tuple(int(size) for size in shape))

    slices, rows, columns = canonical_directions
    handedness = float(np.dot(np.cross(_unit_vector(columns), _unit_vector(rows)),
                              _unit_vector(slices)))
    if handedness > 0.0:
        logging.info(
            "Display frame handedness: right-handed (columns x rows . slices = "
            "%+.3f), matching the native reconstruction.",
            handedness,
        )
    else:
        logging.error(
            "Display frame handedness: LEFT-handed (columns x rows . slices = "
            "%+.3f). The scanner will draw this volume from the opposite side of "
            "the native reconstruction. An acquisition frame is always "
            "right-handed, so this means an axis was reversed without its "
            "direction vector, or DISPLAY_FRAME_TARGETS is inconsistent.",
            handedness,
        )


def _compensate_ice_frame_stacking(volume, slice_dir):
    """Reverse the emitted frame order for the scanner, deliberately breaking the header.

    Returns ``(volume, content_slice_dir, reversed_frames)``. ``slice_dir`` comes
    back unchanged when no compensation is applied.

    When it is applied only the pixels move. Negating the vector as well would
    cancel the reversal, because ICE positions the frames from that same vector,
    so there is no emitted volume that is both header-consistent and displayed
    correctly by this FIRE pipeline. This function chooses the scanner, and the
    resulting image therefore carries ``OUTPUT_FRAME_ORDER_REVERSED_ATTRIBUTE``
    so that consumers reading the header instead -- ``mrd2nifti`` above all --
    can undo it. Anything that reads ``slice_dir`` and ignores that attribute
    will place the volume mirrored through-plane.
    """
    if not ICE_STACKS_FRAMES_AGAINST_SLICE_DIR:
        return volume, np.asarray(slice_dir, dtype=float), False

    logging.warning(
        "ICE frame-stacking compensation applied: the emitted frame order is "
        "reversed because ICE positions frames against slice_dir. The emitted "
        "slice_dir still describes the acquisition, so the pixels and the "
        "header disagree by design and '%s' is set to 1 to declare it. The "
        "content runs along %s with increasing frame number. Any consumer that "
        "builds geometry from slice_dir must honour that attribute.",
        OUTPUT_FRAME_ORDER_REVERSED_ATTRIBUTE,
        _direction_label(-np.asarray(slice_dir, dtype=float)),
    )
    return (
        np.ascontiguousarray(np.flip(np.asarray(volume), axis=0)),
        -np.asarray(slice_dir, dtype=float),
        True,
    )


def _log_emitted_geometry(center_position, read_dir, phase_dir, slice_dir, shape, fov_mm):
    """Log what the scanner should display, so one screenshot can confirm or refute it.

    Everything here is derived from the header this app is about to emit. If the
    scanner shows something different, the header was overridden downstream
    rather than computed wrongly here, and that distinction is the whole reason
    this block exists: 0.1.3 could not be diagnosed from its own log.
    """
    slice_count = int(shape[0])
    spacing = float(fov_mm) / float(slice_count) if slice_count else 0.0

    left, right = _direction_letters(read_dir)
    top, bottom = _direction_letters(phase_dir)
    # The boxed marker is the side the viewer has to look from for the screen
    # basis to be proper, so it follows the DICOM normal columns x rows, not
    # slice_dir. That is why it read 'H' in 0.1.3 and 'F' in 0.1.4 even though
    # the emitted slice_dir was F->H both times.
    view_from, _ = _direction_letters(
        np.cross(_unit_vector(read_dir), _unit_vector(phase_dir))
    )

    logging.info(
        "Predicted scanner display: left edge '%s', right edge '%s', top edge "
        "'%s', bottom edge '%s', viewed from '%s' (the boxed marker). The native "
        "transversal reconstruction shows R, L, A, P and F.",
        left,
        right,
        top,
        bottom,
        view_from,
    )
    logging.info(
        "Emitted volume centre: %s [%s], slice spacing %.5f mm over %d slices",
        _position_label(center_position),
        ",".join(f"{float(value):+.3f}" for value in center_position),
        spacing,
        slice_count,
    )

    if slice_count:
        sample_indices = sorted(
            {0, slice_count // 4, slice_count // 2, (3 * slice_count) // 4, slice_count - 1}
        )
        for index in sample_indices:
            position = _frame_position(
                center_position, slice_dir, index, slice_count, spacing
            )
            logging.info(
                "Predicted frame %d/%d (scanner numbering): SP %s, full position %s",
                index + 1,
                slice_count,
                _slice_position_label(position),
                _position_label(position),
            )
        logging.info(
            "Those positions are where the emitted content sits, with the ICE "
            "frame-stacking compensation (%s) already accounted for. Compare any "
            "one of them against the scanner's slice position for the same frame "
            "number. A match confirms the geometry end to end; a sign difference "
            "means ICE has changed how it stacks frames and "
            "ICE_STACKS_FRAMES_AGAINST_SLICE_DIR must be flipped; a different "
            "magnitude means the volume centre or the field of view disagrees.",
            "on" if ICE_STACKS_FRAMES_AGAINST_SLICE_DIR else "off",
        )


def _log_patient_space_localisation(label, volume, center_position, axis_directions, fov_mm):
    """Log where the signal actually sits in patient coordinates.

    Voxel-index centroids cannot be compared against a scanner screenshot, but
    these positions can. They are what distinguishes a volume that is merely
    stored back to front from one whose content is genuinely in the wrong place.
    """
    values = np.asarray(volume, dtype=np.float64)
    if values.ndim != 3 or not values.size:
        logging.info("Patient-space localisation [%s]: unavailable", label)
        return

    total = float(values.sum())
    if total <= 0.0:
        logging.info("Patient-space localisation [%s]: no signal", label)
        return

    centroid_position = np.asarray(center_position, dtype=float).copy()
    for axis, name in enumerate(DISPLAY_FRAME_AXIS_NAMES):
        profile = values.sum(
            axis=tuple(other for other in range(3) if other != axis)
        )
        count = int(profile.size)
        spacing = float(fov_mm) / float(count)
        indices = np.arange(count, dtype=np.float64)
        centroid_index = float((profile * indices).sum() / profile.sum())
        peak_index = int(np.argmax(profile))

        above = np.flatnonzero(profile >= 0.1 * float(profile.max()))
        extent_mm = (float(above[-1] - above[0]) + 1.0) * spacing if above.size else 0.0

        offset_mm = (centroid_index - (count - 1) / 2.0) * spacing
        direction = _unit_vector(axis_directions[axis])
        centroid_position = centroid_position + offset_mm * direction

        logging.info(
            "Patient-space localisation [%s] %s (%s): centroid index %.2f/%d "
            "(%+.2f mm from centre), peak index %d, signal extent %.1f mm",
            label,
            name,
            _direction_label(direction),
            centroid_index,
            count,
            offset_mm,
            peak_index,
            extent_mm,
        )

    logging.info(
        "Patient-space localisation [%s]: intensity centroid at %s [%s]",
        label,
        _position_label(centroid_position),
        ",".join(f"{float(value):+.3f}" for value in centroid_position),
    )


def _log_volume_statistics(label, volume):
    values = np.asarray(volume, dtype=np.float64)
    if not values.size:
        logging.info("Volume statistics [%s]: empty", label)
        return

    total = float(values.sum())
    if total > 0.0:
        centroid = [
            float(
                (values.sum(axis=tuple(other for other in range(values.ndim) if other != axis))
                 * np.arange(values.shape[axis])).sum()
                / total
            )
            for axis in range(values.ndim)
        ]
        centroid_text = ",".join(
            f"{value:.2f}/{values.shape[axis]}" for axis, value in enumerate(centroid)
        )
    else:
        centroid_text = "undefined"

    logging.info(
        "Volume statistics [%s]: shape=%s dtype=%s min=%.6g max=%.6g mean=%.6g "
        "intensity_centroid_per_axis=%s",
        label,
        tuple(int(size) for size in values.shape),
        np.asarray(volume).dtype,
        float(values.min()),
        float(values.max()),
        float(values.mean()),
        centroid_text,
    )


def _log_reference_geometry(reference_head, metadata):
    try:
        patient_position = str(metadata.measurementInformation.patientPosition)
    except Exception:
        patient_position = "unavailable"

    logging.info(
        "Acquisition geometry: patient_position=%s position=(%s) "
        "patient_table_position=(%s)",
        patient_position,
        ",".join(f"{float(value):+.3f}" for value in reference_head.position),
        ",".join(
            f"{float(value):+.3f}"
            for value in getattr(reference_head, "patient_table_position", ())
        ),
    )
    logging.info("Acquisition read_dir  (MRD x, columns): %s", _format_direction(reference_head.read_dir))
    logging.info("Acquisition phase_dir (MRD y, rows):    %s", _format_direction(reference_head.phase_dir))
    logging.info("Acquisition slice_dir (MRD z, slices):  %s", _format_direction(reference_head.slice_dir))
    logging.info(
        "Acquisition volume centre: %s",
        _position_label(reference_head.position),
    )
    # Siemens builds its PRS frame so that phase x read = slice, which is the
    # opposite cross-product order from DICOM's column x row = normal. Measured
    # data therefore arrives DICOM-left-handed, and that is expected, not a
    # defect: read_dir=(-1,0,0), phase_dir=(0,1,0), slice_dir=(0,0,1) satisfies
    # the Siemens rule exactly. Only the emitted frame has to be DICOM
    # right-handed, and _log_display_frame checks that one.
    prs_handedness = float(
        np.dot(
            np.cross(
                _unit_vector(reference_head.phase_dir),
                _unit_vector(reference_head.read_dir),
            ),
            _unit_vector(reference_head.slice_dir),
        )
    )
    logging.info(
        "Acquisition frame: phase x read . slice = %+.3f (%s under the Siemens "
        "PRS convention). A value near zero would mean the incoming vectors are "
        "not orthonormal and no downstream mapping could be trusted.",
        prs_handedness,
        "consistent" if prs_handedness > 0.0 else "INCONSISTENT",
    )
    logging.info(
        "Geometry stages: 1) trajectory components mapped onto the acquisition "
        "axes, 2) that frame rotated into the DICOM display view with the "
        "direction vectors transformed together with the pixels, 3) the frame "
        "order reversed on its own to compensate for how ICE stacks a 3D volume. "
        "Stage 2 cannot change where the anatomy lands, only how it is stored; "
        "stage 3 is the only stage that moves it."
    )


def _log_acquisition_axes(packed_shape, reference_head, orientation_key, flip_slice):
    transpose_in_plane, reverse_rows, reverse_columns = (
        ORIENTATION_IN_PLANE_TRANSFORMS[orientation_key]
    )
    logging.info(
        "Trajectory orientation '%s': transpose_in_plane=%s reverse_rows=%s "
        "reverse_columns=%s reverse_slices=%s",
        orientation_key,
        transpose_in_plane,
        reverse_rows,
        reverse_columns,
        flip_slice,
    )
    logging.info(
        "Acquisition-frame volume: shape=%s, slices along %s, rows along %s, "
        "columns along %s",
        tuple(int(size) for size in packed_shape),
        _direction_label(reference_head.slice_dir),
        _direction_label(reference_head.phase_dir),
        _direction_label(reference_head.read_dir),
    )


def _build_output_images(
    volume,
    reference_head,
    metadata,
    output_fov_mm,
    orientation=DEFAULT_ORIENTATION,
    flip_slice=False,
    emit_debug_series=False,
):
    volume = np.asarray(volume, dtype=np.float32)
    if volume.ndim != 3:
        raise ValueError(f"Reconstructed volume must be 3D, got shape {volume.shape}")

    _log_reference_geometry(reference_head, metadata)
    _log_volume_statistics("reconstructed volume (z, y, x)", volume)

    display_volume, display_meta = _scale_volume_to_display_range(volume)
    logging.info(
        "Scanner display scaling: input_min=%.6g input_max=%.6g scale=%s formula='%s'",
        display_meta["input_min"],
        display_meta["input_max"],
        _format_display_number(display_meta["scale"]),
        display_meta["formula"],
    )

    # Accept every value the scanner UI offers, including the debug selection,
    # so passing a configured orientation straight through cannot silently fall
    # back to the default.
    orientation, selects_debug_series = _resolve_orientation_selection(orientation)
    emit_debug_series = emit_debug_series or selects_debug_series

    if emit_debug_series:
        # Sweep the slice reversal as well as the in-plane mappings. The eight
        # in-plane keys leave the through-plane axis untouched, so a sweep over
        # them alone cannot reach a volume whose slice order is wrong -- which is
        # precisely the failure 0.1.3 had, and precisely the one its debug sweep
        # could not have surfaced.
        sweep = [
            (key, slice_flip)
            for key in ORIENTATION_DEBUG_ORDER
            for slice_flip in (False, True)
        ]
        logging.warning(
            "Orientation debug sweep enabled: emitting %d canonicalized series, "
            "every combination of the %d trajectory mappings (%s) with the slice "
            "axis kept and reversed. Series are suffixed '_ori_<key>_fz<0|1>'. "
            "Identify the anatomically correct one against an asymmetric object "
            "or a marker on a known side, then set 'orientation' to its key and "
            "'orientationflipslice' to its fz digit. The four 'zxy' mappings are "
            "already excluded by the phantom aspect ratio measured in "
            "sodiumgridding_v0.1.3.PNG and are emitted only as a cross-check.",
            len(sweep),
            len(ORIENTATION_DEBUG_ORDER),
            ", ".join(ORIENTATION_DEBUG_ORDER),
        )
    else:
        sweep = [(orientation, flip_slice)]

    return [
        image
        for offset, (orientation_key, slice_flip) in enumerate(sweep)
        for image in _build_single_output_image(
            display_volume,
            display_meta,
            reference_head,
            metadata,
            output_fov_mm=output_fov_mm,
            orientation_key=orientation_key,
            flip_slice=slice_flip,
            series_index=OUTPUT_IMAGE_SERIES_INDEX + offset,
            label_series=emit_debug_series,
        )
    ]


def _build_single_output_image(
    display_volume,
    display_meta,
    reference_head,
    metadata,
    output_fov_mm,
    orientation_key,
    flip_slice,
    series_index,
    label_series,
):
    # Stage 1 resolves the trajectory components against the acquisition axes.
    # Stage 2 transforms the acquisition vectors together with the pixels in
    # _canonicalize_to_display_frame; no display correction changes metadata
    # without changing the corresponding pixels.
    read_dir = np.asarray(reference_head.read_dir, dtype=float)
    phase_dir = np.asarray(reference_head.phase_dir, dtype=float)
    slice_dir = np.asarray(reference_head.slice_dir, dtype=float)
    slice_dir_norm = float(np.linalg.norm(slice_dir))
    if slice_dir_norm < 1e-6:
        logging.warning("Acquisition slice_dir is degenerate; substituting +z (head)")
        slice_dir = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        slice_dir = slice_dir / slice_dir_norm

    center_position = np.asarray(reference_head.position, dtype=float)

    series_description = f"{_safe_protocol_name(metadata)}_{OUTPUT_SERIES_DESCRIPTION}"
    if label_series:
        series_description = (
            f"{series_description}_ori_{orientation_key}_fz{int(bool(flip_slice))}"
        )
    series_grouping = f"{series_description}_{series_index}"
    series_uid = _new_dicom_uid()
    output_fov_mm = float(output_fov_mm)
    image_comment = _scanner_display_comment(display_meta)

    # Pack the complete matrix as one explicit 3D MRD image. Sending 64 separate
    # 2D messages lets ICE refill each mini-header from the source protocol's
    # NoImagesPerSlab=32 and causes the DICOM writer to flush two 32-frame
    # volumes. One [z, y, x] image gives the writer one 64-frame volume,
    # matching the native ICE reconstruction contract.
    #
    # Stage 1 maps trajectory components into the acquisition's slice, phase
    # and read axes. Stage 2 rotates that acquisition frame into the standard
    # display view while transforming its direction vectors with the pixels.
    packed_volume, orientation_key = _orient_volume(
        display_volume,
        orientation_key,
        flip_slice=flip_slice,
    )
    _log_acquisition_axes(
        packed_volume.shape,
        reference_head,
        orientation_key,
        flip_slice,
    )

    packed_volume, canonical_directions, permutation, signs = (
        _canonicalize_to_display_frame(
            packed_volume,
            (slice_dir, phase_dir, read_dir),
        )
    )
    slice_dir, phase_dir, read_dir = canonical_directions
    _log_display_frame(permutation, signs, canonical_directions, packed_volume.shape)

    # Stage 3 compensates for the scanner rather than describing the data, so it
    # is the only place the emitted pixels stop matching the emitted slice_dir.
    packed_volume, content_slice_dir, frames_reversed = _compensate_ice_frame_stacking(
        packed_volume, slice_dir
    )
    _log_emitted_geometry(
        center_position,
        read_dir,
        phase_dir,
        content_slice_dir,
        packed_volume.shape,
        output_fov_mm,
    )

    slice_count = int(packed_volume.shape[0])
    _log_volume_statistics("packed output", packed_volume)
    _log_patient_space_localisation(
        "packed output",
        packed_volume,
        center_position,
        (content_slice_dir, phase_dir, read_dir),
        output_fov_mm,
    )
    spacing = float(output_fov_mm) / float(slice_count) if slice_count else float(output_fov_mm)

    def _make_image(frame_array, frame_index, frame_position, frame_slice_dir,
                    frame_field_of_view, number_in_series):
        image = ismrmrd.Image.from_array(frame_array, transpose=False)

        new_header = mrdhelper.update_img_header_from_raw(image.getHead(), reference_head)
        new_header.data_type = image.data_type
        new_header.image_type = ismrmrd.IMTYPE_MAGNITUDE

        # The Pulseq interpreter leaves acquisition_time_stamp and
        # physiology_time_stamp at zero -- a product sequence fills them from the
        # MDH, an interpreted one does not. Passing the zero through makes ICE fail
        # in m_pTimeOfFirstScan->setValue, which cascades into
        # calcAcquisitionDateAndTime and FillMiniHead and stops the measurement,
        # with nothing in the FIRE log to explain it. Substitute a valid tick count
        # so the mini header can be built. MRD timestamps count 2.5 ms ticks since
        # midnight; the value is synthetic, not a measured acquisition time.
        if not int(getattr(new_header, "acquisition_time_stamp", 0) or 0):
            now = time.localtime()
            ticks = int(((now.tm_hour * 3600 + now.tm_min * 60 + now.tm_sec) * 1000) / 2.5)
            new_header.acquisition_time_stamp = ticks
            if frame_index == 0:
                logging.info(
                    "Acquisition time stamp was zero (the Pulseq interpreter does not "
                    "set it); substituting %d so ICE can build the mini header", ticks)
        if not any(int(v or 0) for v in getattr(new_header, "physiology_time_stamp", ()) or ()):
            new_header.physiology_time_stamp = (int(new_header.acquisition_time_stamp), 0, 0)
        new_header.image_series_index = series_index
        new_header.image_index = frame_index + 1
        new_header.slice = 0
        new_header.matrix_size = tuple(int(value) for value in image.getHead().matrix_size)
        new_header.position = tuple(float(value) for value in frame_position)
        new_header.read_dir = tuple(float(value) for value in read_dir)
        new_header.phase_dir = tuple(float(value) for value in phase_dir)
        new_header.slice_dir = tuple(float(value) for value in frame_slice_dir)
        new_header.field_of_view = tuple(float(v) for v in frame_field_of_view)
        image.setHead(new_header)
        image.image_series_index = series_index
        image.field_of_view = tuple(ctypes.c_float(float(v)) for v in frame_field_of_view)

        meta = ismrmrd.Meta()
        meta["DataRole"] = "Image"
        meta["ImageProcessingHistory"] = ["PYTHON", "NUMBA", "KAISERBESSEL", "GRIDDING"]
        meta["ImageType"] = "DERIVED\\PRIMARY\\M\\" + META_IMAGE_TYPE
        meta["DicomImageType"] = "DERIVED\\PRIMARY\\M\\" + META_IMAGE_TYPE
        meta["ImageTypeValue4"] = META_IMAGE_TYPE
        meta["ComplexImageComponent"] = "MAGNITUDE"
        meta["SequenceDescriptionAdditional"] = OUTPUT_IMAGE_COMMENT
        meta["SeriesDescription"] = series_description
        meta["SequenceDescription"] = series_description
        meta["ProtocolName"] = series_description
        meta["SeriesNumberRangeNameUID"] = series_grouping
        meta["SeriesInstanceUID"] = series_uid
        meta["SOPInstanceUID"] = _new_dicom_uid()
        meta["ImageComment"] = image_comment
        meta["ImageComments"] = image_comment
        # 1 tells ICE to keep the geometry described in this header instead of
        # rebuilding it and applying its own flip/shift. The in-plane vectors
        # describe the emitted pixels. slice_dir does not: stage 3 reverses the
        # frames against it on purpose, because ICE stacks them that way, and
        # OUTPUT_FRAME_ORDER_REVERSED_ATTRIBUTE below declares that so a consumer
        # reading this header can undo it.
        meta["Keep_image_geometry"] = 1
        meta["partition_count"] = number_in_series
        meta["slice_count"] = slice_count
        meta["NumberOfSlices"] = slice_count
        meta["ImagesInAcquisition"] = slice_count
        meta["NumberInSeries"] = number_in_series
        meta["SliceNo"] = 0
        meta["IsmrmrdSliceNo"] = 0
        meta["AnatomicalSliceNo"] = 0
        meta["ChronSliceNo"] = 0
        meta["ProtocolSliceNumber"] = 0
        meta["Actual3DImagePartNumber"] = frame_index
        meta["Actual3DImaPartNumber"] = frame_index
        meta["AnatomicalPartitionNo"] = frame_index
        meta["ImageRowDir"] = [f"{float(value):.18f}" for value in read_dir]
        meta["ImageColumnDir"] = [f"{float(value):.18f}" for value in phase_dir]
        meta["ImageSliceNormDir"] = [f"{float(value):.18f}" for value in slice_dir]
        meta["SlicePosLightMarker"] = [
            f"{float(value):.18f}" for value in new_header.position
        ]
        meta[META_PREFIX + "DisplayScale"] = _format_display_number(display_meta["scale"])
        meta[META_PREFIX + "DisplayInputMin"] = f"{float(display_meta['input_min']):.6g}"
        meta[META_PREFIX + "DisplayInputMax"] = f"{float(display_meta['input_max']):.6g}"
        meta[META_PREFIX + "DisplayMin"] = str(int(display_meta["display_min"]))
        meta[META_PREFIX + "DisplayMax"] = str(int(display_meta["display_max"]))
        meta[META_PREFIX + "DisplayFormula"] = display_meta["formula"]
        meta[META_PREFIX + "Orientation"] = orientation_key
        meta[META_PREFIX + "OrientationFlipSlice"] = str(int(bool(flip_slice)))
        meta[OUTPUT_FRAME_ORDER_REVERSED_ATTRIBUTE] = str(int(bool(frames_reversed)))
        image.attribute_string = meta.serialize()

        if frame_index == 0:
            logging.info(
            "Emitting image: series_index=%d series_description='%s' matrix_size=%s "
            "slice_count=%d fov_mm=%.3f orientation='%s' flip_slice=%s "
            "columns=%s rows=%s slices=%s "
            "Keep_image_geometry=%s series_uid=%s",
            series_index,
            series_description,
            tuple(int(value) for value in new_header.matrix_size),
            slice_count,
            output_fov_mm,
            orientation_key,
            flip_slice,
            _direction_label(read_dir),
            _direction_label(phase_dir),
            _direction_label(slice_dir),
            meta["Keep_image_geometry"],
            series_uid,
            )

        return image

    if EMIT_VOLUME_AS_SINGLE_IMAGE:
        images = [
            _make_image(
                packed_volume,
                0,
                center_position,
                slice_dir,
                (output_fov_mm, output_fov_mm, output_fov_mm),
                1,
            )
        ]
        logging.info("Emitted 1 image carrying the whole %d-slice volume", slice_count)
        return images

    # One 2D image per slice. IceProgramStandard's image chain crashes on a
    # single 3D MRD image; per-slice messages are the conventional FIRE
    # contract. Each frame carries its own patient-space position along the
    # direction the content actually runs in, so the header and the pixels
    # agree without the stacking compensation having to be inferred.
    images = [
        _make_image(
            packed_volume[frame_index],
            frame_index,
            _frame_position(
                center_position, content_slice_dir, frame_index, slice_count, spacing
            ),
            content_slice_dir,
            (output_fov_mm, output_fov_mm, spacing),
            slice_count,
        )
        for frame_index in range(slice_count)
    ]
    logging.info(
        "Emitted %d single-slice images, %.5f mm apart along %s, series_uid=%s",
        len(images),
        spacing,
        _direction_label(content_slice_dir),
        series_uid,
    )
    return images

# ---------------------------------------------------------------------------
# Public aliases. These are the framework pieces an algorithm normally reuses;
# import them from your app module rather than reimplementing them.
#
# Note there is no config accessor here on purpose: ReconInput.config is
# already resolved and type-coerced, so read it with plain dict access. The
# _config_* helpers only work on the raw {"parameters": {...}} dict the MRD
# server sends, which an algorithm never sees.
# ---------------------------------------------------------------------------
estimate_sensitivities = _estimate_sensitivities
combine_coils = _combine_coils
compress_coils_by_variance = _compress_coils_by_variance
n4_bias_field_correct = _n4_bias_field_correct
build_fermi_filter = _build_fermi_filter


# ---------------------------------------------------------------------------
# What your reconstruct() hook receives.
# ---------------------------------------------------------------------------
@dataclass
class ReconInput:
    """Everything a reconstruction needs, after the standard preparation steps.

    Attributes:
        kspace: ``(coils, samples, readouts)`` complex64. Bad readouts zeroed,
            centre-of-k-space normalised, Fermi-filtered, and PCA-compressed --
            whichever of those the config enabled. ``coils`` counts *virtual*
            coils when ``compresscoils`` is on.
        trajectory: ``(samples, readouts, 3)`` float32 in the trajectory file's
            native units (cycles/cm for the bundled sodium trajectories).
        raw_kspace: ``(coils, readouts, samples)`` complex64, straight off the
            wire with no preparation at all. Use this if your algorithm wants to
            do its own conditioning.
        matrix_size: cubic output matrix, N.
        fov_cm: field of view in cm, used to scale the trajectory.
        config: every resolved OpenRecon parameter, defaults already applied.
        metadata: the ISMRMRD header.
        compression_matrix: ``(physical, virtual)`` PCA basis, or None.

    The two coordinate helpers cover the conventions the common NUFFT
    implementations expect; both return a flat ``(samples * readouts, 3)``
    array whose point order matches ``kspace[c].ravel()``.
    """

    kspace: np.ndarray
    trajectory: np.ndarray
    raw_kspace: np.ndarray
    matrix_size: int
    fov_cm: float
    config: dict
    metadata: Any
    trajectory_source: Any = None
    compression_matrix: Any = None
    cumulative_variance: Any = None
    max_workers: int = 1

    @property
    def num_coils(self):
        return int(self.kspace.shape[0])

    @property
    def image_shape(self):
        return (self.matrix_size, self.matrix_size, self.matrix_size)

    def coords_normalized(self):
        """Coordinates on [-0.5, 0.5], the convention hand-written gridding uses."""
        coords = np.asarray(self.trajectory, dtype=np.float32).reshape(-1, 3)
        coords = coords * (float(self.fov_cm) / float(self.matrix_size))
        extent = float(np.max(np.abs(coords))) if coords.size else 0.0
        if extent > 0.5 + 1e-5:
            logging.warning(
                "Normalized trajectory extends beyond the gridding FOV: max_abs=%.6f",
                extent,
            )
        return np.asarray(coords, dtype=np.float32)

    def coords_index_units(self):
        """Coordinates on [-N/2, N/2], the convention sigpy's nufft expects."""
        coords = np.asarray(self.trajectory, dtype=np.float32).reshape(-1, 3)
        return np.asarray(coords * float(self.fov_cm), dtype=np.float32)

    def radial_k(self):
        """``|k|`` per sample, shape ``(samples, readouts)``, native units."""
        return np.linalg.norm(self.trajectory, axis=-1)

    def save_debug(self, name, array):
        """Drop an array into the debug folder for offline inspection."""
        _ensure_debug_folder()
        np.save(os.path.join(debugFolder, f"{RECON_NAME}_{name}.npy"), np.asarray(array))


def _resolve_config(config, metadata):
    """Merge the OpenRecon UI values over the defaults, with type coercion."""
    resolved = dict(OPENRECON_DEFAULTS)
    for key, default in OPENRECON_DEFAULTS.items():
        if isinstance(default, bool):
            resolved[key] = _config_bool(config, key, default)
        elif isinstance(default, int):
            resolved[key] = _config_int(config, key, default)
        elif isinstance(default, float):
            resolved[key] = _config_float(config, key, default)
        else:
            resolved[key] = _config_str(config, key, default)

    resolved["matrixsize"] = max(
        1, _config_int(config, "matrixsize", _compute_default_matrix_size(metadata))
    )
    resolved["fovcm"] = _config_float(config, "fovcm", _compute_default_fov_cm(metadata))
    resolved["centerwindow"] = max(1, int(resolved["centerwindow"]))
    resolved["maxcoils"] = max(0, int(resolved["maxcoils"]))
    resolved["maxworkers"] = max(1, int(resolved["maxworkers"]))
    resolved["trajectorysampleoffset"] = max(0, int(resolved["trajectorysampleoffset"]))
    resolved["coilvarianceretention"] = float(
        np.clip(resolved["coilvarianceretention"], 0.0, 1.0)
    )
    return resolved


def prepare(group, config, metadata):
    """Run the standard preparation and return a :class:`ReconInput`."""
    params = _resolve_config(config, metadata)

    data = _build_data_array(group)
    # Ordering is reported before the trajectory is resolved: if the trajectory
    # gate rejects the acquisition, its shape and any gap are exactly what the
    # log needs to explain why.
    row_indices = _readout_row_indices(group)
    trajectory = _load_trajectory(group, config)
    data, trajectory = _clip_data_to_trajectory(
        data,
        trajectory,
        sample_offset=params["trajectorysampleoffset"],
        row_indices=row_indices,
    )

    if 0 < params["maxcoils"] < data.shape[0]:
        logging.warning(
            "Limiting reconstruction to first %d of %d coils",
            params["maxcoils"],
            data.shape[0],
        )
        data = data[: params["maxcoils"]]

    logging.info(
        "Prepared input: coils=%d readouts=%d samples=%d matrix=%d fov_cm=%.3f",
        data.shape[0],
        data.shape[1],
        data.shape[2],
        params["matrixsize"],
        params["fovcm"],
    )
    _log_trajectory_extent(trajectory)

    fermi_filter = _build_fermi_filter(
        trajectory,
        apply_fermi_filter=params["applyfermifilter"],
        fermi_width=params["fermiwidth"],
        fermi_cutoff=params["fermicutoff"],
    )

    prepared = np.asarray(
        [
            _prepare_single_coil_data(
                coil_index,
                data[coil_index],
                reject_bad_readouts=params["rejectbadreadouts"],
                bad_readout_sigma=params["badreadoutsigma"],
                center_window=params["centerwindow"],
                fermi_filter=fermi_filter,
            )
            for coil_index in range(data.shape[0])
        ],
        dtype=np.complex64,
    )

    compression_matrix = None
    cumulative_variance = None
    if params["compresscoils"]:
        prepared, cumulative_variance, compression_matrix = _compress_coils_by_variance(
            prepared, variance_retention=params["coilvarianceretention"]
        )

    max_workers = min(params["maxworkers"], max(1, int(prepared.shape[0])))
    _log_cpu_resources(params["maxworkers"], max_workers)

    return ReconInput(
        kspace=prepared,
        trajectory=np.asarray(trajectory, dtype=np.float32),
        raw_kspace=data,
        matrix_size=params["matrixsize"],
        fov_cm=params["fovcm"],
        config=params,
        metadata=metadata,
        trajectory_source=LAST_TRAJECTORY_SOURCE,
        compression_matrix=compression_matrix,
        cumulative_variance=cumulative_variance,
        max_workers=max_workers,
    )


def process_raw(group, connection, config, metadata, reconstruct):
    """Prepare, hand off to the algorithm, then emit scanner images."""
    if not group:
        return []

    tic = perf_counter()
    _ensure_debug_folder()

    recon = prepare(group, config, metadata)
    params = recon.config
    logging.info("Resolved configuration: %s", params)

    reference_head = group[len(group) // 2].getHead()

    volume = reconstruct(recon)
    volume = np.asarray(volume, dtype=np.float32)
    if volume.shape != recon.image_shape:
        raise ValueError(
            f"reconstruct() must return a {recon.image_shape} volume, "
            f"got {volume.shape}"
        )
    _log_volume_statistics("algorithm output", volume)

    if params["applyn4biascorrection"]:
        logging.info("Running N4 bias field correction")
        volume = _n4_bias_field_correct(volume)
        logging.info("Finished N4 bias field correction")

    orientation, orientation_debug_series = _resolve_orientation_config(config)
    for hook in OUTPUT_VOLUME_HOOKS:
        volume = np.asarray(
            hook(volume, dict(reference_head=reference_head, metadata=metadata, params=params,
                              output_fov_mm=float(params["fovcm"]) * 10.0,
                              orientation=orientation, flip_slice=params["orientationflipslice"])),
            dtype=np.float32)
        if volume.shape != recon.image_shape:
            raise ValueError(f"output hook {hook!r} changed the volume shape to {volume.shape}")
        _log_volume_statistics(f"after {getattr(hook, '__name__', 'hook')}", volume)

    recon.save_debug("output_volume", volume)

    message = f"{RECON_NAME} processing time: {(perf_counter() - tic) * 1000.0:.2f} ms"
    logging.info(message)
    connection.send_logging(constants.MRD_LOGGING_INFO, message)

    return _build_output_images(
        volume,
        reference_head,
        metadata,
        output_fov_mm=float(params["fovcm"]) * 10.0,
        orientation=orientation,
        flip_slice=params["orientationflipslice"],
        emit_debug_series=orientation_debug_series,
    )


def process_image(images, connection, config, metadata):
    """Pass images through untouched; this framework only handles raw data."""
    del connection, config, metadata
    logging.info("Passing through %d images unchanged", len(images))
    return images


def run(connection, config, metadata, reconstruct):
    """MRD server entry point. Call this from your app module's ``process``."""
    logging.info("Config:\n%s", config)

    try:
        logging.info("Incoming dataset contains %d encodings", len(metadata.encoding))
        logging.info(
            "First encoding trajectory=%s matrix=(%s x %s x %s) fov=(%s x %s x %s)mm^3",
            metadata.encoding[0].trajectory,
            metadata.encoding[0].encodedSpace.matrixSize.x,
            metadata.encoding[0].encodedSpace.matrixSize.y,
            metadata.encoding[0].encodedSpace.matrixSize.z,
            metadata.encoding[0].encodedSpace.fieldOfView_mm.x,
            metadata.encoding[0].encodedSpace.fieldOfView_mm.y,
            metadata.encoding[0].encodedSpace.fieldOfView_mm.z,
        )
    except Exception:
        logging.info("Improperly formatted metadata: %s", metadata)

    acquisitions = []
    passthrough_images = []
    last_flag_at = None

    try:
        for item in connection:
            if isinstance(item, ismrmrd.Acquisition):
                if (
                    not item.is_flag_set(ismrmrd.ACQ_IS_NOISE_MEASUREMENT)
                    and not item.is_flag_set(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION)
                    and not item.is_flag_set(ismrmrd.ACQ_IS_PHASECORR_DATA)
                    and not item.is_flag_set(ismrmrd.ACQ_IS_NAVIGATION_DATA)
                ):
                    acquisitions.append(item)

                if item.is_flag_set(ismrmrd.ACQ_LAST_IN_MEASUREMENT):
                    # Deliberately not a trigger to reconstruct. ICE has been
                    # observed setting this flag before the last readout -- 128
                    # short of 25536 once LABEL extensions moved the trigger mode
                    # to PerLastScanInMeas. Reconstructing here stops the loop
                    # reading, leaves the remainder in the socket buffer for the
                    # length of the reconstruction, and then starts a second one
                    # on the stragglers. Drain the stream instead and
                    # reconstruct once at the end.
                    last_flag_at = len(acquisitions)
                    logging.info(
                        "ACQ_LAST_IN_MEASUREMENT seen at readout %d; "
                        "continuing to drain the stream before reconstructing",
                        last_flag_at,
                    )

            elif isinstance(item, ismrmrd.Image):
                passthrough_images.append(item)

            elif item is None:
                break

            else:
                logging.error("Unsupported data type %s", type(item).__name__)

        if acquisitions:
            if last_flag_at is not None and last_flag_at != len(acquisitions):
                logging.warning(
                    "%d readout(s) arrived after ACQ_LAST_IN_MEASUREMENT (flagged at "
                    "%d, received %d); all of them are being reconstructed together",
                    len(acquisitions) - last_flag_at,
                    last_flag_at,
                    len(acquisitions),
                )
            logging.info(
                "Processing %d acquired readouts (end of stream)", len(acquisitions)
            )
            connection.send_image(
                process_raw(acquisitions, connection, config, metadata, reconstruct)
            )

        if passthrough_images:
            logging.warning(
                "Received %d images instead of raw data; returning them unchanged",
                len(passthrough_images),
            )
            connection.send_image(
                process_image(passthrough_images, connection, config, metadata)
            )

    except Exception:
        logging.error(traceback.format_exc())
        connection.send_logging(constants.MRD_LOGGING_ERROR, traceback.format_exc())

    finally:
        connection.send_close()
