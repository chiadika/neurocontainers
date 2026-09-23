#!/usr/bin/env python3
"""quickgrid: the simplest reconstruction that produces a usable image.

    image = combine_coils( NUFFT^H ( w * y ) )

One adjoint NUFFT per coil with density compensation, then a coil combination.
No calibration, no iterations, no regularisation. At 128^3 with 25k spokes and
8 coils this runs in seconds; it exists to check that data, trajectory, labels
and the ICE handoff are all right before spending minutes on CG-SENSE.

Everything about talking to the scanner -- receiving readouts, aligning them to
the trajectory by scan_counter, choosing the trajectory from the data shape,
coil compression, orientation, per-slice emission with correct geometry --
lives in mrdrecon.py and is shared with cgsense3d. This file only has to turn a
ReconInput into a real-valued volume.

Where to edit
-------------
  DEFAULTS            the knobs and their defaults (must mirror OpenReconLabel.json)
  gradient_unwarp()   gradient-nonlinearity (distortion) correction hook, maths in gradwarp.py
  density_compensation()   how each k-space sample is weighted
  grid()              coil-parallel adjoint NUFFT; the gridder itself is _grid_coils()
  combine()           how coil images become one image
  reconstruct()       the four-line pipeline that calls the above

Offline check without a scanner:
    python /opt/code/python-ismrmrd-server/run_local.py raw.h5 -o out.nii.gz -m quickgrid
"""

import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

import h5py
import numpy as np
import sigpy
import sigpy.mri

import gradwarp

import mrdrecon

RECON_NAME = "quickgrid"

# Files baked into the image. `trajectoryfile: auto` picks the one whose
# (samples, readouts) matches the acquired data; a wrong pick is refused rather
# than reconstructed, because phyllotaxis directions depend on the spoke count.
BUNDLED_TRAJECTORIES = {
    "radial3d_res128_us2x": f"/opt/{RECON_NAME}/radial3d_res128_us2x_trajectory.h5",
    "radial3d_res128_nyquist": f"/opt/{RECON_NAME}/radial3d_res128_nyquist_trajectory.h5",
    "radial3d_n6434": f"/opt/{RECON_NAME}/radial3d_n6434_trajectory.h5",
    "cones3d_n4846": f"/opt/{RECON_NAME}/cones3d_n4846_trajectory.h5",
    "cones3d_n9598": f"/opt/{RECON_NAME}/cones3d_n9598_trajectory.h5",
    "cones3d_n68785": f"/opt/{RECON_NAME}/cones3d_n68785_trajectory.h5",   # 256^3 / 250 mm / TE 50, matrix 266
    "cones3d_n34506": f"/opt/{RECON_NAME}/cones3d_n34506_trajectory.h5",   # same, R=2, matrix 278
    "cones3d_n206435": f"/opt/{RECON_NAME}/cones3d_n206435_trajectory.h5", # 512^3 / 250 mm / TR 2, R=4, matrix 512
    "cones3d_n412489": f"/opt/{RECON_NAME}/cones3d_n412489_trajectory.h5", # 512^3 / 250 mm / TR 2, R=2, matrix 512
}

# `auto` also searches the scanner's `fire\share` folder (mounted as /tmp/share,
# mrdrecon.TRAJECTORY_SEARCH_DIRS) for *_trajectory.h5, so a trajectory that is
# not baked in only has to be copied there. Entries here just name them in the UI.
SHARE_TRAJECTORIES = {}

# Keep these in step with OpenReconLabel.json (which is what the scanner JSON
# config is generated from). Anything not listed there can still be set here.
DEFAULTS = {
    "config": RECON_NAME,
    "matrixsize": 128,          # 134 for the cones trajectories
    "fovcm": 22.0,
    "trajectoryfile": "auto",
    "trajectorydataset": "k",
    "dcfmode": "auto",          # file -> analytic -> pipe
    "dcfiterations": 10,        # only if Pipe-Menon has to run at recon time
    "coilcombinemode": "SoS",   # SoS or AC (adaptive combine)
    "applyfermifilter": False,  # k-space apodisation before gridding
    "fermiwidth": 0.05,
    "fermicutoff": 0.98,
    "rejectbadreadouts": False,
    "compresscoils": True,
    "coilvarianceretention": 0.95,
    "maxcoils": 0,              # 0 = all physical coils (then PCA compression); >0 keeps the first N
    "maxworkers": 8,            # coil-parallel gridding processes (capped by CPUs and memory)
    "applyn4biascorrection": False,
    "orientation": "zyx",
    "orientationflipslice": False,
    "gradunwarp": "3D",         # off | 3D | 3Dnojac | 2D  (gradient-nonlinearity correction)
    "gradcoeffile": "auto",     # Siemens coeff_<coil>.grad: 'auto' searches /opt/quickgrid and /tmp/share
}

mrdrecon.configure(
    name=RECON_NAME,
    defaults=DEFAULTS,
    trajectories=BUNDLED_TRAJECTORIES,
    image_comment="3D non-Cartesian gridding",
    meta_prefix="QuickGrid",
)


# ---------------------------------------------------------------------------
# Density compensation  (weights w, one per k-space sample)
# ---------------------------------------------------------------------------
DCF_DATASET = "dcf"


def _dcf_from_file(recon, expected_shape):
    """A precomputed `dcf` dataset shipped beside `k` (make_dcf.py). Free."""
    source = getattr(recon, "trajectory_source", None)
    if not source or not os.path.exists(source):
        return None
    with h5py.File(source, "r") as handle:
        if DCF_DATASET not in handle:
            return None
        dcf = np.asarray(handle[DCF_DATASET][...], dtype=np.float32)
        method = handle[DCF_DATASET].attrs.get("method", "unknown")
    if dcf.size != int(np.prod(expected_shape)):
        raise ValueError(
            f"'{DCF_DATASET}' in {source} has {dcf.size} values but the trajectory "
            f"has {int(np.prod(expected_shape))} samples; regenerate it with make_dcf.py."
        )
    logging.info("DCF: precomputed '%s' from %s (method=%s)",
                 DCF_DATASET, os.path.basename(source), method)
    return np.ascontiguousarray(dcf.reshape(-1))


def _looks_uniform_radial(trajectory, tol=0.05):
    """Every spoke runs monotonically outward with constant step: w = |k|^2 is exact."""
    radius = np.linalg.norm(trajectory, axis=-1)
    if radius.ndim != 2 or radius.shape[0] < 3:
        return False
    steps = np.diff(radius, axis=0)
    return bool(np.all(steps > 0)) and float(np.std(steps) / max(float(np.mean(steps)), 1e-12)) <= tol


def _dcf_analytic_radial(recon):
    w = np.square(recon.radial_k(), dtype=np.float32).ravel()
    w /= max(float(w.max()), 1e-12)
    positive = w[w > 0]
    if positive.size:
        w = np.maximum(w, float(positive.min()))  # k=0 sample is the best SNR point, keep it
    return np.ascontiguousarray(w, dtype=np.float32)


def _dcf_pipe_menon(recon, coord, iterations):
    logging.warning("DCF: Pipe-Menon at reconstruction time over %d points (slow). "
                    "Precompute with make_dcf.py and ship it in the trajectory file.",
                    coord.shape[0])
    dcf = sigpy.mri.dcf.pipe_menon_dcf(coord, recon.image_shape,
                                       max_iter=int(iterations), show_pbar=False)
    dcf = np.asarray(np.real(dcf), dtype=np.float32)
    return np.ascontiguousarray(dcf / max(float(dcf.max()), 1e-12))


def density_compensation(recon, coord):
    """Pick weights by `dcfmode`: file | analytic | pipe | auto (in that order)."""
    mode = str(recon.config.get("dcfmode", "auto")).strip().lower()
    if mode not in ("auto", "file", "analytic", "pipe"):
        raise ValueError(f"dcfmode must be auto, file, analytic or pipe; got {mode!r}")

    if mode in ("auto", "file"):
        stored = _dcf_from_file(recon, recon.trajectory.shape[:2])
        if stored is not None:
            return stored
        if mode == "file":
            raise ValueError(f"dcfmode='file' but no '{DCF_DATASET}' dataset in "
                             f"{getattr(recon, 'trajectory_source', None)!r}")

    if mode == "analytic" or (mode == "auto" and _looks_uniform_radial(recon.trajectory)):
        logging.info("DCF: analytic |k|^2 (uniform radial)")
        return _dcf_analytic_radial(recon)

    return _dcf_pipe_menon(recon, coord, int(recon.config.get("dcfiterations", 10)))


# ---------------------------------------------------------------------------
# Gridding and coil combination
# ---------------------------------------------------------------------------
def grid(kspace, coord, weights, shape, workers=1, mode="collect"):
    """Adjoint NUFFT per coil, spread over `workers` processes.

    (coils, samples*readouts) -> (coils, *shape) when mode == "collect", or
    the sum of squared magnitudes over coils, (*shape) float64, when
    mode == "sos" (never holds every coil image at once).

    `coord` is in grid-index units [-N/2, N/2] (sigpy's convention). sigpy's
    gridding is numba code that holds the GIL, so threads would serialise on
    it; processes are forked after the inputs are staged in module globals and
    read them copy-on-write, so nothing large is pickled on the way in. One
    coil per task in fixed coil order, partials summed in that order in
    float64, so the result does not depend on the worker count. To use a
    different gridder, replace `_grid_coils`; keep its output shape.
    """
    y = kspace.reshape(kspace.shape[0], -1).astype(np.complex64, copy=False)
    n_coils = int(y.shape[0])
    workers = max(1, min(int(workers), n_coils))
    tasks = [[c] for c in range(n_coils)]
    global _STAGED
    _STAGED = dict(y=y, coord=np.ascontiguousarray(coord, dtype=np.float32),
                   weights=np.ascontiguousarray(weights, dtype=np.float32),
                   shape=_shape_tuple(shape), mode=mode)
    try:
        if workers == 1:
            parts = [_grid_task(t) for t in tasks]
        else:
            _warm_up_gridder()  # JIT once here; forked workers inherit the compiled code
            logging.info("quickgrid: gridding %d coils across %d processes", n_coils, workers)
            ctx = multiprocessing.get_context("fork")
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
                parts = list(pool.map(_grid_task, tasks))  # fixed task order
    finally:
        _STAGED = None
    if mode == "sos":
        total = np.zeros(_shape_tuple(shape), dtype=np.float64)
        for part in parts:
            total += part
        return total
    return np.concatenate(parts, axis=0).astype(np.complex64, copy=False)


_STAGED = None


def _shape_tuple(shape):
    return tuple(int(n) for n in shape)


def _grid_coils(y, coord, shape):
    """The gridder proper: (n, points) complex64 -> (n, *shape) complex64."""
    images = sigpy.nufft_adjoint(y, coord, (y.shape[0],) + tuple(shape))
    return np.asarray(images, dtype=np.complex64)


def _grid_task(coils):
    st = _STAGED
    y = (st["y"][coils] * st["weights"][None, :]).astype(np.complex64, copy=False)
    images = _grid_coils(y, st["coord"], st["shape"])
    if st["mode"] == "sos":
        return np.sum(np.abs(images).astype(np.float64) ** 2, axis=0)
    return images


def _warm_up_gridder():
    _grid_coils(np.ones((1, 2), np.complex64), np.zeros((2, 3), np.float32), (8, 8, 8))


def worker_count(requested, n_coils, shape):
    """Processes to use: the request, capped by coils, usable CPUs and memory.

    Each worker holds roughly three oversampled grids (sigpy oversamples by
    1.25) plus its output; 60 % of MemAvailable is the budget, the rest stays
    with the parent, which already holds the k-space.
    """
    requested = max(1, int(requested))
    try:
        cpus = len(os.sched_getaffinity(0))
    except Exception:
        cpus = os.cpu_count() or 1
    dims = _shape_tuple(shape)
    per_worker = 3 * int(np.prod([int(np.ceil(1.25 * d)) for d in dims])) * 8 + int(np.prod(dims)) * 8
    budget = _available_memory_bytes()
    by_memory = max(1, int(0.6 * budget // per_worker)) if budget else requested
    workers = max(1, min(requested, n_coils, cpus, by_memory))
    logging.info("quickgrid: workers=%d (requested %d, coils %d, cpus %d, memory allows %d at %.2f GB each)",
                 workers, requested, n_coils, cpus, by_memory, per_worker / 1e9)
    return workers


def _available_memory_bytes():
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return 0


def combine(coil_images, mode):
    """Coil images -> one image. 'SoS' is safest; 'AC' is adaptive combine."""
    mode = str(mode).strip()
    if mode.lower() == "sos":
        return np.sqrt(np.sum(np.abs(coil_images) ** 2, axis=0)).astype(np.float32)
    return np.abs(mrdrecon.combine_coils(coil_images, mode=mode)).astype(np.float32)


# ---------------------------------------------------------------------------
# The reconstruction
# ---------------------------------------------------------------------------
def reconstruct(recon):
    coord = recon.coords_index_units()           # (points, 3) on [-N/2, N/2]
    extent = float(np.abs(coord).max())
    logging.info("quickgrid: %d coils, %d k-space points, matrix %d, |k|max %.2f of %d",
                 recon.num_coils, coord.shape[0], recon.matrix_size, extent,
                 recon.matrix_size // 2)
    if extent > 0.5 * recon.matrix_size + 1e-3:
        logging.warning("Trajectory exceeds the matrix (|k|max %.2f > %d); samples outside "
                        "will be dropped. Raise matrixsize (cones needs 134).",
                        extent, recon.matrix_size // 2)

    weights = density_compensation(recon, coord)
    mode = str(recon.config.get("coilcombinemode", "SoS")).strip()
    workers = worker_count(recon.config.get("maxworkers", 1), recon.num_coils, recon.image_shape)
    if mode.lower() == "sos":
        # streaming: each task returns sum|img|^2 for its coil, so the full
        # (coils, N, N, N) stack is never materialised
        sos = grid(recon.kspace, coord, weights, recon.image_shape, workers=workers, mode="sos")
        return np.sqrt(sos).astype(np.float32)
    coil_images = grid(recon.kspace, coord, weights, recon.image_shape, workers=workers)
    volume = combine(coil_images, mode)
    recon.save_debug("coil_images", coil_images)
    return volume


def gradient_unwarp(volume, context):
    """Output hook: correct gradient nonlinearity in the acquisition frame.

    Skips with a warning (never fails the reconstruction) when switched off or
    when no coefficient file can be found. The file is Siemens-proprietary and
    is not bundled; copy coeff_<coil>.grad into fire\\share (/tmp/share).
    """
    params = context["params"]
    mode = str(params.get("gradunwarp", "off")).strip()
    if mode.lower() in ("", "off", "false", "0"):
        logging.info("gradunwarp: off")
        return volume
    coil = None
    try:
        coil = str(context["metadata"].acquisitionSystemInformation.systemModel)
    except Exception:
        pass
    path = gradwarp.find_coefficient_file(params.get("gradcoeffile", "auto"), coil_name="IMPULSE")
    if path is None or not os.path.exists(path):
        logging.warning("gradunwarp: no gradient coefficient file (gradcoeffile=%r, searched %s); "
                        "emitting the uncorrected volume",
                        params.get("gradcoeffile"), gradwarp.COEFF_SEARCH_DIRS)
        return volume
    coeffs = gradwarp.read_siemens_grad(path)
    head = context["reference_head"]
    try:
        position = str(context["metadata"].measurementInformation.patientPosition)
    except Exception:
        position = "unknown"
    if position.upper() not in ("HFS", "PATIENTPOSITION.HFS", "UNKNOWN"):
        logging.warning("gradunwarp: patient position %s; the LPS->coil-frame mapping is validated "
                        "for HFS only", position)
    packed, key = mrdrecon.to_acquisition_frame(volume, context["orientation"], context["flip_slice"])
    n = packed.shape
    fov = float(context["output_fov_mm"])
    spacing = [fov / n[0], fov / n[1], fov / n[2]]
    dirs = [np.asarray(head.slice_dir, float), np.asarray(head.phase_dir, float), np.asarray(head.read_dir, float)]
    logging.info("gradunwarp: %s from %s (coil %s), centre %s mm, spacing %.3f mm",
                 mode, path, coil, tuple(round(float(v), 2) for v in head.position), spacing[0])
    corrected = gradwarp.unwarp_volume(
        packed, coeffs, center_lps=np.asarray(head.position, float), axis_dirs_lps=dirs,
        spacing_mm=spacing, mode="2D" if mode.upper().startswith("2D") else "3D",
        jacobian=mode.lower() not in ("3dnojac", "nojac"))
    return mrdrecon.from_acquisition_frame(corrected, key, context["flip_slice"])


mrdrecon.OUTPUT_VOLUME_HOOKS.append(gradient_unwarp)


def process(connection, config, metadata):
    return mrdrecon.run(connection, config, metadata, reconstruct)
