#!/usr/bin/env python3
"""CG-SENSE for 3D centre-out radial, built for scanner-time reconstruction.

Solves

    min_x  || W^(1/2) (A x - y) ||^2 + lamda ||x||^2,   (A x)_c = F_nu (S_c x)

by conjugate gradient on the normal equations (A^H W A + lamda I) x = A^H W y,
with ESPIRiT sensitivity maps.

Three things make this fast enough to run online:

  1. Toeplitz normal operator. A^H W A is a convolution, so after one setup pass
     every CG iteration is an FFT pair on a 2N grid instead of two NUFFTs. The
     NUFFT cost here is dominated by interpolating 200k points through a 4^3
     kernel, which is nearly independent of matrix size -- so removing it from
     the inner loop is the single biggest win. Measured on 6434x32 radial
     points at 128^3: ~10 s per NUFFT pair, against ~1.5 s for the FFT pair.

  2. Analytic radial density compensation. This trajectory samples uniformly
     along each spoke (the first ADC sample sits at the end of the gradient
     ramp, so every sample is on the flat top), which makes w = |k|^2 exactly
     right. Pipe-Menon would cost 15 NUFFT pairs to rediscover it.

  3. ESPIRiT at the data's own resolution, then upsampled. Sensitivity maps are
     smooth; calibrating them on a 32^3 grid and zero-pad interpolating is both
     faster and better conditioned than calibrating on a mostly-empty 128^3.

The Toeplitz kernel's scale is calibrated numerically against the direct
operator at startup and the agreement is asserted, so a convention mismatch
fails loudly instead of quietly rescaling the regularisation.

Everything about talking to the scanner -- receiving readouts, aligning them to
the trajectory by scan_counter, choosing the trajectory from the data shape,
splitting multi-echo streams (`echosplit`), coil compression, orientation,
per-slice emission with correct geometry -- lives in mrdrecon.py and is shared
with quickgrid. Since 1.2.0 this app also carries quickgrid's output hook for
gradient-nonlinearity correction (`gradunwarp`, maths in gradwarp.py), bundles
the same trajectory set, and cuts a stored DCF to the trajectory rows an echo
or a short acquisition actually uses.
"""

from concurrent.futures import ThreadPoolExecutor
import logging
import os

import h5py
import numpy as np
import sigpy
import sigpy.mri

import gradwarp

import mrdrecon


RECON_NAME = "cgsense3d"

# Files baked into the image (the same set as quickgrid). `trajectoryfile: auto`
# picks the one whose (samples, readouts) matches the acquired data; a wrong
# pick is refused rather than reconstructed, because phyllotaxis directions
# depend on the spoke count.
BUNDLED_TRAJECTORIES = {
    "radial3d_res128_us2x": f"/opt/{RECON_NAME}/radial3d_res128_us2x_trajectory.h5",
    "radial3d_res128_nyquist": f"/opt/{RECON_NAME}/radial3d_res128_nyquist_trajectory.h5",
    "radial3d_n6434": f"/opt/{RECON_NAME}/radial3d_n6434_trajectory.h5",
    "cones3d_n4846": f"/opt/{RECON_NAME}/cones3d_n4846_trajectory.h5",       # 220 mm R=2, matrix 134, with dcf
    "cones3d_n9598": f"/opt/{RECON_NAME}/cones3d_n9598_trajectory.h5",       # 220 mm R=1, matrix 134, with dcf
    "cones3d_n68785": f"/opt/{RECON_NAME}/cones3d_n68785_trajectory.h5",     # 256^3 / 250 mm / TE 50, matrix 266, with dcf
    "cones3d_n34506": f"/opt/{RECON_NAME}/cones3d_n34506_trajectory.h5",     # same, R=2, matrix 278, with dcf
    "cones3d_n206435": f"/opt/{RECON_NAME}/cones3d_n206435_trajectory.h5",   # 512^3 / 250 mm / TR 2, R=4, matrix 512
    "cones3d_n412489": f"/opt/{RECON_NAME}/cones3d_n412489_trajectory.h5",   # 512^3 / 250 mm / TR 2, R=2, matrix 512
}

# `auto` also searches the scanner's `fire\share` folder (mounted as /tmp/share,
# mrdrecon.TRAJECTORY_SEARCH_DIRS) for *_trajectory.h5, so a trajectory that is
# not baked in only has to be copied there. Entries here just name them in the UI.
SHARE_TRAJECTORIES = {}

# Bundled but deliberately kept out of `auto`: these have the same
# (samples, readouts) as a BUNDLED_TRAJECTORIES entry and differ only in
# readout order, so `auto` could not tell them apart. Select them by name.
NAMED_TRAJECTORIES = {
    # IR-prepared 256^3 cones (..._sp2_ir32_*.seq): same readouts as cones3d_n34506
    # in a seeded random acquisition order; rows permuted, Pipe DCF permuted,
    # `acq_order` dataset records the permutation. matrix 278, fovcm 25.0.
    "cones3d_n34506_ir32": f"/opt/{RECON_NAME}/cones3d_n34506_ir32_trajectory.h5",
}

# Keep these in step with OpenReconLabel.json (which is what the scanner JSON
# config is generated from; it holds at most 14 parameters, so some of these
# are set only from the JSON and have no UI control).
DEFAULTS = {
    "config": RECON_NAME,
    "matrixsize": 128,
    "fovcm": 22.0,
    "trajectoryfile": "auto",
    "trajectorydataset": "k",
    "dcfmode": "auto",          # file -> analytic -> pipe
    "dcfiterations": 15,        # only if Pipe-Menon has to run at recon time
    # Iterative recon should not have its data term tapered; the regulariser
    # does that job, and a k-space window biases the solution.
    "applyfermifilter": False,
    "rejectbadreadouts": False,
    "compresscoils": True,
    "coilvarianceretention": 0.95,
    "maxcoils": 0,
    "cgiterations": 12,
    "cglamda": 1e-3,
    "espiritmatrix": 32,
    "espiritthresh": 0.02,
    "espiritcrop": 0.8,
    "usetoeplitz": True,
    "maxworkers": 8,            # threads for the per-coil Toeplitz convolutions (capped by coils)
    "applyn4biascorrection": False,
    "orientation": "xyz",       # right for the Terra.X Pulseq interpreter (x runs F->H, z runs L->R)
    "orientationflipslice": False,
    "echosplit": 1,             # echoes per TR; >1 reconstructs one series per echo
    "gradunwarp": "3D",         # off | 3D | 3Dnojac | 2D  (gradient-nonlinearity correction)
    "gradcoeffile": "auto",     # Siemens coeff_<coil>.grad: 'auto' searches /opt/cgsense3d and /tmp/share
}

mrdrecon.configure(
    name=RECON_NAME,
    defaults=DEFAULTS,
    trajectories=BUNDLED_TRAJECTORIES,
    image_comment="3D radial CG-SENSE",
    meta_prefix="CgSense3D",
)


# ---------------------------------------------------------------------------
# Density compensation
# ---------------------------------------------------------------------------
DCF_DATASET = "dcf"


def _load_dcf_from_trajectory(recon, expected_shape):
    """Read a precomputed `dcf` dataset from the trajectory file, if present.

    The DCF depends only on the trajectory, never on the data, so the right
    place to compute it is offline (see _seqdev/make_dcf.py) and ship it beside
    the coordinates. Pipe-Menon on this data costs ~73 s per iteration at 3.3M
    points and ~117 s at 6.5M, so computing it per reconstruction is not an
    option at the scanner.
    """
    source = getattr(recon, "trajectory_source", None)
    if not source or not os.path.exists(source):
        return None
    try:
        with h5py.File(source, "r") as handle:
            if DCF_DATASET not in handle:
                return None
            dcf = np.asarray(handle[DCF_DATASET][...], dtype=np.float32)
            method = handle[DCF_DATASET].attrs.get("method", "unknown")
    except Exception as exc:
        logging.warning("Could not read '%s' from %s: %s", DCF_DATASET, source, exc)
        return None

    if dcf.size != int(np.prod(expected_shape)):
        aligned = _align_stored_dcf(dcf, getattr(recon, "trajectory_selection", None), expected_shape)
        if aligned is None:
            raise ValueError(
                f"'{DCF_DATASET}' in {source} has {dcf.size} values but the trajectory "
                f"has {int(np.prod(expected_shape))} samples. They must correspond "
                "one-to-one; regenerate it with make_dcf.py."
            )
        dcf = aligned
    logging.info("Density compensation: precomputed '%s' from %s (method=%s)",
                 DCF_DATASET, os.path.basename(source), method)
    return np.ascontiguousarray(dcf.reshape(-1), dtype=np.float32)


def _align_stored_dcf(dcf, selection, expected_shape):
    """Put a DCF stored for the whole trajectory file through the same row and
    sample selection the trajectory went through in preparation.

    One echo of an alternating-echo stream uses every other trajectory row, and a
    scan that stopped early uses the rows that arrived, so the prepared trajectory
    is a subset of the file's. The stored weights of those rows are the right
    relative weights for them: the subset's sampling density differs from the
    whole file's by a near-constant factor (the echoes interleave uniformly), and
    a constant only rescales the data term against `cglamda`. Returns None when
    the stored array cannot be made to follow the selection.
    """
    if not selection or dcf.ndim != 2:
        return None
    if selection.get("swapped"):
        dcf = dcf.T
    rows = selection.get("rows")
    total_rows = dcf.shape[1]
    if rows is not None:
        rows = np.asarray(rows, dtype=np.int64)
        if rows.size == 0 or int(rows.max()) >= dcf.shape[1]:
            return None
        dcf = dcf[:, rows]
    offset = int(selection.get("sample_offset", 0))
    dcf = dcf[offset:offset + int(selection.get("samples", dcf.shape[0])),
              :int(selection.get("readouts", dcf.shape[1]))]
    if dcf.shape != tuple(int(n) for n in expected_shape):
        return None
    logging.info("Density compensation: stored weights aligned to the %d of %d trajectory rows "
                 "this reconstruction uses", dcf.shape[1], total_rows)
    return np.ascontiguousarray(dcf)


def _looks_uniform_radial(trajectory, tol=0.05):
    """True if every spoke runs monotonically outward with constant spacing.

    That is the condition under which w = |k|^2 is exact. Cones, spirals and
    variable-density radial all fail it.
    """
    radius = np.linalg.norm(trajectory, axis=-1)
    if radius.ndim != 2 or radius.shape[0] < 3:
        return False
    steps = np.diff(radius, axis=0)
    if not np.all(steps > 0):
        return False
    return float(np.std(steps) / max(float(np.mean(steps)), 1e-12)) <= tol


def _analytic_radial_dcf(recon):
    w = np.square(recon.radial_k(), dtype=np.float32).ravel()
    peak = float(w.max())
    if peak > 0:
        w = w / peak
    positive = w[w > 0]
    if positive.size:
        # The k=0 sample of every spoke would otherwise carry zero weight even
        # though it is the highest-SNR point; floor it at the first shell.
        w = np.maximum(w, float(positive.min()))
    return np.ascontiguousarray(w, dtype=np.float32)


def _pipe_menon_dcf(recon, coord, iterations):
    logging.warning(
        "Density compensation: computing Pipe-Menon at reconstruction time over "
        "%d points. This is slow (minutes); precompute it with make_dcf.py and "
        "ship it in the trajectory file instead.", coord.shape[0])
    dcf = sigpy.mri.dcf.pipe_menon_dcf(
        coord, recon.image_shape, max_iter=int(iterations), show_pbar=False)
    dcf = np.asarray(np.real(dcf), dtype=np.float32)
    peak = float(dcf.max())
    return np.ascontiguousarray(dcf / peak if peak > 0 else dcf, dtype=np.float32)


def density_compensation(recon, coord=None):
    """Per-sample density compensation, selected by the `dcfmode` parameter.

        file      read the `dcf` dataset from the trajectory file (free)
        analytic  w = |k|^2  -- exact only for uniformly-sampled radial
        pipe      Pipe-Menon at reconstruction time -- correct for anything, slow
        auto      file if present, else analytic if the trajectory is uniform
                  radial, else pipe
    """
    mode = str(recon.config.get("dcfmode", "auto")).strip().lower()
    shape = recon.trajectory.shape[:2]
    iterations = int(recon.config.get("dcfiterations", 15))

    if mode not in ("auto", "file", "analytic", "pipe"):
        raise ValueError(f"dcfmode must be auto, file, analytic or pipe; got {mode!r}")

    if mode in ("auto", "file"):
        stored = _load_dcf_from_trajectory(recon, shape)
        if stored is not None:
            return stored
        if mode == "file":
            raise ValueError(
                f"dcfmode='file' but no '{DCF_DATASET}' dataset was found in "
                f"{getattr(recon, 'trajectory_source', None)!r}. Generate one with "
                "make_dcf.py, or use dcfmode='auto'.")

    if mode == "analytic" or (mode == "auto" and _looks_uniform_radial(recon.trajectory)):
        logging.info("Density compensation: analytic |k|^2 (%s)",
                     "trajectory is uniform radial" if mode == "auto" else "requested")
        return _analytic_radial_dcf(recon)

    if coord is None:
        coord = recon.coords_index_units()
    if mode == "auto":
        logging.warning(
            "Density compensation: trajectory is not uniformly-sampled radial and no "
            "precomputed '%s' was found, falling back to Pipe-Menon.", DCF_DATASET)
    return _pipe_menon_dcf(recon, coord, iterations)


# ---------------------------------------------------------------------------
# ESPIRiT sensitivity maps
# ---------------------------------------------------------------------------
def espirit_maps(recon, coord, weights):
    """ESPIRiT maps at `espiritmatrix`, zero-pad interpolated to full matrix.

    ESPIRiT wants Cartesian calibration data, so the non-uniform samples are
    gridded to a small Cartesian volume first. Doing this at the trajectory's
    own resolution keeps the calibration matrix well filled -- at 128^3 this
    data occupies only the central quarter of k-space and the outer region
    would contribute nothing but zeros to the calibration.
    """
    matrix = int(recon.matrix_size)
    calib_matrix = min(int(recon.config["espiritmatrix"]), matrix)
    scale = calib_matrix / float(matrix)

    # Only samples inside the calibration volume's own Nyquist radius belong on
    # this grid. Rescaling the full trajectory instead would fold everything
    # beyond |k|*fov = calib_matrix/2 back into the maps, and costs 4x more.
    radius = np.linalg.norm(coord, axis=-1)
    inside = radius <= (calib_matrix / 2.0)
    logging.info("ESPIRiT: gridding calibration volume at %d^3 from %d of %d samples (%.1f%%)",
                 calib_matrix, int(inside.sum()), inside.size, 100.0 * inside.mean())
    if not inside.any():
        raise RuntimeError("No samples fall inside the ESPIRiT calibration radius")

    low_shape = (calib_matrix,) * 3
    low_images = np.asarray(sigpy.nufft_adjoint(
        recon.kspace.reshape(recon.num_coils, -1)[:, inside] * weights[inside],
        coord[inside] * scale,
        (recon.num_coils,) + low_shape,
    ), dtype=np.complex64)

    calib_ksp = sigpy.fft(low_images, axes=(-3, -2, -1))
    logging.info("ESPIRiT: calibrating %d coils", recon.num_coils)
    maps = sigpy.mri.app.EspiritCalib(
        calib_ksp,
        calib_width=min(24, calib_matrix),
        thresh=float(recon.config["espiritthresh"]),
        kernel_width=6,
        crop=float(recon.config["espiritcrop"]),
        show_pbar=False,
    ).run()
    maps = np.asarray(maps, dtype=np.complex64)

    support = float(np.mean(np.abs(maps).sum(0) > 0))
    logging.info("ESPIRiT: support covers %.1f%% of the calibration volume", 100 * support)
    if support < 0.02:
        raise RuntimeError(
            "ESPIRiT produced an almost empty support; lower espiritcrop or "
            "check that the trajectory matches the acquired data"
        )

    if calib_matrix != matrix:
        logging.info("ESPIRiT: interpolating maps %d^3 -> %d^3", calib_matrix, matrix)
        spectrum = sigpy.fft(maps, axes=(-3, -2, -1))
        spectrum = sigpy.resize(spectrum, (recon.num_coils,) + (matrix,) * 3)
        maps = sigpy.ifft(spectrum, axes=(-3, -2, -1))
        # Zero-padding rescales by the volume ratio; renormalise so that the
        # maps stay unit-norm where the coils overlap.
        maps = np.asarray(maps, dtype=np.complex64)
        norm = np.sqrt(np.sum(np.abs(maps) ** 2, axis=0))
        peak = float(np.percentile(norm[norm > 0], 95)) if np.any(norm > 0) else 1.0
        if peak > 0:
            maps = maps / peak

    return np.asarray(maps, dtype=np.complex64)


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------
def forward(x, maps, coord):
    """A x. Batched over coils: sigpy vectorises the interpolation across
    leading axes, which measured 3.4x faster than looping at 3.3M points."""
    return np.asarray(sigpy.nufft(maps * x, coord), dtype=np.complex64)


def adjoint(y, maps, coord, shape):
    """A^H y, batched over coils for the same reason as forward()."""
    images = sigpy.nufft_adjoint(y, coord, (maps.shape[0],) + tuple(shape))
    return np.asarray(np.sum(np.conj(maps) * images, axis=0), dtype=np.complex64)


def _direct_normal(x, maps, coord, weights, shape):
    return adjoint(weights * forward(x, maps, coord), maps, coord, shape)


def build_normal_operator(maps, coord, weights, shape, lamda, use_toeplitz=True, max_workers=1):
    """Return `normal(x) = A^H W A x + lamda x`, Toeplitz-accelerated if possible.

    For a single coil, A^H W A is convolution with the point spread function of
    the weighted sampling pattern, so it can be applied as one FFT pair against
    a precomputed kernel. Two details are easy to get wrong and both are
    load-bearing:

      * The PSF has to be sampled at the *image grid's* voxel spacing over twice
        the extent. sigpy scales coordinates to the output grid, so doubling the
        grid means doubling the coordinates -- otherwise the kernel describes a
        transform at half the spatial frequency.

      * The coil maps do not commute with the transform, so the sum over coils
        cannot be factored into a single |S|^2 weighting. Each coil needs its
        own convolution: A^H W A x = sum_c conj(S_c) T(S_c x).

    The kernel's absolute scale depends on NUFFT normalisation, so rather than
    deriving it we calibrate against one application of the direct operator and
    assert the two agree. A mismatch falls back instead of silently rescaling
    the regularisation.
    """
    def direct(x):
        return _direct_normal(x, maps, coord, weights, shape) + lamda * x

    if not use_toeplitz:
        logging.info("Normal operator: direct (two NUFFTs per coil per iteration)")
        return direct

    big = tuple(2 * s for s in shape)
    axes = (-3, -2, -1)
    logging.info("Toeplitz: building %s point spread function", big)
    psf = sigpy.nufft_adjoint(
        weights.astype(np.complex64), (2.0 * coord).astype(np.float32), big
    )
    kernel = sigpy.fft(np.fft.ifftshift(psf), axes=axes, center=False).astype(np.complex64)
    inner = tuple(slice((b - s) // 2, (b - s) // 2 + s) for b, s in zip(big, shape))

    def convolve(v):
        padded = np.zeros(big, dtype=np.complex64)
        padded[inner] = v
        out = sigpy.ifft(kernel * sigpy.fft(padded, axes=axes, center=False),
                         axes=axes, center=False)
        return np.asarray(out[inner], dtype=np.complex64)

    # Each coil's convolution is independent, and the FFTs release the GIL, so
    # the coil loop threads well. This is the CG inner loop -- 12 iterations
    # times however many virtual coils -- and was the single largest remaining
    # serial cost (measured 23.4 s per iteration for 16 coils at 128^3).
    #
    # The per-coil results are accumulated afterwards in coil order, never
    # inside the workers. Floating-point addition is not associative, so summing
    # in completion order would make the result depend on thread scheduling;
    # accumulating in a fixed order keeps it bit-for-bit identical to the serial
    # path, which the container test asserts.
    n_coils = int(maps.shape[0])
    workers = max(1, min(int(max_workers), n_coils))

    def _coil_term(c):
        return np.conj(maps[c]) * convolve(maps[c] * x_current[0])

    x_current = [None]
    pool = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
    if pool is not None:
        logging.info("Toeplitz: %d coils across %d worker threads", n_coils, workers)

    def unscaled(x):
        out = np.zeros(shape, dtype=np.complex64)
        if pool is None:
            for c in range(n_coils):
                out += np.conj(maps[c]) * convolve(maps[c] * x)
            return out
        x_current[0] = x
        terms = list(pool.map(_coil_term, range(n_coils)))
        for term in terms:            # fixed coil order, not completion order
            out += term
        return out

    rng = np.random.default_rng(0)
    probe = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)
    # The scale relates the convolution kernel to the direct operator and does
    # not depend on the coils, so calibrate with one coil rather than all of
    # them -- the dominant setup cost at these point counts.
    probe_maps = maps[:1]
    reference = _direct_normal(probe, probe_maps, coord, weights, shape)
    approx = np.conj(probe_maps[0]) * convolve(probe_maps[0] * probe)
    denom = float(np.vdot(approx, approx).real)
    scale = float(np.vdot(approx, reference).real / denom) if denom > 0 else 0.0
    residual = np.linalg.norm(reference - scale * approx) / max(np.linalg.norm(reference), 1e-30)
    logging.info("Toeplitz: scale %.6g, relative disagreement %.3e", scale, residual)

    if not np.isfinite(scale) or residual > 5e-2:
        logging.warning(
            "Toeplitz kernel disagrees with the direct operator (%.3e); "
            "falling back to the direct normal operator", residual,
        )
        return direct

    def normal(x):
        return scale * unscaled(x) + lamda * x

    return normal


def conjugate_gradient(normal, b, max_iter, tol=1e-5):
    x = np.zeros_like(b)
    r = b.copy()
    p = r.copy()
    rr = float(np.vdot(r, r).real)
    rr0 = rr
    if rr0 <= 0:
        return x

    for iteration in range(max_iter):
        Ap = normal(p)
        pAp = float(np.vdot(p, Ap).real)
        if pAp <= 0:
            logging.warning("CG: non-positive curvature at iteration %d; stopping", iteration + 1)
            break
        alpha = rr / pAp
        x += alpha * p
        r -= alpha * Ap
        rr_next = float(np.vdot(r, r).real)
        logging.info("CG %2d/%d  residual %.4e", iteration + 1, max_iter, np.sqrt(rr_next / rr0))
        if rr_next <= tol * tol * rr0:
            logging.info("CG: converged at iteration %d", iteration + 1)
            break
        p = r + (rr_next / rr) * p
        rr = rr_next

    return x


# ---------------------------------------------------------------------------
def reconstruct(recon):
    coord = recon.coords_index_units()
    shape = recon.image_shape
    weights = density_compensation(recon)

    extent = float(np.abs(coord).max())
    logging.info(
        "CG-SENSE: %d coils, %d k-space points, matrix %d, |k|max %.2f of %d",
        recon.num_coils, coord.shape[0], recon.matrix_size, extent, recon.matrix_size // 2,
    )
    if extent < 0.45 * recon.matrix_size:
        logging.warning(
            "Trajectory fills only %.0f%% of the requested matrix. The data supports "
            "about %d^3; reconstructing at %d^3 interpolates rather than resolves.",
            100 * extent / (recon.matrix_size / 2), int(round(2 * extent)), recon.matrix_size,
        )
    if extent > 0.5 * recon.matrix_size + 1e-3:
        logging.warning("Trajectory exceeds the matrix (|k|max %.2f > %d); samples outside "
                        "will be dropped. Raise matrixsize to the sequence's label extent.",
                        extent, recon.matrix_size // 2)

    maps = espirit_maps(recon, coord, weights)
    lamda = float(recon.config["cglamda"])

    y = recon.kspace.reshape(recon.num_coils, -1)
    b = adjoint(weights * y, maps, coord, shape)

    normal = build_normal_operator(
        maps, coord, weights, shape, lamda,
        use_toeplitz=bool(recon.config["usetoeplitz"]),
        max_workers=recon.max_workers,
    )
    x = conjugate_gradient(normal, b, max_iter=int(recon.config["cgiterations"]))

    recon.save_debug("espirit_maps", maps)
    return np.abs(x).astype(np.float32)


def gradient_unwarp(volume, context):
    """Output hook: correct gradient nonlinearity in the acquisition frame.

    Skips with a warning (never fails the reconstruction) when switched off or
    when no coefficient file can be found. The file is Siemens-proprietary and
    is not bundled; copy coeff_<coil>.grad into fire\\share (/tmp/share).
    Identical to quickgrid's hook; the maths lives in gradwarp.py.
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
