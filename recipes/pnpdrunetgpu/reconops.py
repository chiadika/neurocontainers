#!/usr/bin/env python3
"""Non-Cartesian reconstruction operators shared by this container's apps.

Lifted verbatim from recipes/cgsense3d/cgsense3d.py, which is where they are
documented and where their tests live. Analytic radial density compensation,
ESPIRiT sensitivity maps, the forward/adjoint NUFFT pair, the Toeplitz normal
operator with its startup self-check, and conjugate gradient.
"""

from concurrent.futures import ThreadPoolExecutor
import logging
import os

import h5py
import numpy as np
import sigpy
import sigpy.mri


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
        raise ValueError(
            f"'{DCF_DATASET}' in {source} has {dcf.size} values but the trajectory "
            f"has {int(np.prod(expected_shape))} samples. They must correspond "
            "one-to-one; regenerate it with make_dcf.py."
        )
    logging.info("Density compensation: precomputed '%s' from %s (method=%s)",
                 DCF_DATASET, os.path.basename(source), method)
    return np.ascontiguousarray(dcf.reshape(-1), dtype=np.float32)


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
