#!/usr/bin/env python3
"""Worked example: CG-SENSE, as a drop-in replacement for the template's
``reconstruct``.

Copy the pieces you want into ``openrecontemplate.py``. Nothing imports this
file at runtime -- it is here to be read.

The problem
-----------
The adjoint reconstruction in the template computes ``x = A^H W y`` in one pass.
That is not the image; it is the image convolved with the point spread function
of the sampling pattern. CG-SENSE instead *solves*

    min_x  || W^(1/2) (A x - y) ||^2  +  lamda ||x||^2

where ``A`` stacks, over coils, "multiply by the coil sensitivity, then
non-uniform Fourier transform":  ``(A x)_c = F_nu (S_c x)``.

Setting the gradient to zero gives the normal equations

    (A^H W A + lamda I) x = A^H W y

whose system matrix is Hermitian positive definite, which is exactly what
conjugate gradient needs. Note the right-hand side is the template's adjoint
reconstruction -- CG starts from what you already had and iterates from there.

Three things to get right
-------------------------
1. ``A`` and ``A^H`` must be an exact adjoint pair. If they are not, CG
   converges to the wrong fixed point, usually while looking plausible for the
   first few iterations. Test it -- see :func:`adjoint_test`.
2. The density compensation ``W`` is a *weighting*, not the reconstruction.
   In the adjoint recon the DCF is what makes the image; here it only
   conditions the system so CG converges in tens rather than hundreds of
   iterations. Pruessmann 2001 introduces it as a preconditioner.
3. Sodium SNR is low, so the iteration count is itself a regulariser. Both
   ``lamda`` and ``max_iter`` control how much noise you let in; tune them
   together, and stop early rather than "converging".
"""

import logging

import numpy as np
import scipy.ndimage as ndi
import sigpy
import sigpy.mri

import mrdrecon


# Two knobs this example reads. They are only honoured if you add them to
# DEFAULTS in openrecontemplate.py -- ReconInput.config is built from
# OPENRECON_DEFAULTS, so a key absent from there always falls back:
#
#     DEFAULTS = {..., "cgiterations": 15, "cglamda": 1e-3}
#
# Add matching entries to OpenReconLabel.json to expose them in the scanner UI.


# ---------------------------------------------------------------------------
# 1. Sensitivity maps
# ---------------------------------------------------------------------------
def sensitivity_maps(recon, coord, weights, mask_percentile=40.0):
    """Smoothed coil maps with a support mask.

    ``mrdrecon.estimate_sensitivities`` is the estimator ``sodiumgridding``
    uses: blur each complex coil image and normalise by the voxel-wise
    root-sum-of-squares, so that ``sum_c |S_c|^2 == 1`` everywhere.

    That is fine for a matched-filter *combine*, but not for SENSE unfolding:
    because the normalisation has an epsilon floor rather than a mask, voxels
    outside the object get noise normalised up to unit norm, and CG will happily
    fit that noise. The mask below is the minimum fix. For a stronger estimate
    use ``sigpy.mri.app.EspiritCalib`` on a gridded low-resolution Cartesian
    k-space, or ``sigpy.mri.app.JsenseRecon`` directly on the non-uniform data.
    """
    coil_images = np.zeros((recon.num_coils,) + recon.image_shape, dtype=np.complex64)
    for coil_index in range(recon.num_coils):
        coil_images[coil_index] = sigpy.nufft_adjoint(
            recon.kspace[coil_index].ravel() * weights, coord, recon.image_shape
        )

    maps = mrdrecon.estimate_sensitivities(coil_images)

    rss = np.sqrt(np.sum(np.abs(coil_images) ** 2, axis=0))
    threshold = float(np.percentile(rss, mask_percentile))
    support = ndi.binary_closing(rss > threshold, np.ones((3, 3, 3)))
    logging.info(
        "Sensitivity support: %.1f%% of voxels above the %.0fth percentile",
        100.0 * support.mean(),
        mask_percentile,
    )
    return (maps * support[None, ...]).astype(np.complex64), coil_images


# ---------------------------------------------------------------------------
# 2a. The short route: sigpy already implements this
# ---------------------------------------------------------------------------
def reconstruct_sigpy(recon):
    """CG-SENSE in six lines.

    ``SenseRecon`` subclasses ``LinearLeastSquares``, which selects
    ``ConjugateGradient`` whenever no proximal operator is given -- so this is
    genuinely CG on the normal equations above, not a different solver.
    """
    coord = recon.coords_index_units()
    weights = sigpy.mri.dcf.pipe_menon_dcf(coord, recon.image_shape, max_iter=15, show_pbar=False)
    weights = (weights / weights.max()).astype(np.float32)

    maps, _ = sensitivity_maps(recon, coord, weights)

    image = sigpy.mri.app.SenseRecon(
        recon.kspace.reshape(recon.num_coils, -1),
        maps,
        lamda=float(recon.config.get("cglamda", 1e-3)),
        weights=weights,
        coord=coord,
        max_iter=int(recon.config.get("cgiterations", 15)),
        show_pbar=False,
    ).run()

    return np.abs(np.asarray(image)).astype(np.float32)


# ---------------------------------------------------------------------------
# 2b. The explicit route: the same thing, written out
# ---------------------------------------------------------------------------
def forward(x, maps, coord):
    """A x -- image to multi-coil non-uniform k-space, shape (coils, npts)."""
    return np.stack([sigpy.nufft(maps[c] * x, coord) for c in range(maps.shape[0])])


def adjoint(y, maps, coord, shape):
    """A^H y -- multi-coil non-uniform k-space back to a single image."""
    out = np.zeros(shape, dtype=np.complex64)
    for c in range(maps.shape[0]):
        out += np.conj(maps[c]) * sigpy.nufft_adjoint(y[c], coord, shape)
    return out


def adjoint_test(maps, coord, shape, seed=0):
    """<A x, y> must equal <x, A^H y>. Run this once; trust CG only after it passes."""
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)
    y = (rng.standard_normal((maps.shape[0], coord.shape[0]))
         + 1j * rng.standard_normal((maps.shape[0], coord.shape[0]))).astype(np.complex64)

    lhs = np.vdot(forward(x, maps, coord), y)
    rhs = np.vdot(x, adjoint(y, maps, coord, shape))
    error = abs(lhs - rhs) / max(abs(lhs), 1e-12)
    logging.info("Adjoint test: relative error %.3e", error)
    return error


def conjugate_gradient(normal, b, max_iter=15, tol=1e-6):
    """Solve ``normal(x) = b`` for Hermitian positive definite ``normal``.

    Textbook CG. Every iteration costs one application of the normal operator,
    which here is two NUFFTs per coil -- the reason iteration count matters.
    """
    x = np.zeros_like(b)
    r = b.copy()
    p = r.copy()
    rr = float(np.vdot(r, r).real)
    rr0 = rr

    for iteration in range(max_iter):
        Ap = normal(p)
        alpha = rr / float(np.vdot(p, Ap).real)
        x += alpha * p
        r -= alpha * Ap
        rr_next = float(np.vdot(r, r).real)
        logging.info(
            "CG %2d/%d  residual %.4e", iteration + 1, max_iter, np.sqrt(rr_next / rr0)
        )
        if rr_next <= tol * tol * rr0:
            break
        p = r + (rr_next / rr) * p
        rr = rr_next

    return x


def reconstruct_explicit(recon):
    """CG-SENSE with the operators and the solver written out in full."""
    coord = recon.coords_index_units()
    shape = recon.image_shape

    weights = sigpy.mri.dcf.pipe_menon_dcf(coord, shape, max_iter=15, show_pbar=False)
    weights = (weights / weights.max()).astype(np.float32)

    maps, _ = sensitivity_maps(recon, coord, weights)
    adjoint_test(maps, coord, shape)

    lamda = float(recon.config.get("cglamda", 1e-3))
    y = recon.kspace.reshape(recon.num_coils, -1)

    def normal(x):
        return adjoint(weights * forward(x, maps, coord), maps, coord, shape) + lamda * x

    b = adjoint(weights * y, maps, coord, shape)   # == the adjoint reconstruction
    x = conjugate_gradient(normal, b, max_iter=int(recon.config.get("cgiterations", 15)))

    recon.save_debug("cgsense_maps", maps)
    return np.abs(x).astype(np.float32)


# Pick one and assign it to `reconstruct` in openrecontemplate.py.
reconstruct = reconstruct_sigpy


# ---------------------------------------------------------------------------
# Making it fast enough for the scanner
# ---------------------------------------------------------------------------
# The normal operator above costs 2 NUFFTs per coil per iteration. At N=128
# with 10 virtual coils and 15 iterations that is 300 gridding passes against
# the 10 the adjoint recon needs -- roughly 20-30x the runtime. Three levers,
# in the order worth pulling:
#
#   1. Coil compression. Already on by default ("compresscoils"), and it is what
#      turns 32 physical channels into ~8 virtual ones. Non-negotiable here.
#
#   2. Toeplitz embedding. A^H W A is a convolution, so it can be applied as one
#      FFT pair against a precomputed kernel instead of two NUFFTs:
#
#          psf = sigpy.fourier.toeplitz_psf(coord, shape)
#
#      sigpy.linop.NUFFT(shape, coord, toeplitz=True) builds the operator whose
#      _normal_linop uses it. Note that sigpy.mri.linop.Sense in 0.1.27 does not
#      expose a toeplitz argument, so assemble the SENSE operator yourself if
#      you want this.
#
#   3. Drop matrixsize. Sodium resolution is limited by SNR long before it is
#      limited by the matrix; 64^3 costs an eighth of 128^3.
#
# Whatever you land on, check it against min_required_memory and
# min_count_required_cpu_cores in OpenReconLabel.json -- the scanner enforces
# them, and a recon that needs more than it declared will be refused.
