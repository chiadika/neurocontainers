#!/usr/bin/env python3
"""OpenRecon reconstruction template -- this is the only file you edit.

Everything that is not the algorithm (MRD streaming, trajectory loading,
k-space assembly, coil preparation, patient-space geometry, DICOM display
scaling) lives in ``mrdrecon.py``. You write :func:`reconstruct`.

The shipped implementation is a density-compensated adjoint NUFFT -- the same
thing ``sodiumgridding`` and ``sodiumnufft`` do -- so the container reconstructs
out of the box and you have a working baseline to diff against. Replace the body
of :func:`reconstruct` with your own algorithm; see ``example_cgsense.py`` for a
worked iterative example.

Iterate without a scanner:

    python /opt/code/python-ismrmrd-server/run_local.py raw.h5 -o recon.nii.gz
"""

import logging

import numpy as np
import sigpy

import mrdrecon


# ---------------------------------------------------------------------------
# App identity and the parameters the OpenRecon UI exposes.
#
# Every key here becomes readable in reconstruct() as recon.config["key"], and
# every key you want the scanner operator to set must ALSO appear in
# OpenReconLabel.json with a matching id. Keys defined only here are still
# usable -- they just fall back to the default below.
#
# The framework already defines the shared keys (matrixsize, fovcm, the
# trajectory keys, the coil-preparation keys, orientation, ...); listing one
# here overrides its default.
# ---------------------------------------------------------------------------
RECON_NAME = "openrecontemplate"

BUNDLED_TRAJECTORIES = {
    "23Na_n28": f"/opt/{RECON_NAME}/23Na_n28_trajectory.h5",
    "23Na_n50": f"/opt/{RECON_NAME}/23Na_n50_trajectory.h5",
}

DEFAULTS = {
    "config": RECON_NAME,
    "matrixsize": 128,
    "fovcm": 22.0,
    "trajectorypreset": "23Na_n28",
    "dcfiterations": 5,
    "coilcombinemode": "AC",
    "applyn4biascorrection": False,
}

mrdrecon.configure(
    name=RECON_NAME,
    defaults=DEFAULTS,
    trajectories=BUNDLED_TRAJECTORIES,
    image_comment="OpenRecon template reconstruction",
)


# ---------------------------------------------------------------------------
# Density compensation.
#
# Non-uniform sampling means the centre of k-space is visited far more often
# than the edge. Without a per-sample weight the adjoint transform is dominated
# by the centre and the image is a blur. Two ways to get those weights:
#
#   Pipe-Menon (iterative): grid a vector of ones, sample the resulting density
#   back at the same coordinates, divide, repeat. Self-consistent with whatever
#   interpolation kernel you actually grid with, and makes no assumption about
#   the trajectory. This is what sodiumgridding uses.
#
#   Analytic: for a radial trajectory the density falls as 1/|k|^2, so the
#   weight is |k|^2. Cheap and exact for pure radial -- but a TPI trajectory is
#   density-adapted (constant density past the twist point), so the analytic
#   weight has to be clipped there or it over-weights the outer samples. This is
#   what sodiumnufft does by default.
# ---------------------------------------------------------------------------
def density_compensation(recon):
    iterations = int(recon.config["dcfiterations"])
    coord = recon.coords_index_units()

    if iterations <= 0:
        logging.info("Using analytic radial |k|^2 density compensation")
        return np.square(recon.radial_k(), dtype=np.float32).ravel()

    logging.info("Estimating Pipe-Menon density compensation, %d iterations", iterations)
    dcf = np.ones(coord.shape[0], dtype=np.complex64)
    for iteration in range(iterations):
        psf = sigpy.nufft_adjoint(dcf, coord, recon.image_shape)
        dcf = dcf / (np.abs(sigpy.nufft(psf, coord)) + 1e-8)
        logging.info("  DCF iteration %d/%d", iteration + 1, iterations)

    dcf = np.real(dcf).astype(np.float32)
    median = float(np.median(dcf))
    if median > 0.0 and np.isfinite(median):
        dcf /= median
    return dcf


# ===========================================================================
#
#   YOUR ALGORITHM GOES HERE
#
#   In:  recon.kspace       (coils, samples, readouts) complex64, prepared
#        recon.trajectory   (samples, readouts, 3) float32, native units
#        recon.raw_kspace   (coils, readouts, samples) complex64, untouched
#        recon.image_shape  (N, N, N)
#        recon.config       every resolved OpenRecon parameter
#
#        recon.coords_index_units()  flat (samples*readouts, 3) on [-N/2, N/2]
#        recon.coords_normalized()   flat (samples*readouts, 3) on [-0.5, 0.5]
#        recon.radial_k()            |k| per sample, (samples, readouts)
#
#   Out: a real float32 volume of shape recon.image_shape, indexed [z, y, x] in
#        the acquisition frame. The framework orients it into the DICOM display
#        frame and scales it for the scanner; you do not handle geometry.
#
#   Reusable pieces on mrdrecon: estimate_sensitivities, combine_coils,
#   compress_coils_by_variance, n4_bias_field_correct, build_fermi_filter.
#
# ===========================================================================
def reconstruct(recon):
    """Density-compensated adjoint NUFFT: x = A^H W y, one pass, no iteration."""
    coord = recon.coords_index_units()
    weights = density_compensation(recon)

    logging.info(
        "Adjoint NUFFT: %d coils, %d k-space points -> %s",
        recon.num_coils,
        coord.shape[0],
        recon.image_shape,
    )

    coil_images = np.zeros((recon.num_coils,) + recon.image_shape, dtype=np.complex64)
    for coil_index in range(recon.num_coils):
        coil_images[coil_index] = sigpy.nufft_adjoint(
            recon.kspace[coil_index].ravel() * weights,
            coord,
            recon.image_shape,
        )
        logging.info("Finished coil %d/%d", coil_index + 1, recon.num_coils)

    recon.save_debug("coil_images", coil_images)

    # "AC" is an adaptive (Roemer) combine through smoothed sensitivity maps,
    # "SoS" is root-sum-of-squares. AC keeps phase coherence and avoids the
    # noise-rectification bias SoS has in low-SNR voxels.
    return mrdrecon.combine_coils(
        coil_images, mode=recon.config.get("coilcombinemode", "AC")
    )


# ---------------------------------------------------------------------------
# MRD server entry point. The python-ismrmrd-server imports the module named by
# the incoming ``config`` value and calls its ``process``; leave this alone.
# ---------------------------------------------------------------------------
def process(connection, config, metadata):
    return mrdrecon.run(connection, config, metadata, reconstruct)
