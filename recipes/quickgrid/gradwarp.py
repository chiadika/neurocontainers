#!/usr/bin/env python3
"""gradwarp: gradient-nonlinearity ("distortion") correction for reconstructed volumes.

Siemens' ICE applies its DistorCor functor to images it reconstructs itself;
images injected by FIRE under IceProgramStandard enter the chain after that
functor and never get corrected (journal 4.9). This module does the same
correction inside the reconstruction, from the scanner's own gradient-coil
coefficient file (`coeff_<coil>.grad`, e.g. coeff_IMPULSE.grad).

The maths is a port of the Siemens path of gradunwarp
(Washington-University/gradunwarp, MIT licence, Human Connectome Project):
the displacement at a point is the spherical-harmonic expansion of the coil's
field deviation, evaluated with Siemens' Legendre normalisation, times R0.
An unwarped voxel at true position p takes the acquired value at p + dv(p),
optionally multiplied by the Jacobian determinant |I + grad dv| for intensity
(capped at 10, as gradunwarp does).

Frames: the reconstruction's geometry is in the MRD patient frame (DICOM LPS:
+x left, +y posterior, +z head). gradunwarp evaluates the harmonics in the
frame it calls LAI, which is LPS with y and z negated; that is the convention
its Siemens users have validated against product images, so it is used here
unchanged (assumes a head-first-supine patient position; others are logged).

The coefficient file is Siemens-proprietary: it is never bundled in the image.
It is read from the share folder (`fire\\share`, mounted as /tmp/share) or an
explicit path.
"""

from __future__ import annotations

import glob
import logging
import math
import os
import re
from dataclasses import dataclass

import numpy as np
import scipy.ndimage
import scipy.special

SHARE_DIR = "/tmp/share"
COEFF_SEARCH_DIRS = ["/opt/quickgrid", SHARE_DIR]
COEFF_GLOB = "coeff_*.grad"
MAX_JACOBIAN = 10.0        # gradunwarp's siemens_max_det
_GRAD_LINE = re.compile(
    r"(?P<no>\d+)\s+(?P<aorb>[AB])\s*\(\s*(?P<n>\d+),\s*(?P<m>\d+)\)\s+"
    r"(?P<value>[-+]?\d+\.\d+(?:[eE][-+]?\d+)?)\s+(?P<axis>[xyz])"
)
_R0_LINE = re.compile(r"(?P<R0>\d+\.\d+)\s*m\s*=\s*R0")


@dataclass
class Coefficients:
    """Spherical-harmonic coefficients of one gradient coil."""
    alpha: dict          # axis -> (nmax+1, nmax+1) array, cos terms
    beta: dict           # axis -> (nmax+1, nmax+1) array, sin terms
    R0_m: float
    source: str = ""

    @property
    def order(self) -> int:
        return int(self.alpha["x"].shape[0]) - 1


def read_siemens_grad(path: str) -> Coefficients:
    """Parse a Siemens `coeff_*.grad` file (same regex as gradunwarp)."""
    size = 100
    alpha = {ax: np.zeros((size, size)) for ax in "xyz"}
    beta = {ax: np.zeros((size, size)) for ax in "xyz"}
    R0_m = None
    nmax = 0
    with open(path, "r") as fh:
        for line in fh:
            match = _GRAD_LINE.search(line)
            if match:
                n, m = int(match["n"]), int(match["m"])
                (alpha if match["aorb"] == "A" else beta)[match["axis"]][n, m] = float(match["value"])
                nmax = max(nmax, n, m)
                continue
            match = _R0_LINE.search(line)
            if match:
                R0_m = float(match["R0"])
    if R0_m is None:
        raise ValueError(f"{path}: no 'R0' line found; not a Siemens .grad coefficient file")
    if nmax == 0:
        raise ValueError(f"{path}: no A(n,m)/B(n,m) coefficient lines found")
    k = nmax + 1
    return Coefficients({ax: alpha[ax][:k, :k].copy() for ax in "xyz"},
                        {ax: beta[ax][:k, :k].copy() for ax in "xyz"},
                        R0_m, source=str(path))


def find_coefficient_file(setting: str, coil_name: str | None = None) -> str | None:
    """Resolve the `gradcoeffile` parameter: a path, or 'auto' to search the share dirs."""
    setting = str(setting or "auto").strip()
    if setting.lower() not in ("", "auto"):
        return setting
    found = []
    for directory in COEFF_SEARCH_DIRS:
        found.extend(sorted(glob.glob(os.path.join(directory, COEFF_GLOB))))
    if not found:
        return None
    if coil_name:
        preferred = [f for f in found if coil_name.lower() in os.path.basename(f).lower()]
        if preferred:
            return preferred[0]
    if len(found) > 1:
        logging.warning("Several coefficient files found (%s); using %s", found, found[0])
    return found[0]


def _siemens_B(alpha, beta, r, cos_theta, phi, R0):
    """Field deviation (dimensionless, times R0 gives mm) at spherical coords."""
    nmax = alpha.shape[0] - 1
    b = np.zeros(r.shape, dtype=np.float64)
    for n in range(nmax + 1):
        f = np.power(r / R0, n)
        for m in range(n + 1):
            if alpha[n, m] == 0.0 and beta[n, m] == 0.0:
                continue
            f2 = alpha[n, m] * np.cos(m * phi) + beta[n, m] * np.sin(m * phi)
            p = scipy.special.lpmv(m, n, cos_theta)
            if m > 0:
                p = p * (math.pow(-1, m) * math.sqrt((2 * n + 1) * math.factorial(n - m)
                                                     / (2.0 * math.factorial(n + m))))
            b += f * p * f2
    return b


def displacement_lai(coeffs: Coefficients, x, y, z):
    """Displacement (dx, dy, dz) in mm at LAI-frame points (x, y, z) in mm."""
    x = np.asarray(x, dtype=np.float64) + 1e-4       # gradunwarp's r=0 guard
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    R0 = coeffs.R0_m * 1000.0
    r = np.sqrt(x * x + y * y + z * z)
    cos_theta = z / r
    phi = np.arctan2(y / r, x / r)
    return tuple(R0 * _siemens_B(coeffs.alpha[ax], coeffs.beta[ax], r, cos_theta, phi, R0)
                 for ax in "xyz")


def displacement_lps(coeffs: Coefficients, positions_lps):
    """Displacement in the MRD/DICOM LPS frame, shape (..., 3), mm in and out."""
    p = np.asarray(positions_lps, dtype=np.float64)
    dx, dy, dz = displacement_lai(coeffs, p[..., 0], -p[..., 1], -p[..., 2])
    return np.stack([dx, -dy, -dz], axis=-1)


def _jacobian(dv, spacing_mm):
    """|det(I + grad dv)| on a grid; dv (3, ns, nr, nc) in mm, spacing per axis in mm."""
    g = [np.gradient(dv[i], *spacing_mm) for i in range(3)]   # g[i][k] = d dv_i / d axis_k
    j = np.empty(dv.shape[1:], dtype=np.float64)
    a = [[g[i][k] + (1.0 if i == k else 0.0) for k in range(3)] for i in range(3)]
    j[...] = (a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
              - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
              + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))
    np.abs(j, out=j)
    np.minimum(j, MAX_JACOBIAN, out=j)
    return j


def unwarp_volume(volume, coeffs: Coefficients, center_lps, axis_dirs_lps, spacing_mm,
                  mode="3D", jacobian=True, order=1, coarse_step_mm=8.0, slab=16):
    """Correct a (slices, rows, columns) volume for gradient nonlinearity.

    center_lps: patient-frame centre of the volume (mm); axis_dirs_lps: unit
    directions of axes 0, 1, 2 (slice, phase, read); spacing_mm: voxel size
    per axis. mode '3D' corrects all components, '2D' only the in-plane ones
    (rows/columns), like the product DistorCor2D. The field is evaluated on a
    coarse grid (~coarse_step_mm) and linearly interpolated, and the volume is
    resampled `slab` slices at a time so peak memory stays a few copies of one
    slab rather than of the volume.
    """
    vol = np.asarray(volume, dtype=np.float32)
    if vol.ndim != 3:
        raise ValueError(f"expected a 3D volume, got {vol.shape}")
    shape = np.array(vol.shape)
    dirs = np.asarray(axis_dirs_lps, dtype=np.float64)
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    spacing = np.asarray(spacing_mm, dtype=np.float64)
    center = np.asarray(center_lps, dtype=np.float64)
    mode = str(mode).upper()
    if mode not in ("3D", "2D"):
        raise ValueError("mode must be '3D' or '2D'")

    # coarse index grid, endpoints included
    steps = np.maximum(1, np.round(coarse_step_mm / spacing).astype(int))
    coarse_axes = [np.unique(np.concatenate([np.arange(0, n, s), [n - 1]])).astype(float)
                   for n, s in zip(shape, steps)]
    cs, cr, cc = np.meshgrid(*coarse_axes, indexing="ij")
    offsets = [(cs - (shape[0] - 1) / 2.0) * spacing[0],
               (cr - (shape[1] - 1) / 2.0) * spacing[1],
               (cc - (shape[2] - 1) / 2.0) * spacing[2]]
    pos = center + sum(offsets[k][..., None] * dirs[k] for k in range(3))   # (..., 3) LPS mm
    dv_lps = displacement_lps(coeffs, pos)                                   # (..., 3)
    # displacement expressed along the volume axes (mm per axis)
    dv_axes = np.stack([dv_lps @ dirs[k] for k in range(3)], axis=0)         # (3, ...)
    if mode == "2D":
        dv_axes[0] = 0.0
    # coarse-grid Jacobian (grid spacing in mm per axis)
    coarse_spacing = [float(np.mean(np.diff(ax))) * sp for ax, sp in zip(coarse_axes, spacing)]
    jac = _jacobian(dv_axes, coarse_spacing) if jacobian else None
    logging.info("gradunwarp: order %d, R0 %.3f m, mode %s, |dv| max %.2f mm over the volume, "
                 "Jacobian %s (coarse grid %s)", coeffs.order, coeffs.R0_m, mode,
                 float(np.linalg.norm(dv_axes, axis=0).max()),
                 "on" if jacobian else "off", tuple(len(a) for a in coarse_axes))

    # map full-resolution indices onto the coarse grid (linear in index space)
    scale = [(len(ax) - 1) / max(float(n - 1), 1.0) for ax, n in zip(coarse_axes, shape)]
    out = np.empty(vol.shape, dtype=np.float32)
    rr, cc_ = np.meshgrid(np.arange(shape[1], dtype=np.float32),
                          np.arange(shape[2], dtype=np.float32), indexing="ij")
    for s0 in range(0, int(shape[0]), slab):
        s1 = min(s0 + slab, int(shape[0]))
        ss = np.arange(s0, s1, dtype=np.float32)[:, None, None] * np.ones_like(rr)[None]
        idx = np.stack([ss, np.broadcast_to(rr, ss.shape), np.broadcast_to(cc_, ss.shape)])
        cidx = np.stack([idx[k] * scale[k] for k in range(3)])
        d = np.stack([scipy.ndimage.map_coordinates(dv_axes[k], cidx, order=1, mode="nearest")
                      for k in range(3)])
        sample = idx + d / spacing[:, None, None, None]         # p + dv, in index units
        out[s0:s1] = scipy.ndimage.map_coordinates(vol, sample, order=order, mode="constant", cval=0.0)
        if jacobian:
            out[s0:s1] *= scipy.ndimage.map_coordinates(jac, cidx, order=1, mode="nearest")
    return out
