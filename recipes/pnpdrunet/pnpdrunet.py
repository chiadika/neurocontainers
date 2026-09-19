#!/usr/bin/env python3
"""Plug-and-play 3D radial reconstruction with a DRUNet denoiser prior.

Learned regularisation that actually fits a scanner time budget. Where CG-SENSE
solves

    min_x  ||W^(1/2)(Ax - y)||^2 + lamda||x||^2

with a hand-picked quadratic penalty, this replaces that penalty with a trained
denoiser via half-quadratic splitting (Zhang et al., DPIR, TPAMI 2021):

    z_k = argmin_z ||W^(1/2)(Az - y)||^2 + mu_k ||z - x_k||^2     (data)
    x_k = D_sigma_k(z_k)                                          (prior)

The data step is the same Toeplitz-accelerated normal operator CG-SENSE uses,
warm-started from the previous iterate, so a handful of inner CG steps suffice.
The prior step is one DRUNet pass. sigma decreases geometrically, which is what
makes HQS behave: early iterations denoise hard and pull the iterate onto the
image manifold, later ones barely touch it and let data consistency finish.

Why this and not diffusion posterior sampling: DPS needs ~1000 network
evaluations each wrapped in a forward+adjoint pair. This needs ~6. On this
trajectory that is the difference between hours and minutes.

Three practical choices worth knowing about:

  * DRUNet is a 2D grayscale denoiser, so it is applied slice-wise. Set
    `denoiseaxes` to 3 to denoise along all three orthogonal orientations and
    average, which removes the through-plane anisotropy at 3x the prior cost.

  * The iterate is complex. Real and imaginary parts are denoised as separate
    images in one batch, rather than denoising the magnitude, so the prior
    stays a genuine plug-in for the quadratic penalty.

  * DRUNet was trained on natural grayscale images in [0, 1]. The volume is
    rescaled into that range before each prior step and back afterwards, so
    `sigma` is expressed on the same normalised scale.
"""

import logging
import os

import numpy as np
import torch

import mrdrecon
import reconops
import sigpy
from drunet import load_drunet


RECON_NAME = "pnpdrunet"
CHECKPOINT = f"/opt/{RECON_NAME}/drunet_gray.pth"

BUNDLED_TRAJECTORIES = {
    "radial3d_res128_us2x": f"/opt/{RECON_NAME}/radial3d_res128_us2x_trajectory.h5",
    "radial3d_res128_nyquist": f"/opt/{RECON_NAME}/radial3d_res128_nyquist_trajectory.h5",
    "radial3d_n6434": f"/opt/{RECON_NAME}/radial3d_n6434_trajectory.h5",
}

DEFAULTS = {
    "config": RECON_NAME,
    "matrixsize": 128,
    "fovcm": 22.0,
    "trajectoryfile": "auto",
    "trajectorydataset": "k",
    "applyfermifilter": False,
    "rejectbadreadouts": False,
    "compresscoils": True,
    "coilvarianceretention": 0.95,
    "maxcoils": 0,
    "pnpiterations": 6,
    "pnpinnercg": 4,
    "pnpsigmamax": 0.10,
    "pnpsigmamin": 0.02,
    "pnplambda": 0.23,
    "denoiseaxes": 1,
    "usegpu": "auto",
    "denoisebatch": 32,
    "dcfmode": "auto",
    "dcfiterations": 15,
    "espiritmatrix": 32,
    "espiritthresh": 0.02,
    "espiritcrop": 0.8,
    "usetoeplitz": True,
    "applyn4biascorrection": False,
    "orientation": "zyx",
}

mrdrecon.configure(
    name=RECON_NAME,
    defaults=DEFAULTS,
    trajectories=BUNDLED_TRAJECTORIES,
    image_comment="3D radial plug-and-play DRUNet",
    meta_prefix="PnpDrunet",
)

_MODEL = None
_DEVICE = None


def resolve_device(preference="auto"):
    """Pick CUDA if it genuinely works, otherwise CPU.

    `torch.cuda.is_available()` is not sufficient: it returns True in
    containers where the driver is visible but no device is assigned, where the
    driver and runtime versions disagree, and where the GPU is already full. So
    we run a real allocation and a real kernel and check the answer. Anything
    that throws falls back to CPU with a warning rather than failing the
    reconstruction -- a slow image beats no image on a scanner.
    """
    global _DEVICE
    if _DEVICE is not None:
        return _DEVICE

    pref = str(preference).strip().lower()
    if pref in ("cpu", "off", "false", "0"):
        logging.info("Device: CPU (requested)")
        _DEVICE = torch.device("cpu")
        return _DEVICE

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is False")
        device = torch.device("cuda")
        probe = torch.ones(1024, device=device, dtype=torch.float32) * 2.0
        torch.cuda.synchronize()
        if float(probe.sum().item()) != 2048.0:
            raise RuntimeError("CUDA arithmetic self-test returned the wrong value")
        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        logging.info("Device: CUDA -- %s, %.1f GB, torch %s (CUDA %s)",
                     name, total, torch.__version__, torch.version.cuda)
        _DEVICE = device
    except Exception as exc:
        level = logging.WARNING if pref in ("cuda", "gpu", "on", "true", "1") else logging.INFO
        logging.log(level, "CUDA unavailable (%s: %s); falling back to CPU",
                    type(exc).__name__, exc)
        threads = max(1, len(os.sched_getaffinity(0)))
        torch.set_num_threads(threads)
        logging.info("Device: CPU with %d torch threads", threads)
        _DEVICE = torch.device("cpu")

    return _DEVICE


def get_model(device=None):
    global _MODEL
    device = device or resolve_device()
    if _MODEL is None:
        logging.info("Loading DRUNet from %s onto %s", CHECKPOINT, device)
        _MODEL = load_drunet(CHECKPOINT, device=str(device))
    return _MODEL


def _denoise_stack(planes, sigma, batch, device):
    """Run DRUNet over a stack of 2D planes already scaled into [0, 1]."""
    model = get_model(device)
    out = np.empty_like(planes)
    with torch.no_grad():
        for start in range(0, planes.shape[0], batch):
            chunk = torch.from_numpy(planes[start:start + batch]).unsqueeze(1).to(device)
            noise = torch.full_like(chunk, float(sigma))
            res = model(torch.cat([chunk, noise], dim=1)).squeeze(1)
            out[start:start + batch] = res.cpu().numpy()
    return out


def denoise_volume(volume, sigma, axes=1, batch=32, device=None):
    """Denoise a complex 3D volume slice-wise with the 2D DRUNet prior.

    Real and imaginary parts go through as separate images in one stack, so the
    prior acts on the complex iterate rather than on its magnitude.
    """
    device = device or resolve_device()
    scale = float(np.abs(volume).max())
    if scale <= 0:
        return volume
    v = volume / scale

    accum = np.zeros_like(v)
    for axis in range(int(axes)):
        moved = np.moveaxis(v, axis, 0)
        planes = np.concatenate([moved.real, moved.imag]).astype(np.float32)
        # DRUNet expects roughly [0, 1]; real/imag are signed, so shift into range
        den = _denoise_stack(planes * 0.5 + 0.5, sigma * 0.5, batch, device)
        den = (den - 0.5) * 2.0
        n = moved.shape[0]
        accum += np.moveaxis(den[:n] + 1j * den[n:], 0, axis)

    return (accum / float(axes) * scale).astype(np.complex64)


def build_normal_operator(maps, coord, weights, shape, lamda, device, use_toeplitz=True,
                          max_workers=1):
    """Toeplitz normal operator, run through torch so it follows the device.

    Identical maths to reconops.build_normal_operator -- the point spread
    function still comes from one NUFFT on a doubled grid with doubled
    coordinates, and the coil maps are still applied around each convolution
    rather than factored out. Only the FFT pair moves onto the GPU, which is
    where the inner loop's time goes once the NUFFTs are out of it.

    The result is calibrated and checked against the CPU direct operator exactly
    as reconops does. On mismatch, or if anything on the device throws (a small
    card will OOM on the 2N kernel), it falls back to the tested CPU path rather
    than to something unverified.
    """
    if not use_toeplitz or device.type != "cuda":
        return reconops.build_normal_operator(maps, coord, weights, shape, lamda,
                                              use_toeplitz=use_toeplitz,
                                              max_workers=max_workers)

    try:
        big = tuple(2 * s for s in shape)
        logging.info("Toeplitz (torch/%s): building %s point spread function", device.type, big)
        psf = sigpy.nufft_adjoint(weights.astype(np.complex64),
                                  (2.0 * coord).astype(np.float32), big)
        kernel = torch.from_numpy(np.fft.ifftshift(psf).astype(np.complex64)).to(device)
        kernel = torch.fft.fftn(kernel)
        maps_t = torch.from_numpy(maps).to(device)
        inner = tuple(slice((b - s) // 2, (b - s) // 2 + s) for b, s in zip(big, shape))

        def unscaled_t(x_t):
            out = torch.zeros(shape, dtype=torch.complex64, device=device)
            for c in range(maps_t.shape[0]):
                padded = torch.zeros(big, dtype=torch.complex64, device=device)
                padded[inner] = maps_t[c] * x_t
                conv = torch.fft.ifftn(kernel * torch.fft.fftn(padded))[inner]
                out += torch.conj(maps_t[c]) * conv
            return out

        rng = np.random.default_rng(0)
        probe = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)
        reference = reconops._direct_normal(probe, maps, coord, weights, shape)
        approx = unscaled_t(torch.from_numpy(probe).to(device)).cpu().numpy()
        denom = float(np.vdot(approx, approx).real)
        scale = float(np.vdot(approx, reference).real / denom) if denom > 0 else 0.0
        residual = np.linalg.norm(reference - scale * approx) / max(np.linalg.norm(reference), 1e-30)
        logging.info("Toeplitz (torch): scale %.6g, relative disagreement %.3e", scale, residual)
        if not np.isfinite(scale) or residual > 5e-2:
            raise RuntimeError(f"device Toeplitz disagrees with the direct operator ({residual:.3e})")

        def normal(x):
            x_t = torch.from_numpy(np.ascontiguousarray(x)).to(device)
            out = scale * unscaled_t(x_t)
            return out.cpu().numpy().astype(np.complex64) + lamda * x

        return normal

    except Exception as exc:
        logging.warning("Device Toeplitz unavailable (%s: %s); using the CPU operator",
                        type(exc).__name__, exc)
        return reconops.build_normal_operator(maps, coord, weights, shape, lamda,
                                              use_toeplitz=use_toeplitz,
                                              max_workers=max_workers)


def reconstruct(recon):
    coord = recon.coords_index_units()
    shape = recon.image_shape
    cfg = recon.config
    weights = reconops.density_compensation(recon)
    device = resolve_device(cfg.get("usegpu", "auto"))

    extent = float(np.abs(coord).max())
    logging.info("PnP-DRUNet: %d coils, %d points, matrix %d, |k|max %.2f of %d",
                 recon.num_coils, coord.shape[0], recon.matrix_size, extent, recon.matrix_size // 2)

    maps = reconops.espirit_maps(recon, coord, weights)
    y = recon.kspace.reshape(recon.num_coils, -1)
    b = reconops.adjoint(weights * y, maps, coord, shape)

    outer = int(cfg["pnpiterations"])
    inner = int(cfg["pnpinnercg"])
    s_max, s_min = float(cfg["pnpsigmamax"]), float(cfg["pnpsigmamin"])
    lam = float(cfg["pnplambda"])
    sigmas = np.geomspace(s_max, s_min, outer)

    x = np.zeros(shape, dtype=np.complex64)
    for k, sigma in enumerate(sigmas, 1):
        # DPIR ties the data weight to the current noise level: mu = lambda/sigma^2
        mu = lam / (sigma ** 2)
        normal = build_normal_operator(
            maps, coord, weights, shape, mu, device,
            use_toeplitz=bool(cfg["usetoeplitz"]), max_workers=recon.max_workers)
        z = reconops.conjugate_gradient(normal, b + mu * x, max_iter=inner)
        x = denoise_volume(z, sigma, axes=int(cfg["denoiseaxes"]),
                           batch=int(cfg["denoisebatch"]), device=device)
        logging.info("PnP %d/%d  sigma=%.4f mu=%.4g  |x|max=%.4g",
                     k, outer, sigma, mu, float(np.abs(x).max()))

    recon.save_debug("pnp_maps", maps)
    return np.abs(x).astype(np.float32)


def process(connection, config, metadata):
    return mrdrecon.run(connection, config, metadata, reconstruct)
