# pnpdrunet

Plug-and-play reconstruction for 3D centre-out radial, using a DRUNet denoiser
as the regulariser instead of a quadratic penalty.

> **Research use only.** DRUNet was trained on natural grayscale images, not on
> MR data. It will impose plausible-looking structure that the acquisition does
> not constrain. Always reconstruct the same data with `cgsense3d` and compare
> before trusting any feature that appears here and not there.

## What it does

CG-SENSE solves `min_x ||W^½(Ax−y)||² + λ‖x‖²`. The `λ‖x‖²` term is a crude
statement of "images are small". Plug-and-play replaces it with a trained
denoiser via half-quadratic splitting (Zhang et al., DPIR, TPAMI 2021):

```
z_k = argmin_z ||W^½(Az − y)||² + μ_k ||z − x_k||²      (data)
x_k = D_σk(z_k)                                          (prior)
```

The data step is the same Toeplitz normal operator `cgsense3d` uses, warm-started
from the previous iterate, so 4 inner CG steps suffice. `σ` decreases
geometrically from `pnpsigmamax` to `pnpsigmamin`, and `μ_k = λ/σ_k²` couples the
two -- early iterations denoise hard and pull the iterate onto the image
manifold, later ones barely touch it and let data consistency finish.

## Why this and not DPS

Diffusion posterior sampling needs ~1000 network evaluations, each wrapped in a
forward and adjoint operator application. On this trajectory a NUFFT pair costs
about 10 s at 128³, so DPS is an hours-per-volume method. This needs
`pnpiterations` (default 6) network passes. That is the whole argument.

The other half of the argument: your 2× undersampled sequence is exactly where
a prior earns its cost. On the Nyquist trajectory the data already determines
the image and the prior mostly adds bias -- use `cgsense3d` there.

## Three implementation choices

**DRUNet is 2D**, so it is applied slice-wise. `denoiseaxes: 3` denoises along
all three orthogonal orientations and averages, which removes through-plane
anisotropy at 3× the prior cost. Default is axial only.

**The iterate is complex.** Real and imaginary parts are denoised as separate
images in a single batch, rather than denoising the magnitude, so the prior
remains a genuine drop-in for the quadratic penalty rather than an operation on
a nonlinear function of the iterate.

**DRUNet expects roughly [0, 1].** The volume is rescaled into that range before
each prior step and back afterwards, so `pnpsigmamax`/`pnpsigmamin` are on a
normalised intensity scale, not in image units.

## The checkpoint

`drunet_gray.pth` from the DPIR release (`cszn/KAIR` v1.0), 130 MB, md5
`f3a001301c519d4c18438a2f4e87cb68`. The architecture in `drunet.py` is vendored
so the container needs no extra package, and the weights are loaded with
`strict=True` -- any drift between the vendored definition and the checkpoint
fails loudly instead of silently producing noise. The container test asserts
both the md5 and a real denoising result.

## Tuning

| | |
| --- | --- |
| `pnpiterations` | 6. More is not obviously better; watch for over-smoothing. |
| `pnpsigmamax` | 0.10. Raise if the starting image is very noisy. |
| `pnpsigmamin` | 0.02. **This sets how smooth the result looks.** Lower it if detail is being erased. |
| `pnplambda` | 0.23 (DPIR's default). Raise to weight data consistency over the prior. |
| `denoiseaxes` | 1. Set 3 if through-plane blockiness is visible. |

If the result looks like a cartoon, lower `pnpsigmamin` first, then raise
`pnplambda`. If it looks like CG-SENSE, the prior is not engaging -- raise
`pnpsigmamax`.

## Developing

```bash
python /opt/code/python-ismrmrd-server/run_local.py raw.h5 -o pnp.nii.gz -m pnpdrunet
python /opt/code/python-ismrmrd-server/run_local.py raw.h5 -o cg.nii.gz  -m cgsense3d   # if present
python /opt/code/python-ismrmrd-server/run_local.py raw.h5 -o pnp3.nii.gz -m pnpdrunet -p denoiseaxes=3
```
