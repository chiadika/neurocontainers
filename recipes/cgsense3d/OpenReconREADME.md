# cgsense3d

CG-SENSE for 3D centre-out radial, with ESPIRiT coil maps and a
Toeplitz-accelerated normal operator. Built to run inside a scanner's
reconstruction time budget.

## What it solves

    min_x  || W^(1/2) (A x - y) ||^2 + lamda ||x||^2,    (A x)_c = F_nu (S_c x)

by conjugate gradient on `(A^H W A + lamda I) x = A^H W y`. The right-hand side
is the density-compensated adjoint reconstruction, so CG starts from a gridding
image and iterates from there.

## The bundled trajectory

Integrated from the Pulseq gradient waveforms in `radial3d_n6434_s117_c0.seq`,
not measured. Stored as dataset `k`, shape `(32, 6434, 3)`, units cycles/cm.

| | |
| --- | --- |
| Spokes | 6434, centre-out, isotropic directions (`mean direction ~ 0`) |
| Samples per spoke | 32 at 20 µs dwell, all on the gradient flat top |
| Gradient | constant \|G\| = 113636 Hz/m, trapezoid 20/640/20 µs |
| \|k\|max | 71.59 1/m = 0.7159 cycles/cm |
| Resolution | 1/(2·kmax) = **6.98 mm** |
| Matrix the data resolves | 0.22 m / 6.98 mm = **31.5**, i.e. ~32³ |
| Spokes for 3D radial Nyquist | 4π(kmax·FOV)² = 3117 → this data is **2.1× oversampled** |

Because the first ADC sample sits at the end of the gradient ramp, every sample
lies on the flat top and the radial sample spacing is uniform. That makes the
analytic `w = |k|²` density compensation exactly right, which is why this app
does not spend 15 NUFFT pairs on Pipe-Menon.

**Read the resolution row before choosing `matrixsize`.** At 128³ the data fills
only the central 25 % of k-space, the system is ~10× underdetermined
(205 888 samples against 2 097 152 unknowns), and CG is interpolating rather
than resolving. At 32³ it is ~6× overdetermined and well conditioned. 64³ is a
reasonable middle if you want smoother-looking images.

## Speed

The NUFFT cost here is dominated by interpolating 205 888 points through a 4³
kernel, which barely depends on matrix size -- measured ~10 s per forward+adjoint
pair at 128³, and still ~5 s at 32³. Two NUFFTs per coil per iteration is
therefore the whole runtime.

The Toeplitz operator removes them from the inner loop: `A^H W A` is a
convolution, so after one setup pass each iteration is an FFT pair on a doubled
grid per coil. Its absolute scale depends on NUFFT normalisation conventions, so
it is calibrated against one application of the direct operator at startup and
the agreement asserted. If it disagrees by more than 5 %, the app logs a warning
and falls back to the direct operator rather than silently rescaling `lamda`.

Set `usetoeplitz` false to force the direct path.

## Coil sensitivities

ESPIRiT (`sigpy.mri.app.EspiritCalib`), not the smoothed matched filter that
`sodiumgridding` uses. ESPIRiT needs Cartesian calibration data, so the
non-uniform samples are gridded to a small Cartesian volume first -- at
`espiritmatrix` (default 32), the trajectory's own resolution, where the
calibration region is fully sampled. The maps are then zero-pad interpolated up
to the output matrix, which is safe because sensitivities are smooth.

Calibrating at 128³ instead would waste most of the work on the empty outer 75 %
of k-space.

If `espiritcrop` leaves an almost-empty support the app raises rather than
returning noise; lower it, or check that the trajectory matches the data.

## Parameters that matter

| | |
| --- | --- |
| `matrixsize` | 128 as configured. See the resolution note above. |
| `cgiterations` | 12. This *is* the regulariser -- stop early rather than converge. |
| `cglamda` | 1e-3 Tikhonov weight. Raise it if the image looks noisy. |
| `espiritmatrix` | 32. Raise only if the coil maps look too coarse. |
| `usetoeplitz` | true. Turn off only to check the fast path against the slow one. |

## Developing

```bash
python /opt/code/python-ismrmrd-server/twix2mrd.py meas.dat -o raw.h5
python /opt/code/python-ismrmrd-server/run_local.py raw.h5 -o recon.nii.gz -m cgsense3d
python /opt/code/python-ismrmrd-server/run_local.py raw.h5 -o r32.nii.gz -m cgsense3d -p matrixsize=32
```

`run_local.py` uses the same `prepare()` → `reconstruct()` path as the server.
