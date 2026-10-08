# quickgrid

The simplest reconstruction that gives a usable image, for checking a sequence,
trajectory and the FIRE handoff on the scanner in seconds:

    image = combine_coils( NUFFT^H ( w * y ) )

No calibration, no iterations. Density compensation is the shipped `dcf`
dataset when the trajectory file has one (cones), analytic |k|² for uniform
radial, Pipe-Menon otherwise. Coil combination is sum-of-squares by default.

Bundled trajectories (`trajectoryfile: auto` picks by acquired shape):

| key | shape (samples, readouts) | matrixsize |
|---|---|---|
| radial3d_res128_us2x | (128, 25536) | 128 |
| radial3d_res128_nyquist | (128, 51071) | 128 |
| radial3d_n6434 | (32, 6434) | 128 (resolves ~32) |
| cones3d_n4846 | (312, 4846), with dcf | **134** |
| cones3d_n9598 | (312, 9598), with dcf | **134** |
| cones3d_n68785 | (312, 68785), with dcf | **266**, fovcm 25.0 |
| cones3d_n34506 | (312, 34506), with dcf | **278**, fovcm 25.0 |

| cones3d_n206435 | (317, 206435), analytic dcf | **512**, fovcm 25.0 |
| cones3d_n412489 | (317, 412489), analytic dcf | **512**, fovcm 25.0 |
| cones3d_n34506_ir32 (select by name) | (312, 34506), Pipe dcf, random order | **278**, fovcm 25.0 |

`cones3d_n34506_ir32` serves every `…_sp2_ir32_*` IR-prepared sequence: same
readouts as `cones3d_n34506` in a seeded random acquisition order, with the DCF
rows permuted to match and the permutation stored as `acq_order`. It has the
same shape as the plain n34506 file, so `auto` cannot pick it: set
`trajectoryfile` to `/opt/quickgrid/cones3d_n34506_ir32_trajectory.h5` for IR
scans and back to `auto` afterwards.

Orientation (`orientation`, default `xyz` since 1.2.2): the three letters name
the trajectory components placed on (slices, rows, columns); `_fx`/`_fy`
reverse columns/rows and `orientationflipslice` the slices. quickgrid grids
with sigpy, whose output axes follow the trajectory columns, so `xyz` is the
natural order and makes the volume agree with the interpreter's header (read
L-R, phase A-P, slice F-H) with no flips. `zyx`, the pre-1.2.2 default, was
written for a gridder with reversed axes and shows sagittal content in
transversal frames. Verified with an eraser at the left ear and the fill plug
at the vertex of a head phantom. Set `"orientation": "xyz"` in
`wip_070_fire_quickgrid.json` as well.

Trajectories in `fire\share` (mounted as `/tmp/share`) are also candidates for
`auto`, matched by `(samples, readouts)`, so a new one only has to be copied there.

Gradient-nonlinearity (distortion) correction is applied in the recon
(`gradunwarp`: off / 3D / 3Dnojac / 2D, default 3D), because images injected by
FIRE under IceProgramStandard enter the chain after ICE's DistorCor functor. It
needs the scanner's Siemens coefficient file, which is proprietary and **not
bundled**: copy `coeff_IMPULSE.grad` into `fire\share` (or the directory
mounted as /tmp/share on the remote recon server), or point `gradcoeffile` at
it. Without the file the recon logs a warning and emits uncorrected images.
The maths follows gradunwarp (HCP, MIT): spherical-harmonic displacement field
times R0, trilinear resampling of the volume at p + dv(p), Jacobian intensity
factor capped at 10. Images remain tagged ND on the scanner (ICE did not
correct them); `DistorCorMode` stays ND in the XML.

Multi-echo data (1.2.4): `echosplit` 2 reconstructs the two-echo cones files
into two series, `_echo1` (TE 80 µs) and `_echo2`. Acquisitions are assigned to
echoes by their position in the TR (scan counter), or by the contrast counter
when the sequence sets an ECO label. Rewind-and-repeat files (`_e2`, two ADCs
per TR, twice as many acquisitions as trajectory rows) share one trajectory row
per TR; alternating files (`_de2000`, one ADC per TR) use one row each. With
`echosplit` 1 a two-echo stream is silently cropped to the first half and every
second readout lands on the wrong cone, so set it for those files.

Intensity normalisation (1.2.3): `coilcombinemode` `SoSnorm` divides the
sum-of-squares image by its own low-resolution envelope (centre of k-space,
`normfraction` 0.125 of kmax, Gaussian-smoothed, regularised division), which
removes every smooth multiplicative shading, receive and transmit alike, so use
it for display only. `SoSvbc` divides by the array's sensitivity relative to
the first principal virtual coil, a near-uniform combination of all elements
standing in for the missing body coil: the self-calibrated analogue of the
product's Prescan Normalize. Transmit shading is common to both and stays, so
the image remains proportional to the transmit field, which is what a
B1+-corrected quantitative pipeline wants. Neither replaces a measured B1+ map.

Coils are gridded in parallel processes (`maxworkers`, default 8), capped by
the CPUs and memory the container is given. The result does not depend on the
worker count. Sum-of-squares combination is streamed, so the full stack of coil
images is never held; adaptive combine still needs it.

Scanner-side: `Ice\fire\config\wip_070_fire_quickgrid.json` holds the
parameters; the workflow XML anchors the Marshal at `Flags` and the Injector at
`imafinish` (see `~/_seqdev/FIRE_ICE_PIPELINE.md`).

Offline:

    python /opt/code/python-ismrmrd-server/run_local.py raw.h5 -o out.nii.gz -m quickgrid -p matrixsize=134
