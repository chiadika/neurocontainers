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

Coils are gridded in parallel processes (`maxworkers`, default 8), capped by
the CPUs and memory the container is given. The result does not depend on the
worker count. Sum-of-squares combination is streamed, so the full stack of coil
images is never held; adaptive combine still needs it.

Scanner-side: `Ice\fire\config\wip_070_fire_quickgrid.json` holds the
parameters; the workflow XML anchors the Marshal at `Flags` and the Injector at
`imafinish` (see `~/_seqdev/FIRE_ICE_PIPELINE.md`).

Offline:

    python /opt/code/python-ismrmrd-server/run_local.py raw.h5 -o out.nii.gz -m quickgrid -p matrixsize=134
