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

Trajectories in `fire\share` (mounted as `/tmp/share`) are also candidates for
`auto`, matched by `(samples, readouts)`, and the 512^3 ones are meant to live
there rather than in the image:

| file in fire\share | shape | matrixsize |
|---|---|---|
| cones3d_n206435_trajectory.h5 | (~360, 206435), with dcf | **512**, fovcm 25.0 |
| cones3d_n412489_trajectory.h5 | (~360, 412489), with dcf | **512**, fovcm 25.0 |

Coils are gridded in parallel processes (`maxworkers`, default 8), capped by
the CPUs and memory the container is given. The result does not depend on the
worker count. Sum-of-squares combination is streamed, so the full stack of coil
images is never held; adaptive combine still needs it.

Scanner-side: `Ice\fire\config\wip_070_fire_quickgrid.json` holds the
parameters; the workflow XML anchors the Marshal at `Flags` and the Injector at
`imafinish` (see `~/_seqdev/FIRE_ICE_PIPELINE.md`).

Offline:

    python /opt/code/python-ismrmrd-server/run_local.py raw.h5 -o out.nii.gz -m quickgrid -p matrixsize=134
