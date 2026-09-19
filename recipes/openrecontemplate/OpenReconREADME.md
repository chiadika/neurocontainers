# openrecontemplate

A starting point for writing an OpenRecon reconstruction. The container gives
you a working non-Cartesian reconstruction app; you replace the algorithm.

## The split

| File | Role |
| --- | --- |
| `openrecontemplate.py` | **The file you edit.** App identity, parameter defaults, and `reconstruct()`. |
| `mrdrecon.py` | The plumbing. MRD streaming, trajectory loading, k-space assembly, coil preparation, patient-space geometry, DICOM display scaling. You should not need to touch it. |
| `example_cgsense.py` | A worked iterative example. Read it, copy from it; nothing imports it at runtime. |
| `run_local.py` | Runs the same pipeline on an MRD file with no scanner and no server. |
| `twix2mrd.py`, `siemens_twix2mrd.py` | Siemens raw → MRD. |
| `mrd2nifti.py` | MRD images → NIfTI. |
| `OpenReconLabel.json` | What the scanner UI shows and what the app declares it needs. |

`mrdrecon.py` was carved out of `recipes/sodiumgridding/sodiumgridding.py`,
which is 2249 lines of which roughly 1900 are plumbing. The geometry stages in
particular carry comments recording what was measured on the scanner and why
each sign is what it is -- worth reading before you change any of it.

## The contract

```python
def reconstruct(recon):
    return volume   # real float32, shape recon.image_shape, indexed [z, y, x]
```

What you get:

| | |
| --- | --- |
| `recon.kspace` | `(coils, samples, readouts)` complex64, after preparation. `coils` counts *virtual* coils when `compresscoils` is on. |
| `recon.trajectory` | `(samples, readouts, 3)` float32, native units (1/cm for the bundled sodium trajectories). |
| `recon.raw_kspace` | `(coils, readouts, samples)` complex64, straight off the wire, no preparation at all. |
| `recon.image_shape` | `(N, N, N)`. |
| `recon.config` | Every resolved OpenRecon parameter, defaults applied. |
| `recon.coords_index_units()` | Flat `(samples*readouts, 3)` on `[-N/2, N/2]` -- sigpy's convention. |
| `recon.coords_normalized()` | Flat `(samples*readouts, 3)` on `[-0.5, 0.5]` -- hand-written gridding's convention. |
| `recon.radial_k()` | `|k|` per sample, `(samples, readouts)`. |
| `recon.save_debug(name, array)` | Writes to `/tmp/share/debug` for offline inspection. |

Point order in the flat coordinate arrays matches `recon.kspace[c].ravel()`.

Preparation the framework has already done, each switchable from the UI:
clip data to the trajectory, limit channels (`maxcoils`), reject low-signal
readouts (`rejectbadreadouts`), normalise the k-space centre, apply the Fermi
taper (`applyfermifilter`), and PCA-compress the channels (`compresscoils`).
After you return, it optionally runs N4 (`applyn4biascorrection`), then handles
orientation and display scaling.

Reusable pieces, importable from `mrdrecon`: `estimate_sensitivities`,
`combine_coils`, `compress_coils_by_variance`, `n4_bias_field_correct`,
and `build_fermi_filter`.

`recon.config` is already resolved and type-coerced, so read it with plain dict
access (`recon.config["matrixsize"]`). A key you want to read must appear in
`DEFAULTS` in `openrecontemplate.py`, and in `OpenReconLabel.json` if the
scanner operator should be able to set it.

## Developing

```bash
# Siemens raw -> MRD, once
python /opt/code/python-ismrmrd-server/twix2mrd.py meas_MID00083.dat -o raw.h5

# then loop: edit openrecontemplate.py, run, look
python /opt/code/python-ismrmrd-server/run_local.py raw.h5 -o recon.nii.gz
python /opt/code/python-ismrmrd-server/run_local.py raw.h5 -o cg.nii.gz --module example_cgsense
python /opt/code/python-ismrmrd-server/run_local.py raw.h5 -p matrixsize=64 -p dcfiterations=0
```

`run_local.py` calls the same `prepare()` → `reconstruct()` path the server
does. It writes the volume in the acquisition frame, not the DICOM display
frame -- the orientation stages only run when images are emitted to the scanner.

To exercise the full server path, run `main.py` and point the stock `client.py`
at it, as with any python-ismrmrd-server app.

## Renaming the app

The MRD server loads the module named by the incoming `config` value, so three
names have to agree:

1. the app module filename, `openrecontemplate.py`;
2. `RECON_NAME` inside it, which `mrdrecon.configure()` uses for the series
   description, the DICOM `ImageType`, and the `Meta` attribute prefix;
3. the `config` parameter's `id` and `default` in `OpenReconLabel.json`.

Then update `name:` in `build.yaml`, the `cp` targets, and `/opt/<name>` if you
keep bundled trajectories there.

## Declaring what you need

`OpenReconLabel.json` declares `min_required_memory` (32 GB) and
`min_count_required_cpu_cores` (10). The scanner enforces these. An iterative
reconstruction can easily exceed what a one-pass adjoint needs -- a 256³
complex128 working grid is 268 MB per coil on its own -- so revisit both numbers
before deploying, and cap concurrency with `maxworkers`.
