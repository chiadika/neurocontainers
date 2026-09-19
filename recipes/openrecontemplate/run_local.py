#!/usr/bin/env python3
"""Run the reconstruction on an MRD file, with no scanner and no server.

This is the loop you want while developing an algorithm: edit
``openrecontemplate.py``, run this, look at the NIfTI. It calls exactly the same
``prepare`` -> ``reconstruct`` path the OpenRecon server does, so anything that
works here works in the app -- only the image emission and the patient-space
orientation are skipped.

    python run_local.py raw.h5 -o recon.nii.gz
    python run_local.py raw.h5 -o recon.nii.gz -p matrixsize=64 -p dcfiterations=0
    python run_local.py raw.h5 -o recon.nii.gz --module example_cgsense

Convert a Siemens raw file to MRD first with ``twix2mrd.py`` or
``siemens_twix2mrd.py``.

The output is in the acquisition frame ([z, y, x] as the trajectory defines it),
not the DICOM display frame -- ``mrdrecon`` only applies the orientation stages
when it emits images to the scanner.
"""

import argparse
import importlib
import logging
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ismrmrd  # noqa: E402

import mrdrecon  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reconstruct an MRD file offline using the template pipeline.",
    )
    parser.add_argument("input", help="MRD (ISMRMRD) HDF5 file with raw acquisitions")
    parser.add_argument("-o", "--output", default="recon.nii.gz",
                        help="output NIfTI (default: recon.nii.gz)")
    parser.add_argument("-m", "--module", default="openrecontemplate",
                        help="module providing reconstruct() (default: openrecontemplate)")
    parser.add_argument("-g", "--group", default="dataset",
                        help="HDF5 group holding the acquisitions (default: dataset)")
    parser.add_argument("-p", "--param", action="append", default=[], metavar="KEY=VALUE",
                        help="override an OpenRecon parameter; repeatable")
    parser.add_argument("--npy", action="store_true",
                        help="also write the volume as .npy")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def load_acquisitions(path, group):
    dataset = ismrmrd.Dataset(path, group, create_if_needed=False)
    metadata = ismrmrd.xsd.CreateFromDocument(dataset.read_xml_header())
    count = dataset.number_of_acquisitions()
    logging.info("Reading %d acquisitions from %s", count, path)

    skip = (
        ismrmrd.ACQ_IS_NOISE_MEASUREMENT,
        ismrmrd.ACQ_IS_PARALLEL_CALIBRATION,
        ismrmrd.ACQ_IS_PHASECORR_DATA,
        ismrmrd.ACQ_IS_NAVIGATION_DATA,
    )
    acquisitions = []
    for index in range(count):
        acquisition = dataset.read_acquisition(index)
        if not any(acquisition.is_flag_set(flag) for flag in skip):
            acquisitions.append(acquisition)
    dataset.close()

    if not acquisitions:
        raise SystemExit(f"No imaging acquisitions found in {path}")
    logging.info("Kept %d imaging readouts", len(acquisitions))
    return acquisitions, metadata


def write_nifti(volume, path, fov_cm):
    voxel_mm = (float(fov_cm) * 10.0) / float(volume.shape[0])
    try:
        import nibabel as nib
    except ImportError:
        fallback = Path(path).with_suffix(".npy")
        np.save(fallback, volume)
        logging.warning("nibabel unavailable; wrote %s instead", fallback)
        return fallback

    # volume is [z, y, x]; NIfTI wants [x, y, z].
    affine = np.diag([voxel_mm, voxel_mm, voxel_mm, 1.0])
    nib.save(nib.Nifti1Image(np.transpose(volume, (2, 1, 0)), affine), path)
    return path


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
    )

    overrides = {}
    for item in args.param:
        if "=" not in item:
            raise SystemExit(f"--param expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        overrides[key.strip()] = value.strip()

    module = importlib.import_module(args.module)
    if not hasattr(module, "reconstruct"):
        raise SystemExit(f"{args.module} does not define reconstruct()")

    acquisitions, metadata = load_acquisitions(args.input, args.group)
    config = {"parameters": overrides}

    tic = perf_counter()
    recon = mrdrecon.prepare(acquisitions, config, metadata)
    logging.info("Prepared in %.1f s", perf_counter() - tic)
    logging.info("Configuration: %s", recon.config)

    tic = perf_counter()
    volume = np.asarray(module.reconstruct(recon), dtype=np.float32)
    logging.info("Reconstructed in %.1f s", perf_counter() - tic)

    if volume.shape != recon.image_shape:
        raise SystemExit(
            f"reconstruct() returned {volume.shape}, expected {recon.image_shape}"
        )
    logging.info(
        "Volume: shape=%s min=%.4g max=%.4g mean=%.4g",
        volume.shape, volume.min(), volume.max(), volume.mean(),
    )

    if args.npy:
        np.save(os.path.splitext(args.output)[0] + ".npy", volume)
    written = write_nifti(volume, args.output, recon.fov_cm)
    logging.info("Wrote %s", written)


if __name__ == "__main__":
    main()
