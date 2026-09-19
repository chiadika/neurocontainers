# twix2mrd.py — changes required to get it running

Tracking every code change made to get `twix2mrd.py` working end-to-end,
in the order they were made.

## 1. `ismrmrdHeader.__init__()` missing required `experimentalConditions`

**Error:**
```
TypeError: ismrmrdHeader.__init__() missing 1 required keyword-only argument: 'experimentalConditions'
```

**Cause:** The installed `ismrmrd.xsd` module defines `ismrmrdHeader` as a
dataclass where `experimentalConditions` is a required keyword-only field
(no default value), unlike `measurementInformation` and
`acquisitionSystemInformation`, which default to `None`. The original code
called `ismrmrd.xsd.ismrmrdHeader()` with no arguments and set attributes
afterward, which fails at construction time before any attributes can be
assigned.

**Fix:** In `_build_header()` ([twix2mrd.py](twix2mrd.py)), build
`measurement_information`, `acquisition_system_information`, and
`experimental_conditions` first, then pass them into the
`ismrmrd.xsd.ismrmrdHeader(...)` constructor as keyword arguments instead of
assigning them after an empty construction.

## 2. Same pattern repeats through the rest of `_build_header()`

**Error (after fix #1):**
```
TypeError: measurementInformationType.__init__() missing 1 required keyword-only argument: 'patientPosition'
```

**Cause:** Every dataclass in this `ismrmrd.xsd` module follows the same
rule — fields with no sensible default are required, keyword-only
constructor arguments; the original script's pattern of `Type()` then
attribute assignment only works for fields that already default to `None`.
Inspecting each class used in the script
(`python3 -c "import ismrmrd.xsd, inspect; ..."`) showed the following
required fields with no defaults:

| Type | Required fields |
|---|---|
| `measurementInformationType` | `patientPosition` |
| `experimentalConditionsType` | `H1resonanceFrequency_Hz` |
| `encodingType` | `encodedSpace`, `reconSpace`, `encodingLimits`, `trajectory` |
| `encodingSpaceType` | `matrixSize`, `fieldOfView_mm` |
| `fieldOfViewMm` | `x`, `y`, `z` |
| `ismrmrdHeader` | `experimentalConditions` |

`acquisitionSystemInformationType`, `matrixSizeType`, `encodingLimitsType`,
and `limitType` have no required fields, so `Type()` + attribute assignment
still works for those, but was switched to constructor kwargs anyway for
consistency where it was cheap to do so.

**Fix:** Rewrote `_build_header()` in [twix2mrd.py](twix2mrd.py) to build
objects bottom-up (`matrixSizeType`/`fieldOfViewMm` → `encodingSpaceType` →
`encodingType`; `limitType` → `encodingLimitsType`) and pass every required
field as a constructor keyword argument at creation time, rather than
constructing empty objects and assigning attributes afterward.

## Verified working

```
python3 twix2mrd.py -f meas_MID00083_FID00701_tpiTqf_HighSp_n72_p9_g5_TE0_07_Liliana.dat -o echo1_test.h5
```
completes successfully:
```
Converted Twix imaging data to MRD: echo1_test.h5 (readouts=23770, coils=64, samples=192)
```

# Rename fallout — sodiumgridding → myrecon

This recipe was forked from `recipes/sodiumgridding/`. The reconstruction
script itself (`myrecon.py`) was renamed and updated internally
(`"config": "myrecon"`, `/opt/myrecon/...` paths, `OUTPUT_SERIES_DESCRIPTION
= "myrecon"`), but several other files that also embed the old name were not
updated to match, which broke `python -m pytest recipes/myrecon_sodiumgridding/test_*.py`
end to end with `ModuleNotFoundError: No module named 'sodiumgridding'` and,
after that, several more failures one at a time.

## 3. `build.yaml` was an unedited copy of `recipes/sodiumgridding/build.yaml`

**Cause:** `name: sodiumgridding`, the `files:` entry `filename:
sodiumgridding.py` (which doesn't exist in this directory — only
`myrecon.py` does), `NUMBA_CACHE_DIR: /tmp/numba-sodiumgridding`, `workdir:
/opt/sodiumgridding`, and every `cp` directive still targeted
`sodiumgridding.py`/`/opt/sodiumgridding/...`. This would have failed at
`sf-build`, not just at test time.

**Fix:** Updated `name`, the `files:` entry, `NUMBA_CACHE_DIR`, `workdir`,
and both `cp` directives (script copy and trajectory copy) to `myrecon`/
`/opt/myrecon`. Updated the inline `readme:` block's build commands and
recipe path too.

## 4. `test_myrecon.py` still imported the module as `sodiumgridding`

**Error:**
```
ModuleNotFoundError: No module named 'sodiumgridding'
```

**Cause:** `test_myrecon.py` is a copy of `recipes/sodiumgridding/test_sodiumgridding.py`.
`importlib.import_module("sodiumgridding")` has nothing to find on
`sys.path` — the file in this directory is `myrecon.py`.

**Fix:** Changed the import target to `"myrecon"`. This got the module
importing, but surfaced a second wave of failures: several assertions in the
same file still expected the *old* metadata strings that `sodiumgridding.py`
emits, which don't match what `myrecon.py` actually emits (it uses a plain
lowercase `myrecon` prefix, not an uppercased or PascalCase form). Fixed
each of the following to match `myrecon.py`'s real output:
- `SeriesDescription`/`SeriesNumberRangeNameUID` suffix: `_sodiumgridding` → `_myrecon`
- `ImageType`/`DicomImageType`/`ImageTypeValue4`: `SODIUMGRIDDING` → `myrecon`
- `meta["SodiumGriddingDisplay*"]` keys → `meta["myreconDisplay*"]`
- `meta["SodiumGriddingOrientation"]`/`OrientationFlipSlice` → `meta["myreconOrientation"]`/`OrientationFlipSlice`
- debug array filenames `sodiumgridding_coil_images.npy` /
  `sodiumgridding_magnitude_volume.npy` → `myrecon_coil_images.npy` /
  `myrecon_magnitude_volume.npy`

These didn't show up all at once — pytest only reports the first assertion
failure per test, so each fix revealed the next one underneath it.

## 5. `mrd2nifti.py` silently skipped the frame-reversal correction

**Symptom:** `test_mrd2nifti_undoes_the_frame_reversal` failed with the
converted NIfTI's centroid on the wrong side (`patient[2] == 61.875`
instead of `< 0.0`), while the byte-for-byte-equivalent test against
`recipes/sodiumgridding/` passed. No exception, no import error — just
wrong geometry.

**Cause:** `mrd2nifti.py` hardcodes
`FRAME_ORDER_REVERSED_ATTRIBUTE = "SodiumGriddingIceFrameOrderReversed"`,
deliberately duplicated (not imported) from the recon script's
`OUTPUT_FRAME_ORDER_REVERSED_ATTRIBUTE` constant so the converter has no
runtime dependency on the recon module. But `myrecon.py` writes that flag
under the key `"myreconIceFrameOrderReversed"` — a different string — so
`mrd2nifti.py` never found the flag, treated frames as not reversed, and
built an affine from the raw (mirrored) header. This is the most dangerous
class of leftover: it doesn't error, it just quietly produces wrong patient
geometry.

**Fix:** Changed `FRAME_ORDER_REVERSED_ATTRIBUTE` to
`"myreconIceFrameOrderReversed"` to match `myrecon.py`.

## 6. `OpenReconLabel.json` still pointed the scanner at the old name

**Cause:** The OpenRecon UI sends whatever `config` value is selected
straight to the ISMRMRD server, which dispatches to the Python module of
that name. `OpenReconLabel.json` still had `general.id: "sodiumgridding"`,
the `config` parameter's choice `id`/`name`/`default` set to
`"sodiumgridding"`, and both `trajectoryfile` choice `id`s / the default
pointing at `/opt/sodiumgridding/...` — none of which match `myrecon.py` or
the `/opt/myrecon/...` paths fixed in `build.yaml` above. Left unfixed, the
scanner would never have been able to route to this reconstruction at all.

**Fix:** Updated `general.name`/`id`/`device_trade_name`/`material_number`,
the `config` parameter's choice `id`/`name`/`default`, and both
`trajectoryfile` choice `id`s/`default` to `myrecon`/`/opt/myrecon/...`.
Left `regulatory_information.gtin`/`udi`/`manufacture_date` untouched —
those are device-registration fields that need a human decision, not a
find-and-replace.

**Not fixed (cosmetic, doesn't affect execution):** `OpenReconREADME.md`
still says `sodiumgridding` throughout (module name, config table, trajectory
paths, the `SodiumGriddingIceFrameOrderReversed` attribute name, and the
recipe URL). Worth a pass for accuracy, but nothing reads this file at
build or run time.

## Renaming checklist for the next fork

A plain `grep -ri sodiumgridding .` in the recipe directory before calling a
rename "done" would have caught all of the above in one shot, instead of
one failure at a time. Everywhere the tool's name has to be threaded
through when forking a recipe like this one:

1. **`build.yaml`** — top-level `name:`, the `files:` entry for the recon
   script, `NUMBA_CACHE_DIR`, `workdir:`, every `cp` destination (both
   `/opt/code/python-ismrmrd-server/<name>.py` and
   `/opt/<name>/...trajectory.h5`), and the readme block.
2. **The recon script itself** — `"config"` in `OPENRECON_DEFAULTS` (this is
   what the scanner sends to select the handler), `BUNDLED_TRAJECTORIES`
   paths, `OUTPUT_SERIES_DESCRIPTION`, `OUTPUT_FRAME_ORDER_REVERSED_ATTRIBUTE`,
   the `ImageType`/`DicomImageType`/`ImageTypeValue4` literals, every
   `meta["<Name>Display...">]`/`meta["<Name>Orientation..."]` key, and the
   debug `.npy` filename prefixes.
3. **`mrd2nifti.py`** — the sneaky one. `FRAME_ORDER_REVERSED_ATTRIBUTE` is a
   hardcoded copy of the recon script's constant, by design not imported.
   Nothing errors if it's stale — it just silently stops applying the
   frame-reversal fix, and you get mirrored geometry. Easy to miss because
   there's no `ModuleNotFoundError` to catch it.
4. **`OpenReconLabel.json`** — `general.name`/`id`, the `config` parameter's
   `values[].id`/`default` (must equal the recon script's `"config"` string
   exactly, or the scanner can't route to it), and the `trajectoryfile`
   choice `id`s/`default` (must equal the `build.yaml` destination paths).
5. **`fulltest.yaml`** — `name:`/`version:` must match the recipe
   directory/`build.yaml` per AGENTS.md, plus every `import <module>` and
   trajectory path inside test scripts.
6. **`test_*.py`** (if copied from the source recipe) — the import target,
   and *every* string literal asserted against the module's metadata
   keys/values, not just the obvious ones. These don't fail at import time;
   they fail one by one as `KeyError`/`AssertionError` once the import
   itself works, so a partial rename can look "done" after fixing the first
   error and still have several more hiding underneath it.
7. **Docs** (`OpenReconREADME.md`) — not functionally required, but worth a
   pass for accuracy.

# Bundling the n72 sodium trajectory

The test dataset `meas_MID00083_FID00701_tpiTqf_HighSp_n72_p9_g5_TE0_07_Liliana.dat`
(23770 imaging readouts) doesn't match either bundled preset — `build.yaml`
only shipped `23Na_n50_trajectory.h5` and `23Na_n28_trajectory.h5`. A matching
trajectory file, `kspace_trajectory.h5` (dataset `k`, shape
`(206, 23770, 3)` — readout count matches the .dat exactly), was placed in
the recipe directory and bundled the same way as the existing presets.

**Confirmed the raw data itself carries no embedded trajectory** first
(inspected MDH flags across the imaging readouts with `twixtools`: only
`FIRSTSCANINSLICE`, `ONLINE`, `RAWDATACORRECTION`, `SYNCDATA` — no trajectory
flag), so this sequence genuinely requires an external trajectory file; there
was no way to avoid bundling one.

**Changes:**
- `build.yaml`: added a `files:` entry `sodium_trajectory_n72` (`filename:
  kspace_trajectory.h5`, local — unlike the n50/n28 entries, which fetch from
  a remote object-store URL), and a `cp` directive copying it to
  `/opt/myrecon/23Na_n72_trajectory.h5`. Bumped `version: 0.1.6` → `0.1.7`.
- `myrecon.py`: added `"sodiumn72"` and `"23Na_n72"` keys to
  `BUNDLED_TRAJECTORIES`, pointing at `/opt/myrecon/23Na_n72_trajectory.h5`.
- `OpenReconLabel.json`: added a `"23Na n72"` entry to the `trajectoryfile`
  choice parameter's `values`, pointing at the same path.
- `OpenReconREADME.md`: updated the bundled-trajectory list/aliases and the
  GUI parameter table to mention n72.
- `fulltest.yaml`: added `/opt/myrecon/23Na_n72_trajectory.h5` to the
  packaged-asset existence/readability check alongside n50/n28, and bumped
  its own `version: 0.1.6` → `0.1.7` to match `build.yaml` (per the renaming
  checklist above: this file's `name:`/`version:` must track the recipe).

## n72 made the default trajectory

Switched the default from n28 to n72:
- `myrecon.py`: `OPENRECON_DEFAULTS["trajectorypreset"]` changed
  `"23Na_n28"` → `"23Na_n72"`.
- `OpenReconLabel.json`: the `trajectoryfile` parameter's `"default"` changed
  to `/opt/myrecon/23Na_n72_trajectory.h5`.
- `OpenReconREADME.md`: updated the parameter table's default column and
  wording to list n72 first.

Checked `test_myrecon.py` for anything pinned to the old default first —
nothing asserts against `OPENRECON_DEFAULTS["trajectorypreset"]` or a
default `trajectoryfile` value (the one `n28` string in that file is an
unrelated fixture protocol name, `tpiTqf_23Na_n28_TE05_FIRE`), so no test
changes were needed. n28 and n50 both remain available as explicit choices;
this only changes what's used when nothing is specified.

**Rebuild steps** (from the `neurocontainers` repo root, with the builder's
venv active):
```bash
source env/bin/activate
python -m builder generate myrecon_sodiumgridding --recreate --architecture x86_64
sf-build myrecon_sodiumgridding --architecture x86_64
```
`--recreate` regenerates the Dockerfile from the edited `build.yaml` (picking
up the new `files:`/`cp` entries); `sf-build` then actually builds the image,
re-downloading/copying every bundled file including `kspace_trajectory.h5`.
