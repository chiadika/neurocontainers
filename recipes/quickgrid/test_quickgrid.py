"""Focused tests for the quickgrid app module.

Run with:  python -m pytest recipes/quickgrid/test_*.py -q

The scanner-side modules (`mrdhelper`, `constants`) are stubbed the same way the
sibling recipes do it; everything else is the real code.
"""

import importlib
import json
import sys
import types
from pathlib import Path

import h5py
import numpy as np
import pytest

RECIPE_DIR = Path(__file__).resolve().parent


def _import_quickgrid(monkeypatch):
    constants = types.ModuleType("constants")
    constants.MRD_LOGGING_INFO = 1
    mrdhelper = types.ModuleType("mrdhelper")
    mrdhelper.update_img_header_from_raw = lambda image_header, reference_head: image_header
    mrdhelper.get_json_config_param = lambda cfg, key, default=None, type=None: cfg.get(key, default)
    monkeypatch.syspath_prepend(str(RECIPE_DIR))
    monkeypatch.setitem(sys.modules, "constants", constants)
    monkeypatch.setitem(sys.modules, "mrdhelper", mrdhelper)
    for name in ("quickgrid", "mrdrecon"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    return importlib.import_module("quickgrid")


def _radial_trajectory(samples=16, spokes=300, kmax=0.5):
    """Uniform centre-out 3D radial in cycles/cm, shape (samples, spokes, 3)."""
    n = np.arange(1, spokes + 1)
    z = 1 - 2 * (n - 0.5) / spokes
    r = np.sqrt(np.maximum(0.0, 1 - z * z))
    az = np.deg2rad(137.508) * n
    d = np.stack([r * np.cos(az), r * np.sin(az), z], axis=1)
    radius = (np.arange(samples) + 0.5) / samples * kmax
    return (radius[:, None, None] * d[None, :, :]).astype(np.float32)


def _recon_input(m, traj, kspace, matrix, fov=22.0, source=None, **config):
    return m.mrdrecon.ReconInput(
        kspace=kspace, trajectory=traj, raw_kspace=np.transpose(kspace, (0, 2, 1)),
        matrix_size=matrix, fov_cm=fov, config=dict(m.DEFAULTS, **config),
        metadata=None, trajectory_source=source)


def test_defaults_match_the_openrecon_label(monkeypatch):
    """The scanner JSON is generated from the label; the module must agree with it."""
    m = _import_quickgrid(monkeypatch)
    label = json.loads((RECIPE_DIR / "OpenReconLabel.json").read_text())
    params = {p["id"]: p for p in label["parameters"]}
    assert len(params) <= 14
    assert params["config"]["default"] == m.RECON_NAME == "quickgrid"
    for pid, p in params.items():
        assert pid in m.DEFAULTS, f"{pid} is in the label but not in DEFAULTS"
        default = p["default"]
        if p["type"] == "choice":
            assert default in {v["id"] for v in p["values"]}, pid
        expected = m.DEFAULTS[pid]
        assert str(default).lower() == str(expected).lower(), (pid, default, expected)
    choices = {v["id"] for v in params["trajectoryfile"]["values"]}
    assert set(m.BUNDLED_TRAJECTORIES.values()) <= choices


def test_bundled_trajectories_exist_and_have_distinct_shapes(monkeypatch):
    """Auto-selection identifies the sequence by (samples, readouts); shapes must not collide."""
    m = _import_quickgrid(monkeypatch)
    shapes = {}
    for key, container_path in m.BUNDLED_TRAJECTORIES.items():
        local = RECIPE_DIR / Path(container_path).name
        assert local.exists(), local
        with h5py.File(local, "r") as h:
            shapes[key] = h["k"].shape[:2]
            if key.startswith("cones"):
                assert "dcf" in h and h["dcf"].shape == h["k"].shape[:2], key
    assert len(set(shapes.values())) == len(shapes), shapes


def test_uniform_radial_is_detected_and_cones_is_not(monkeypatch):
    m = _import_quickgrid(monkeypatch)
    assert m._looks_uniform_radial(_radial_trajectory())
    with h5py.File(RECIPE_DIR / "cones3d_n4846_trajectory.h5", "r") as h:
        cones = h["k"][:, :200, :]
    assert not m._looks_uniform_radial(cones)


def test_dcf_prefers_file_then_analytic(monkeypatch, tmp_path):
    m = _import_quickgrid(monkeypatch)
    traj = _radial_trajectory()
    ks = np.zeros((1,) + traj.shape[:2], np.complex64)

    # analytic for uniform radial when nothing is shipped: |k|^2 with a floored centre
    recon = _recon_input(m, traj, ks, 16, dcfmode="auto")
    w = m.density_compensation(recon, recon.coords_index_units())
    assert w.shape == (traj.size // 3,) and w.max() == pytest.approx(1.0) and w.min() > 0
    assert np.argmax(w.reshape(traj.shape[:2]).mean(axis=1)) == traj.shape[0] - 1

    # a shipped dcf wins over analytic
    path = tmp_path / "t.h5"
    stored = np.linspace(0.1, 1.0, traj.size // 3).astype(np.float32).reshape(traj.shape[:2])
    with h5py.File(path, "w") as h:
        h.create_dataset("k", data=traj); h.create_dataset("dcf", data=stored)
    recon = _recon_input(m, traj, ks, 16, source=str(path), dcfmode="auto")
    assert np.allclose(m.density_compensation(recon, recon.coords_index_units()), stored.ravel())

    # dcfmode=file without a dataset is an error, not a silent fallback
    with h5py.File(path, "w") as h:
        h.create_dataset("k", data=traj)
    recon = _recon_input(m, traj, ks, 16, source=str(path), dcfmode="file")
    with pytest.raises(ValueError):
        m.density_compensation(recon, recon.coords_index_units())

    # a mismatched dcf is refused
    with h5py.File(path, "w") as h:
        h.create_dataset("k", data=traj); h.create_dataset("dcf", data=stored[:, :10])
    recon = _recon_input(m, traj, ks, 16, source=str(path), dcfmode="auto")
    with pytest.raises(ValueError):
        m.density_compensation(recon, recon.coords_index_units())


def test_reconstruct_recovers_a_phantom(monkeypatch):
    """Adjoint NUFFT of simulated multi-coil radial data must correlate with the object."""
    m = _import_quickgrid(monkeypatch)
    import sigpy
    N, fov, coils = 16, 22.0, 3
    traj = _radial_trajectory(samples=N, spokes=600, kmax=N / (2 * fov))
    g = np.stack(np.meshgrid(*[np.linspace(-1, 1, N)] * 3, indexing="ij"))
    truth = (np.linalg.norm(g, axis=0) < 0.6).astype(np.float32)
    maps = np.stack([np.exp(-(g[i] - 0.4) ** 2) for i in range(coils)]).astype(np.complex64)
    coord = (traj.reshape(-1, 3) * fov).astype(np.float32)
    y = sigpy.nufft(maps * truth[None], coord).reshape(coils, N, -1).astype(np.complex64)

    for mode in ("SoS", "AC"):
        vol = m.reconstruct(_recon_input(m, traj, y, N, fov, coilcombinemode=mode))
        assert vol.shape == (N, N, N) and vol.dtype == np.float32 and np.all(np.isfinite(vol))
        ref = np.sqrt(np.sum(np.abs(maps * truth[None]) ** 2, axis=0))
        corr = np.corrcoef(vol.ravel(), ref.ravel())[0, 1]
        assert corr > 0.85, (mode, corr)


def test_unknown_dcfmode_is_rejected(monkeypatch):
    m = _import_quickgrid(monkeypatch)
    traj = _radial_trajectory()
    recon = _recon_input(m, traj, np.zeros((1,) + traj.shape[:2], np.complex64), 16, dcfmode="bogus")
    with pytest.raises(ValueError):
        m.density_compensation(recon, recon.coords_index_units())


def test_parallel_gridding_matches_serial(monkeypatch):
    """Worker count must not change the result: same tasks, same order, float64 sums."""
    m = _import_quickgrid(monkeypatch)
    rng = np.random.default_rng(0)
    traj = _radial_trajectory(samples=12, spokes=200)
    N, coils = 16, 3
    kspace = (rng.standard_normal((coils,) + traj.shape[:2]) + 1j * rng.standard_normal((coils,) + traj.shape[:2])).astype(np.complex64)
    recon = _recon_input(m, traj, kspace, N)
    coord = recon.coords_index_units()
    w = m.density_compensation(recon, coord)
    serial_sos = m.grid(kspace, coord, w, recon.image_shape, workers=1, mode="sos")
    parallel_sos = m.grid(kspace, coord, w, recon.image_shape, workers=2, mode="sos")
    assert serial_sos.dtype == np.float64 and np.array_equal(serial_sos, parallel_sos)
    serial = m.grid(kspace, coord, w, recon.image_shape, workers=1)
    parallel = m.grid(kspace, coord, w, recon.image_shape, workers=3)
    assert serial.shape == (coils, N, N, N) and np.array_equal(serial, parallel)
    # the streaming SoS path equals SoS of the collected coil images
    assert np.allclose(np.sqrt(serial_sos), np.sqrt(np.sum(np.abs(serial).astype(np.float64) ** 2, axis=0)))
    # reconstruct() takes the same route and is unaffected by maxworkers
    recon.config["maxworkers"] = 1
    v1 = m.reconstruct(recon)
    recon.config["maxworkers"] = 3
    v3 = m.reconstruct(recon)
    assert v1.dtype == np.float32 and np.array_equal(v1, v3)


def test_worker_count_is_capped_by_coils_cpus_and_memory(monkeypatch):
    m = _import_quickgrid(monkeypatch)
    assert m.worker_count(8, 2, (16, 16, 16)) <= 2
    assert m.worker_count(0, 4, (16, 16, 16)) == 1
    monkeypatch.setattr(m, "_available_memory_bytes", lambda: 1)   # no memory: still one worker
    assert m.worker_count(8, 8, (256, 256, 256)) == 1
    monkeypatch.setattr(m, "_available_memory_bytes", lambda: 0)   # unknown: trust the request
    assert m.worker_count(2, 8, (16, 16, 16)) == 2


def test_auto_selection_also_searches_the_share_folder(monkeypatch, tmp_path):
    """A *_trajectory.h5 dropped into fire\\share (/tmp/share) is a candidate for `auto`."""
    m = _import_quickgrid(monkeypatch)
    traj = _radial_trajectory(samples=10, spokes=77)
    with h5py.File(tmp_path / "mine_trajectory.h5", "w") as h:
        h["k"] = traj
    monkeypatch.setattr(m.mrdrecon, "TRAJECTORY_SEARCH_DIRS", [str(tmp_path)])
    candidates = m.mrdrecon._trajectory_candidates()
    assert candidates["mine"] == str(tmp_path / "mine_trajectory.h5")
    assert set(m.BUNDLED_TRAJECTORIES) <= set(candidates)
    assert m.mrdrecon._trajectory_dimensions(str(tmp_path / "mine_trajectory.h5"), "k") == (10, 77)
    # shape probing without loading follows the same axis rule as the loader
    # (a different spoke count, so it cannot collide with 'mine' below)
    with h5py.File(tmp_path / "swapped_trajectory.h5", "w") as h:
        h["k"] = np.moveaxis(_radial_trajectory(samples=10, spokes=61), -1, 0)
    assert m.mrdrecon._trajectory_dimensions(str(tmp_path / "swapped_trajectory.h5"), "k") == (10, 61)
    class Acq:  # minimal stand-in for ismrmrd.Acquisition
        def __init__(self): self.data = np.zeros((1, 10), np.complex64)
    picked = m.mrdrecon._autoselect_trajectory([Acq() for _ in range(77)], "k")
    assert picked == str(tmp_path / "mine_trajectory.h5")
