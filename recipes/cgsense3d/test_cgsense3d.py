"""Focused tests for the cgsense3d app module.

Run with:  python -m pytest recipes/cgsense3d/test_*.py -q

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


def _import(monkeypatch):
    constants = types.ModuleType("constants")
    constants.MRD_LOGGING_INFO = 1
    mrdhelper = types.ModuleType("mrdhelper")
    mrdhelper.update_img_header_from_raw = lambda image_header, reference_head: image_header
    mrdhelper.get_json_config_param = lambda cfg, key, default=None, type=None: cfg.get(key, default)
    monkeypatch.syspath_prepend(str(RECIPE_DIR))
    monkeypatch.setitem(sys.modules, "constants", constants)
    monkeypatch.setitem(sys.modules, "mrdhelper", mrdhelper)
    for name in ("cgsense3d", "mrdrecon", "gradwarp"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    return importlib.import_module("cgsense3d")


def _radial_trajectory(samples=16, spokes=300, kmax=0.5):
    """Uniform centre-out 3D radial in cycles/cm, shape (samples, spokes, 3)."""
    n = np.arange(1, spokes + 1)
    z = 1 - 2 * (n - 0.5) / spokes
    r = np.sqrt(np.maximum(0.0, 1 - z * z))
    az = np.deg2rad(137.508) * n
    d = np.stack([r * np.cos(az), r * np.sin(az), z], axis=1)
    radius = (np.arange(samples) + 0.5) / samples * kmax
    return (radius[:, None, None] * d[None, :, :]).astype(np.float32)


def _same(a, b):
    """Label defaults are strings ('1e-3'); compare numerically when both parse."""
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return str(a).lower() == str(b).lower()


def _recon_input(m, traj, kspace, matrix, fov=22.0, source=None, selection=None, workers=1, **config):
    return m.mrdrecon.ReconInput(
        kspace=kspace, trajectory=traj, raw_kspace=np.transpose(kspace, (0, 2, 1)),
        matrix_size=matrix, fov_cm=fov, config=dict(m.DEFAULTS, **config),
        metadata=None, trajectory_source=source, trajectory_selection=selection, max_workers=workers)


def test_defaults_match_the_openrecon_label(monkeypatch):
    """The scanner JSON is generated from the label; the module must agree with it,
    and the label must stay within OpenRecon's 14-parameter cap."""
    m = _import(monkeypatch)
    label = json.loads((RECIPE_DIR / "OpenReconLabel.json").read_text())
    params = {p["id"]: p for p in label["parameters"]}
    assert len(params) <= 14
    assert params["config"]["default"] == m.RECON_NAME == "cgsense3d"
    for pid, p in params.items():
        assert pid in m.DEFAULTS, f"{pid} is in the label but not in DEFAULTS"
        default = p["default"]
        if p["type"] == "choice":
            assert default in {v["id"] for v in p["values"]}, pid
        assert _same(default, m.DEFAULTS[pid]), (pid, default, m.DEFAULTS[pid])
    choices = {v["id"] for v in params["trajectoryfile"]["values"]}
    assert set(m.BUNDLED_TRAJECTORIES.values()) <= choices
    assert set(m.NAMED_TRAJECTORIES.values()) <= choices
    # the framework took the app's defaults, including the ones that have no UI control
    assert m.mrdrecon.OPENRECON_DEFAULTS["orientation"] == "xyz"
    assert m.mrdrecon.OPENRECON_DEFAULTS["maxcoils"] == 0
    assert m.mrdrecon.OPENRECON_DEFAULTS["espiritcrop"] == 0.8
    assert m.gradient_unwarp in m.mrdrecon.OUTPUT_VOLUME_HOOKS
    orient = {v["id"] for v in params["orientation"]["values"]}
    assert orient - {"debug"} <= set(m.mrdrecon.ORIENTATION_TRANSFORMS) and "xyz" in orient


def test_bundled_trajectories_exist_and_have_distinct_shapes(monkeypatch):
    """Auto-selection identifies the sequence by (samples, readouts); shapes must not collide.
    The same files quickgrid bundles are expected next to this recipe (hard links)."""
    m = _import(monkeypatch)
    shapes = {}
    for key, container_path in m.BUNDLED_TRAJECTORIES.items():
        local = RECIPE_DIR / Path(container_path).name
        assert local.exists(), local
        with h5py.File(local, "r") as h:
            shapes[key] = h["k"].shape[:2]
            if key.startswith("cones"):
                assert "dcf" in h and h["dcf"].shape == h["k"].shape[:2], key
    assert len(set(shapes.values())) == len(shapes), shapes
    for key, container_path in m.NAMED_TRAJECTORIES.items():
        with h5py.File(RECIPE_DIR / Path(container_path).name, "r") as h:
            assert "acq_order" in h and h["dcf"].shape == h["k"].shape[:2], key


def test_dcf_modes_file_analytic_and_row_selection(monkeypatch, tmp_path):
    m = _import(monkeypatch)
    traj = _radial_trajectory()
    ks = np.zeros((1,) + traj.shape[:2], np.complex64)
    # analytic for uniform radial when nothing is shipped
    w = m.density_compensation(_recon_input(m, traj, ks, 16, dcfmode="auto"))
    assert w.shape == (traj.size // 3,) and w.max() == pytest.approx(1.0) and w.min() > 0
    # a shipped dcf wins, and is cut to the rows one echo uses
    stored = np.random.default_rng(1).random(traj.shape[:2]).astype(np.float32)
    path = tmp_path / "pair_trajectory.h5"
    with h5py.File(path, "w") as h:
        h.create_dataset("k", data=traj); h.create_dataset("dcf", data=stored)
    assert np.allclose(m.density_compensation(_recon_input(m, traj, ks, 16, source=str(path))), stored.ravel())
    rows = np.arange(0, traj.shape[1], 2)
    sel = dict(swapped=False, rows=rows, sample_offset=0, samples=traj.shape[0], readouts=rows.size)
    r = _recon_input(m, traj[:, rows], np.zeros((1, traj.shape[0], rows.size), np.complex64), 16,
                     source=str(path), selection=sel)
    assert np.array_equal(m.density_compensation(r).reshape(traj.shape[0], rows.size), stored[:, rows])
    # without a selection that explains the size difference it is refused
    r = _recon_input(m, traj[:, rows], np.zeros((1, traj.shape[0], rows.size), np.complex64), 16, source=str(path))
    with pytest.raises(ValueError):
        m.density_compensation(r)
    with pytest.raises(ValueError):
        m.density_compensation(_recon_input(m, traj, ks, 16, dcfmode="nonsense"))


def test_toeplitz_matches_direct_and_is_worker_independent(monkeypatch):
    m = _import(monkeypatch)
    N, coils, pts = 12, 3, 2000
    rng = np.random.default_rng(0)
    coord = rng.uniform(-N / 2, N / 2, (pts, 3)).astype(np.float32)
    w = np.abs(rng.standard_normal(pts)).astype(np.float32)
    maps = (rng.standard_normal((coils,) + (N,) * 3) + 1j * rng.standard_normal((coils,) + (N,) * 3)).astype(np.complex64)
    x = (rng.standard_normal((N,) * 3) + 1j * rng.standard_normal((N,) * 3)).astype(np.complex64)
    fast = m.build_normal_operator(maps, coord, w, (N,) * 3, 0.0, use_toeplitz=True)
    slow = m.build_normal_operator(maps, coord, w, (N,) * 3, 0.0, use_toeplitz=False)
    a, b = fast(x), slow(x)
    assert np.linalg.norm(a - b) / np.linalg.norm(b) < 5e-2
    fast3 = m.build_normal_operator(maps, coord, w, (N,) * 3, 0.0, use_toeplitz=True, max_workers=3)
    assert np.array_equal(fast3(x), a)
    y = (rng.standard_normal((N,) * 3) + 1j * rng.standard_normal((N,) * 3)).astype(np.complex64)
    assert abs(np.vdot(fast(x), y) - np.vdot(x, fast(y))) / abs(np.vdot(fast(x), y)) < 1e-3
    assert np.vdot(x, fast(x)).real > 0


def test_cgsense_recovers_an_undersampled_phantom_better_than_gridding(monkeypatch):
    """Two-fold undersampled radial, 4 coils: CG-SENSE must beat the density-compensated
    adjoint on the same data (that is the whole point of the app)."""
    m = _import(monkeypatch)
    import sigpy
    N, fov, coils = 16, 22.0, 4
    traj = _radial_trajectory(samples=N, spokes=250, kmax=N / (2 * fov))   # ~1/3 of 3D radial Nyquist
    g = np.stack(np.meshgrid(*[np.linspace(-1, 1, N)] * 3, indexing="ij"))
    r = np.linalg.norm(g, axis=0)
    truth = ((r < 0.7) * (1 + 0.5 * (np.abs(g[0]) < 0.25))).astype(np.float32)
    maps = np.stack([np.exp(-((g[i % 3] - (0.6 if i < 2 else -0.6)) ** 2)) for i in range(coils)]).astype(np.complex64)
    coord = (traj.reshape(-1, 3) * fov).astype(np.float32)
    y = sigpy.nufft(maps * truth[None], coord).reshape(coils, N, -1).astype(np.complex64)
    recon = _recon_input(m, traj, y, N, fov, cgiterations=10, espiritmatrix=16, workers=2)
    vol = m.reconstruct(recon)
    assert vol.shape == (N, N, N) and vol.dtype == np.float32 and np.isfinite(vol).all()
    w = m.density_compensation(recon)
    grid = np.sqrt(np.sum(np.abs(sigpy.nufft_adjoint(y.reshape(coils, -1) * w, coord, (coils, N, N, N))) ** 2, axis=0))
    sos_truth = np.sqrt(np.sum(np.abs(maps * truth[None]) ** 2, axis=0))
    c_cg = np.corrcoef(vol.ravel(), truth.ravel())[0, 1]
    c_grid = np.corrcoef(grid.ravel(), sos_truth.ravel())[0, 1]
    assert c_cg > 0.9 and c_cg > c_grid, (c_cg, c_grid)


def test_gradient_unwarp_hook_skips_cleanly_without_a_coefficient_file(monkeypatch, tmp_path):
    m = _import(monkeypatch)
    monkeypatch.setattr(m.gradwarp, "COEFF_SEARCH_DIRS", [str(tmp_path)])
    vol = np.random.default_rng(0).random((6, 6, 6)).astype(np.float32)
    ctx = dict(params={"gradunwarp": "3D", "gradcoeffile": "auto"}, metadata=None, reference_head=None,
               output_fov_mm=220.0, orientation="xyz", flip_slice=False)
    assert np.array_equal(m.gradient_unwarp(vol, ctx), vol)
    ctx["params"]["gradunwarp"] = "off"
    assert np.array_equal(m.gradient_unwarp(vol, ctx), vol)
