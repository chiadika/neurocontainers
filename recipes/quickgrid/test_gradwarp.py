"""Tests for quickgrid's gradient-nonlinearity correction.

The displacement field and Jacobian are checked against the reference
implementation (the `gradunwarp` package, MIT) on a synthetic coefficient
file, so the port cannot drift from it. The resampling is checked for the
properties that matter: linear-only coefficients leave a volume untouched,
and a known deformation is undone. No proprietary coefficient file is used.

Run with:  python -m pytest recipes/quickgrid/test_gradwarp.py -q
"""

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

RECIPE_DIR = Path(__file__).resolve().parent


def _module(monkeypatch):
    monkeypatch.syspath_prepend(str(RECIPE_DIR))
    monkeypatch.delitem(sys.modules, "gradwarp", raising=False)
    return importlib.import_module("gradwarp")


def _synthetic_grad(path, R0=0.25, with_linear=False):
    """A plausible-looking Siemens .grad file with made-up, small coefficients."""
    rng = np.random.default_rng(7)
    lines = ["#*[ Synthetic test coefficients ]*", f"  {R0:.3f} m = R0", " "]
    k = 1
    for ax in "xyz":
        for n in range(1, 8):
            for m in range(0, n + 1):
                if n == 1 and not with_linear:
                    continue
                for ab in ("A", "B"):
                    if ab == "B" and m == 0:
                        continue
                    v = 0.02 * rng.standard_normal() / (n * n)
                    lines.append(f"{k:3d} {ab}({n:2d},{m:2d}) {v:12.7f} {ax}")
                    k += 1
    path.write_text("\n".join(lines) + "\n")
    return path


def test_parser_and_field_match_reference_package(monkeypatch, tmp_path):
    gw = _module(monkeypatch)
    ref = pytest.importorskip("gradunwarp.core.unwarp_resample")
    ref_coeffs = pytest.importorskip("gradunwarp.core.coeffs")
    gfile = _synthetic_grad(tmp_path / "coeff_TEST.grad")
    c = gw.read_siemens_grad(str(gfile))
    assert c.R0_m == 0.25 and c.order == 7
    rc = ref_coeffs.get_siemens_grad(str(gfile))
    for ax in "xyz":
        assert np.allclose(c.alpha[ax], getattr(rc, f"alpha_{ax}")[: c.order + 1, : c.order + 1])
        assert np.allclose(c.beta[ax], getattr(rc, f"beta_{ax}")[: c.order + 1, : c.order + 1])
    rng = np.random.default_rng(1)
    pts = rng.uniform(-120, 120, size=(500, 3))                # mm, LAI frame
    dx, dy, dz = gw.displacement_lai(c, pts[:, 0], pts[:, 1], pts[:, 2])
    r, ct, th, ph = ref.cart2sph(pts[:, 0], pts[:, 1], pts[:, 2])
    R0 = c.R0_m * 1000.0
    rdx = R0 * ref.siemens_B(rc.alpha_x, rc.beta_x, r, ct, th, ph, R0)
    rdy = R0 * ref.siemens_B(rc.alpha_y, rc.beta_y, r, ct, th, ph, R0)
    rdz = R0 * ref.siemens_B(rc.alpha_z, rc.beta_z, r, ct, th, ph, R0)
    assert np.allclose(dx, rdx, atol=1e-9) and np.allclose(dy, rdy, atol=1e-9) and np.allclose(dz, rdz, atol=1e-9)
    assert float(np.abs(np.stack([dx, dy, dz])).max()) > 0.1     # the synthetic field is not trivial


def test_lps_frame_conversion(monkeypatch, tmp_path):
    gw = _module(monkeypatch)
    c = gw.read_siemens_grad(str(_synthetic_grad(tmp_path / "coeff_TEST.grad")))
    p = np.array([[30.0, -40.0, 55.0]])
    d = gw.displacement_lps(c, p)[0]
    dx, dy, dz = gw.displacement_lai(c, p[:, 0], -p[:, 1], -p[:, 2])
    assert np.allclose(d, [dx[0], -dy[0], -dz[0]])


def test_linear_only_coefficients_change_nothing(monkeypatch, tmp_path):
    gw = _module(monkeypatch)
    gfile = tmp_path / "coeff_LIN.grad"
    gfile.write_text("#\n 0.250 m = R0\n  1 A( 1, 1)  0.0000000 x\n  2 A( 1, 0)  0.0000000 z\n  3 B( 1, 1)  0.0000000 y\n")
    c = gw.read_siemens_grad(str(gfile))
    rng = np.random.default_rng(3)
    vol = rng.random((24, 20, 22)).astype(np.float32)
    out = gw.unwarp_volume(vol, c, center_lps=[0, 0, 0], axis_dirs_lps=np.eye(3),
                           spacing_mm=[2.0, 2.0, 2.0], jacobian=True)
    assert out.shape == vol.shape and np.allclose(out, vol, atol=1e-5)


def test_unwarp_undoes_a_known_warp(monkeypatch, tmp_path):
    """Warp a smooth phantom with the field (p -> p - dv), then unwarp it back."""
    gw = _module(monkeypatch)
    import scipy.ndimage
    c = gw.read_siemens_grad(str(_synthetic_grad(tmp_path / "coeff_TEST.grad")))
    n = 48
    sp = 4.0                                                      # 192 mm cube
    g = (np.arange(n) - (n - 1) / 2) * sp
    S, R, C = np.meshgrid(g, g, g, indexing="ij")
    truth = np.exp(-((S / 60) ** 2 + (R / 50) ** 2 + (C / 70) ** 2)).astype(np.float32)
    truth *= (1 + 0.3 * np.cos(S / 15) * np.cos(C / 12)).astype(np.float32)
    dirs = np.eye(3)
    pos = np.stack([S, R, C], axis=-1)                            # axes aligned with LPS here
    dv = gw.displacement_lps(c, pos)                              # mm
    assert 0.5 < float(np.abs(dv).max()) < 20.0, float(np.abs(dv).max())
    # acquired(p') = truth(p) where p' = p + dv(p)  ->  acquired(q) ~ truth(q - dv(q)) to first order
    idx = np.stack([S, R, C]) / sp + (n - 1) / 2
    warped_sample = idx - np.moveaxis(dv, -1, 0) / sp
    acquired = scipy.ndimage.map_coordinates(truth, warped_sample, order=1, mode="constant")
    corrected = gw.unwarp_volume(acquired, c, center_lps=[0, 0, 0], axis_dirs_lps=dirs,
                                 spacing_mm=[sp] * 3, jacobian=False, coarse_step_mm=8.0)
    inner = (slice(6, -6),) * 3
    err_before = np.abs(acquired - truth)[inner].mean()
    err_after = np.abs(corrected - truth)[inner].mean()
    assert err_after < 0.35 * err_before, (err_before, err_after)


def test_2d_mode_leaves_through_plane_untouched(monkeypatch, tmp_path):
    gw = _module(monkeypatch)
    c = gw.read_siemens_grad(str(_synthetic_grad(tmp_path / "coeff_TEST.grad")))
    n = 32
    vol = np.zeros((n, n, n), np.float32)
    vol[n // 2] = 1.0                                             # a single bright slice
    out = gw.unwarp_volume(vol, c, center_lps=[0, 0, 0], axis_dirs_lps=np.eye(3),
                           spacing_mm=[4.0] * 3, mode="2D", jacobian=False)
    assert out[n // 2].sum() > 0.9 * vol.sum() and out.sum() == pytest.approx(out[n // 2].sum(), rel=1e-6)


def test_coefficient_file_lookup(monkeypatch, tmp_path):
    gw = _module(monkeypatch)
    (tmp_path / "coeff_OTHER.grad").write_text("x")
    (tmp_path / "coeff_IMPULSE.grad").write_text("x")
    monkeypatch.setattr(gw, "COEFF_SEARCH_DIRS", [str(tmp_path)])
    assert gw.find_coefficient_file("auto", "IMPULSE").endswith("coeff_IMPULSE.grad")
    assert gw.find_coefficient_file("/explicit/file.grad") == "/explicit/file.grad"
    monkeypatch.setattr(gw, "COEFF_SEARCH_DIRS", [str(tmp_path / "nothing")])
    assert gw.find_coefficient_file("auto") is None
