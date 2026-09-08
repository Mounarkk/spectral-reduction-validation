"""
Reduced fluorescence validation in the 4-channel XYZU basis (300-799 nm).

Five colour-transport pipelines are compared against a full spectral
reference over K bounces, for sRGB-authored materials under six CIE
illuminants (D65, D50, HP1, E, A, LED-B1):

  Spectral ref   : per-wavelength transport P = diag(R) + F*diag(1-R)
                   (500x500); projected to XYZU only for display.
  BF-reduced     : 4x4 operator with Fbar = S4^T F Sb4 ; brute-force reduction
                   of the true spectral F.
  Analytic-red.  : 4x4 operator with Fbar from the closed-form Gaussian model
                   (Belcour et al. 2025, Eq. 15) ; what a shader evaluates.
  Hybrid         : diagonal reflectance on the first bounce, analytic-reduced
                   operator afterwards.
  Naive          : component-wise power of the XYZU albedo.
  R-only         : spectral transport with fluorescence removed ; a baseline
                   that measures how much of the signal *is* fluorescence.

Ground-truth albedo spectra come from Jakob & Hanika (2019) sigmoid
upsampling (data/srgb.coeff) evaluated on 300-799 nm; below 380 nm this is a
smooth extrapolation of the visible-range fit.  The U-channel albedo is
computed two ways and both are run:
  Method 1 : rho_U = R_jh * S4[:,3]              (spectral integral)
  Method 2 : rho_U = clip(T_U * xyz_alb, 0, 1)   (linear projection, Eq. 31)

Outputs (results/fluorescence/):
  method{1,2}/<material>.png                  swatches + DeltaE2000-vs-bounce
  method{1,2}/<material>_diag_D65.png         2x3 diagnostics
  method{1,2}/<mat>_param_sweep_D65_M*.png    DeltaE heatmap over (mu_a, mu_e)
  method{1,2}/<mat>_sigma_sweep_D65_M*.png    DeltaE curves over sigma_a, sigma_e
  u_channel_comparison.png                    rho_U, Method 1 vs Method 2
  uv_test_M{1,2}/                             D65 vs UV-Gaussian illuminant

Paper: Belcour, Fichet, Barla.
       "A Fluorescent Material Model for Non-Spectral Editing & Rendering"
       SIGGRAPH 2025.  hal-05267431
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import colour
import os
import struct
from pathlib import Path
from scipy.special import erf

from _paths import CMF, DATA, RESULTS

matplotlib.rcParams["font.family"] = "monospace"

# =============================================================================
# 0. Wavelength grid  (300–799 nm, 500 points)
# =============================================================================
WL   = np.arange(300, 800, 1)      # 500 points, 1 nm steps
N_WL = len(WL)

# =============================================================================
# 1. XYZU CMF  —  4-channel (X, Y, Z, U=UV Gaussian), col-sum normalised
#    xyzu.csv: columns = [wavelength, x-bar, y-bar, z-bar, u-bar], 300–799 nm
# =============================================================================
_xyzu_data  = np.loadtxt(CMF / "xyzu.csv", delimiter=",")
assert _xyzu_data.shape == (500, 5), f"Unexpected xyzu.csv shape {_xyzu_data.shape}"
_xyzu_raw   = _xyzu_data[:, 1:5]           # (500, 4)

_col_sums4  = _xyzu_raw.sum(axis=0)        # (4,)  column sums of raw CMF
S4          = _xyzu_raw / _col_sums4        # (500, 4)  normalised 4-channel basis
Sb4         = S4 @ np.linalg.pinv(S4.T @ S4)   # (500, 4) dual basis  S4.T@Sb4=I4

# 3-channel sub-basis  (X, Y, Z only — used for upsampling and Method-2 T_U)
S3          = S4[:, :3]                     # (500, 3)
Sb3         = S3 @ np.linalg.pinv(S3.T @ S3)   # (500, 3)

# Method-2 transfer weights  T_U = S4[:,3]^T @ Sb3   (paper Eq. 31)
# Given xyz_alb (normalised XYZ), rho_U ≈ clip(T_U @ xyz_alb, 0, 1)
T_U         = S4[:, 3] @ Sb3               # (3,)

# Reduced reflectance tensor  Rk4[..., k] = S4.T @ diag(Sb4[:,k]) @ Sb4
Rk4         = np.stack(
    [S4.T @ np.diag(Sb4[:, k]) @ Sb4 for k in range(4)], axis=-1
)   # (4, 4, 4)

_col_sums3  = _xyzu_raw[:, :3].sum(axis=0) # (3,) xyz column sums on 300–799 grid

print(f"S4 col sums        : {S4.sum(axis=0).round(6)}")
print(f"S4.T @ Sb4 = I4 ?  : max off-diag = {np.abs(S4.T @ Sb4 - np.eye(4)).max():.2e}")
print(f"T_U weights (M2)   : {T_U.round(6)}")

# =============================================================================
# 2. Normalisation matrices
#    Normalised XYZ (col-sum) ↔ Standard CIE XYZ
#    Normalised XYZ[j] = Standard XYZ[j] × col_sum_Y / col_sum_j
# =============================================================================
_col_y      = _col_sums3[1]                     # col-sum of normalised y-bar
NORM_TO_STD = np.diag(_col_sums3 / _col_y)     # (3,3) normalised → standard XYZ
STD_TO_NORM = np.diag(_col_y / _col_sums3)     # (3,3) standard → normalised XYZ

# =============================================================================
# 3. sRGB conversions
# =============================================================================
XYZ_TO_LIN = np.array([
    [ 3.2404542, -1.5371385, -0.4985314],
    [-0.9692660,  1.8760108,  0.0415560],
    [ 0.0556434, -0.2040259,  1.0572252],
])
LIN_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])
_LIN_TO_NORM = STD_TO_NORM @ LIN_TO_XYZ        # linear sRGB → normalised XYZ
_NORM_TO_LIN = XYZ_TO_LIN  @ NORM_TO_STD       # normalised XYZ → linear sRGB

def srgb_to_linear(srgb):
    c = np.clip(np.asarray(srgb, dtype=float), 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

def linear_to_srgb(lin):
    c = np.clip(lin, 0.0, None)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1.0 / 2.4) - 0.055)

def srgb_to_norm_xyz(srgb):
    """sRGB [0,1] → normalised 3-channel XYZ (col-sum, 300–799 grid)."""
    return _LIN_TO_NORM @ srgb_to_linear(srgb)

def norm_xyz_to_lin(xyz_norm):
    """Normalised 3-channel XYZ → linear sRGB (unclamped)."""
    return _NORM_TO_LIN @ xyz_norm

def norm_xyz_to_srgb(xyz_norm):
    return linear_to_srgb(norm_xyz_to_lin(xyz_norm))

def xyzu_to_lin(xyzu4):
    """4-channel XYZU → linear sRGB (drop U, convert XYZ only)."""
    return norm_xyz_to_lin(xyzu4[:3])

def xyzu_to_srgb(xyzu4):
    return linear_to_srgb(xyzu_to_lin(xyzu4))

# =============================================================================
# 4. ΔE 2000
# =============================================================================
_D65_WHITE = np.array([0.95047, 1.00000, 1.08883])

def delta_e_2000(lin_a, lin_b):
    """CIE DE2000 between two linear-sRGB triples."""
    xyz_a = LIN_TO_XYZ @ np.clip(lin_a, 0, None)
    xyz_b = LIN_TO_XYZ @ np.clip(lin_b, 0, None)
    lab_a = colour.XYZ_to_Lab(xyz_a, illuminant=_D65_WHITE)
    lab_b = colour.XYZ_to_Lab(xyz_b, illuminant=_D65_WHITE)
    return float(colour.delta_E(lab_a, lab_b, method="CIE 2000"))

# =============================================================================
# 5. Illuminant helpers  —  300–799 nm
# =============================================================================
def load_illuminant(name: str) -> np.ndarray:
    """Load CIE illuminant aligned to 300–799 nm (500 points)."""
    ill = colour.SDS_ILLUMINANTS[name]
    return ill.copy().align(colour.SpectralShape(300, 799, 1)).values

def normalize_illuminant(I_raw: np.ndarray) -> np.ndarray:
    """Scale so that a perfect white reflector gives Y_norm = 1."""
    return I_raw / float(I_raw @ S4[:, 1])

def gaussian_illuminant(mu_i: float, sigma_i: float) -> np.ndarray:
    """Gaussian spectral illuminant, normalised to Y = 1 (for UV testing)."""
    G_ill = np.exp(-0.5 * ((WL - mu_i) / sigma_i) ** 2)
    return G_ill / float(G_ill @ S4[:, 1])

# =============================================================================
# 6. Gaussian CMF basis  —  5 Gaussians fitted to 4-channel S4 on 300–799 nm
#    G  : (500, 5)   — 5 discretised 1D Gaussians (paper Table 1)
#    T_G: (5,   4)   — transfer matrix  S4 ≈ G @ T_G  (via lstsq)
#    C4 : (4,   4)   — correction matrix  C = pinv(S4_gauss.T @ S4_gauss)
#
#    The 5th Gaussian (UV, mu=382.5, sigma=57.4) captures the U channel;
#    with the wider 300–799 nm grid it has full support in the UV region.
# =============================================================================
_aG = np.array([0.35087, 1.141263, 1.024335, 1.915863, 1.0])
_mG = np.array([443.412226, 596.813847, 560.186336, 447.268188, 382.535501])
_sG = np.array([20.838149, 33.276659, 43.898132, 23.542626, 57.432550])

def _g1d(wl, a, m, s):
    return a * np.exp(-0.5 * ((wl - m) / s) ** 2)

G        = np.column_stack([_g1d(WL, _aG[i], _mG[i], _sG[i]) for i in range(5)])  # (500, 5)
T_G, _, _, _ = np.linalg.lstsq(G, S4, rcond=None)      # (5, 4)
S4_gauss = G @ T_G                                       # (500, 4) Gaussian approx of S4
C4       = np.linalg.pinv(S4_gauss.T @ S4_gauss)        # (4, 4) correction matrix

print(f"Gaussian fit max error |S4 - G@T_G| : {np.abs(S4 - S4_gauss).max():.6f}")

# =============================================================================
# 7. Analytical reduced Fbar (4×4)  —  Eq. 15 of Belcour et al. 2025
#    Fbar_an = T_G.T @ F_circ @ T_G @ C4
#    T_G is (5,4), C4 is (4,4) → result is (4,4)
# =============================================================================
def _gprod(a1, m1, s1, a2, m2, s2):
    """Product of two 1D Gaussians (amplitude, mean, std)."""
    s1s, s2s = s1 ** 2, s2 ** 2
    d = s1s + s2s
    return (a1 * a2 * np.exp(-0.5 * (m1 - m2) ** 2 / d),
            (m1 * s2s + m2 * s1s) / d,
            s1 * s2 / np.sqrt(d))

def _Fcirc_jk(alpha, mu_a, sa, mu_e, se, aK, mK, sK, aJ, mJ, sJ):
    """One (j,k) entry of the 5×5 F_circ matrix (Eq. 28)."""
    aa, ma, ssa = _gprod(1.0, mu_a, sa, aK, mK, sK)
    ae, me, sse = _gprod(1.0, mu_e, se, aJ, mJ, sJ)
    return (np.pi * alpha * aa * ae * ssa * sse
            * (1.0 - erf((ma - me) / np.sqrt(2.0 * (ssa ** 2 + sse ** 2)))))

def _alpha_max(mu_e, se):
    """Conservative upper bound on alpha for energy conservation (Eq. 33)."""
    return 1.0 / (np.sqrt(np.pi / 2.0) * se * (1.0 + erf(mu_e / (np.sqrt(2.0) * se))))

def build_F_bar_analytic(alpha_bar, mu_a, sa, mu_e, se):
    """4×4 reduced Fbar via analytical Gaussian integration (Eq. 15)."""
    alpha = alpha_bar * _alpha_max(mu_e, se)
    M     = len(_aG)
    Fc    = np.zeros((M, M))
    for j in range(M):
        for k in range(M):
            Fc[j, k] = _Fcirc_jk(alpha, mu_a, sa, mu_e, se,
                                   _aG[k], _mG[k], _sG[k],
                                   _aG[j], _mG[j], _sG[j])
    return T_G.T @ Fc @ T_G @ C4   # (4, 4)

# =============================================================================
# 8. Spectral fluorescence  (brute-force reference, 500×500)
# =============================================================================
def build_F_bar_spectral(alpha_bar, mu_a, sa, mu_e, se):
    """500×500 normalised spectral fluorescence matrix (heaviside masked)."""
    li  = WL[np.newaxis, :]
    lo  = WL[:, np.newaxis]
    Ga  = np.exp(-0.5 * ((li - mu_a) / sa) ** 2)
    Ge  = np.exp(-0.5 * ((lo - mu_e) / se) ** 2)
    F   = Ga * Ge * (lo > li).astype(float)
    cs  = F.sum(axis=0)
    amax = 1.0 / cs.max() if cs.max() > 1e-10 else 1.0
    return alpha_bar * amax * F     # (500, 500)

# =============================================================================
# 9. Jakob & Hanika RGB2Spec model
# =============================================================================
RGB2SPEC_N_COEFFS = 3

class RGB2SpecModel:
    """Load and evaluate Jakob & Hanika's RGB→Spectrum sigmoid model."""

    def __init__(self, filename):
        self._load(filename)

    def _load(self, filename):
        with open(filename, "rb") as f:
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            f.seek(0)
            if f.read(4) != b"SPEC":
                raise ValueError("Invalid RGB2Spec header")
            self.res   = struct.unpack("I", f.read(4))[0]
            self.scale = np.frombuffer(f.read(self.res * 4), dtype=np.float32)
            remaining  = file_size - f.tell()
            data_raw   = np.frombuffer(f.read(remaining), dtype=np.float32)
            expected   = 3 * self.res ** 3 * RGB2SPEC_N_COEFFS
            if data_raw.shape[0] != expected:
                raise ValueError(f"RGB2Spec data mismatch: {data_raw.shape[0]} vs {expected}")
            self.data  = data_raw.reshape(3, self.res, self.res, self.res, RGB2SPEC_N_COEFFS)
        print(f"  RGB2Spec: res={self.res}, scale=[{self.scale.min():.4f}, {self.scale.max():.4f}]")

    def fetch_coefficients(self, rgb_linear):
        rgb = np.clip(rgb_linear, 0.0, 1.0)
        i   = np.argmax(rgb)
        z   = rgb[i]
        if z == 0:
            return np.zeros(RGB2SPEC_N_COEFFS)
        res   = self.res
        scale = (res - 1) / z
        x, y  = rgb[(i + 1) % 3] * scale, rgb[(i + 2) % 3] * scale
        xi    = int(min(np.floor(x), res - 2))
        yi    = int(min(np.floor(y), res - 2))
        zi    = int(np.clip(np.searchsorted(self.scale, z) - 1, 0, res - 2))
        x1, x0 = x - xi, 1.0 - (x - xi)
        y1, y0 = y - yi, 1.0 - (y - yi)
        s0, s1  = self.scale[zi], self.scale[zi + 1]
        z1 = 0.0 if s1 == s0 else (z - s0) / (s1 - s0)
        z0 = 1.0 - z1
        coeffs = np.zeros(RGB2SPEC_N_COEFFS)
        for j in range(RGB2SPEC_N_COEFFS):
            c000 = self.data[i, zi,     yi,     xi,     j]
            c100 = self.data[i, zi,     yi,     xi + 1, j]
            c010 = self.data[i, zi,     yi + 1, xi,     j]
            c110 = self.data[i, zi,     yi + 1, xi + 1, j]
            c001 = self.data[i, zi + 1, yi,     xi,     j]
            c101 = self.data[i, zi + 1, yi,     xi + 1, j]
            c011 = self.data[i, zi + 1, yi + 1, xi,     j]
            c111 = self.data[i, zi + 1, yi + 1, xi + 1, j]
            c0   = (c000 * x0 + c100 * x1) * y0 + (c010 * x0 + c110 * x1) * y1
            c1   = (c001 * x0 + c101 * x1) * y0 + (c011 * x0 + c111 * x1) * y1
            coeffs[j] = c0 * z0 + c1 * z1
        return coeffs

    def evaluate_spectrum(self, rgb_linear, wavelengths=None):
        """Evaluate reflectance spectrum via sigmoid model.

        Wavelengths in [300, 380) are a smooth extrapolation of the trained
        sigmoid — physically plausible UV tail, not physically measured.
        """
        if wavelengths is None:
            wavelengths = WL
        coeffs = self.fetch_coefficients(rgb_linear)
        x = coeffs[0] * wavelengths ** 2 + coeffs[1] * wavelengths + coeffs[2]
        return 0.5 + 0.5 * x / np.sqrt(1.0 + x ** 2)


# =============================================================================
# 10. Helper: build all matrices for one material × method
# =============================================================================
def _build_material(srgb_in, jakob_model, method,
                    alpha_bar, mu_a, sigma_a, mu_e, sigma_e):
    """Compute J&H spectrum, 4D albedo, spectral and reduced matrices."""
    lin_srgb = srgb_to_linear(srgb_in)

    # J&H reflectance spectrum on 300–799 nm
    R_jh     = jakob_model.evaluate_spectrum(lin_srgb, wavelengths=WL)   # (500,)

    # 3-channel albedo from J&H spectrum
    xyz_alb  = R_jh @ S3           # (3,) normalised XYZ albedo

    # U-channel albedo
    if method == 1:
        u_alb = float(R_jh @ S4[:, 3])                     # spectral integral
    else:
        u_alb = float(np.clip(T_U @ xyz_alb, 0.0, 1.0))   # linear projection Eq.31

    rho_xyzu = np.append(xyz_alb, u_alb)           # (4,) 4D albedo

    # 4×4 reduced reflectance
    R_red4   = np.einsum("ijk,k->ij", Rk4, rho_xyzu)   # (4, 4)

    # Spectral matrices  (500×500)
    Fbar_sp  = build_F_bar_spectral(alpha_bar, mu_a, sigma_a, mu_e, sigma_e)
    P_sp     = np.diag(R_jh) + Fbar_sp @ np.diag(1.0 - R_jh)

    # Reduced matrices  (4×4)
    Fbar_bf4 = S4.T @ Fbar_sp @ Sb4                        # brute-force
    Fbar_an4 = build_F_bar_analytic(alpha_bar, mu_a, sigma_a, mu_e, sigma_e)
    P_bf4    = R_red4 + Fbar_bf4 @ (np.eye(4) - R_red4)
    P_an4    = R_red4 + Fbar_an4 @ (np.eye(4) - R_red4)

    # Hybrid first-bounce diagonal
    D4   = np.diag(rho_xyzu)
    H4   = D4 + Fbar_an4 @ (np.eye(4) - D4)

    return (R_jh, xyz_alb, u_alb, rho_xyzu, R_red4,
            P_sp, Fbar_bf4, Fbar_an4, P_bf4, P_an4, H4)


# =============================================================================
# 11. Main validation figure
# =============================================================================
def validate_fluorescence_xyzu(
    name, srgb_in, jakob_model,
    method=1,
    illuminant_names=None, K=20,
    output_dir=RESULTS / "fluorescence", show_plot=False,
    alpha_bar=0.8, mu_a=430.0, mu_e=560.0, sigma_a=40.0, sigma_e=40.0,
):
    """
    6-column swatch figure for 4-channel XYZU fluorescence pipeline.

    Columns: Illuminant | BF-reduced | Analytic-red. | Hybrid | Naive | ΔE2000
    """
    if illuminant_names is None:
        illuminant_names = ["D65", "D50", "HP1", "E", "A", "LED-B1"]

    (R_jh, xyz_alb, u_alb, rho_xyzu, R_red4,
     P_sp, Fbar_bf4, Fbar_an4, P_bf4, P_an4, H4) = _build_material(
        srgb_in, jakob_model, method, alpha_bar, mu_a, sigma_a, mu_e, sigma_e)

    print(f"\n{'='*60}")
    print(f"  Material   : {name}  [Method {method}]")
    print(f"  sRGB input : {srgb_in.round(3)}")
    print(f"  xyz_alb    : {xyz_alb.round(5)}")
    print(f"  u_alb      : {u_alb:.5f}  (rho_U method {method})")
    print(f"  ||Fbar_bf4 - Fbar_an4|| = {np.linalg.norm(Fbar_bf4 - Fbar_an4):.6f}")

    n_rows = len(illuminant_names)
    fig, axes = plt.subplots(
        n_rows, 6, figsize=(30, 3.5 * n_rows),
        gridspec_kw={"width_ratios": [0.55, 1.2, 1.2, 1.2, 1.2, 2.0]},
    )
    fig.patch.set_facecolor("#0f0f1e")
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    bw, sw_h, half_h = 8, 80, 40
    col_titles = [
        "Illuminant",
        "BF-reduced\nvs Spec-ref",
        "Analytic-red.\nvs Spec-ref",
        "Hybrid\nvs Spec-ref",
        "Naive\nvs Spec-ref",
        "ΔE2000 vs bounce",
    ]

    def _swatch(ref_d, mdl_d):
        sw = np.zeros((sw_h, K * bw, 3))
        for n in range(K):
            x0, x1 = n * bw, (n + 1) * bw
            sw[:half_h, x0:x1]  = np.clip(ref_d[n], 0, 1)
            sw[half_h:,  x0:x1] = np.clip(mdl_d[n], 0, 1)
        return sw

    for row, ill_name in enumerate(illuminant_names):
        I_n  = normalize_illuminant(load_illuminant(ill_name))
        c_I4 = I_n @ S4     # (4,) 4-channel illuminant colour

        # ── 1. Spectral reference ──────────────────────────────────────────
        ref_lin, ref_disp = [], []
        L = I_n.copy()
        for _ in range(K):
            L   = P_sp @ L
            lin = xyzu_to_lin(L @ S4)
            ref_lin.append(lin);  ref_disp.append(linear_to_srgb(lin))

        # ── 2. BF-reduced ─────────────────────────────────────────────────
        bf_lin, bf_disp = [], []
        M_bf = P_bf4.copy()
        for _ in range(K):
            lin = xyzu_to_lin(M_bf @ c_I4)
            bf_lin.append(lin);  bf_disp.append(linear_to_srgb(lin))
            M_bf = M_bf @ P_bf4

        # ── 3. Analytic-reduced ───────────────────────────────────────────
        an_lin, an_disp = [], []
        M_an = P_an4.copy()
        for _ in range(K):
            lin = xyzu_to_lin(M_an @ c_I4)
            an_lin.append(lin);  an_disp.append(linear_to_srgb(lin))
            M_an = M_an @ P_an4

        # ── 4. Hybrid ─────────────────────────────────────────────────────
        hyb_lin, hyb_disp = [], []
        lin = xyzu_to_lin(H4 @ c_I4)
        hyb_lin.append(lin);  hyb_disp.append(linear_to_srgb(lin))
        Ppow = P_an4.copy()
        for _ in range(1, K):
            lin = xyzu_to_lin(Ppow @ H4 @ c_I4)
            hyb_lin.append(lin);  hyb_disp.append(linear_to_srgb(lin))
            Ppow = Ppow @ P_an4

        # ── 5. Naive (component-wise in XYZU, display drops U) ────────────
        nai_lin, nai_disp = [], []
        for n in range(1, K + 1):
            lin = xyzu_to_lin((rho_xyzu ** n) * c_I4)
            nai_lin.append(lin);  nai_disp.append(linear_to_srgb(lin))

        # ── R-only spectral reference (no fluorescence) ───────────────────
        ronly_lin = []
        Lr = I_n.copy()
        for _ in range(K):
            Lr = np.diag(R_jh) @ Lr
            ronly_lin.append(xyzu_to_lin(Lr @ S4))

        # ── Col 0: illuminant swatch ──────────────────────────────────────
        ax = axes[row, 0]
        ax.imshow(np.ones((sw_h, sw_h, 3)) * np.clip(xyzu_to_srgb(c_I4), 0, 1),
                  aspect="auto")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor("#0f0f1e")
        for sp in ax.spines.values():
            sp.set_edgecolor("#555577"); sp.set_linewidth(1)
        ax.text(0.5, 0.5, ill_name, transform=ax.transAxes, ha="center",
                va="center", fontsize=10, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.2", fc="#0f0f1e", alpha=0.7))
        if row == 0:
            ax.set_title(col_titles[0], color="white", fontsize=7.5,
                         fontweight="bold", pad=5)

        # ── Cols 1–4: model swatches ──────────────────────────────────────
        for ref_d, mdl_d, col in [
            (ref_disp, bf_disp,  1),
            (ref_disp, an_disp,  2),
            (ref_disp, hyb_disp, 3),
            (ref_disp, nai_disp, 4),
        ]:
            ax = axes[row, col]
            ax.imshow(_swatch(ref_d, mdl_d), aspect="auto")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_facecolor("#0f0f1e")
            for sp in ax.spines.values():
                sp.set_edgecolor("#555577"); sp.set_linewidth(1)
            if row == 0:
                ax.set_title(col_titles[col], color="white", fontsize=7.5,
                             fontweight="bold", pad=5)
                for n in range(K):
                    ax.text((n + 0.5) / K, -0.06, str(n + 1),
                            transform=ax.transAxes, ha="center", va="top",
                            fontsize=4, color="#8888aa")
            ax.text(-0.02, 0.75, "ref", transform=ax.transAxes, rotation=90,
                    ha="right", va="center", fontsize=5.5, color="#88aaff")
            ax.text(-0.02, 0.25, "mdl", transform=ax.transAxes, rotation=90,
                    ha="right", va="center", fontsize=5.5, color="#ff8888")

        # ── Col 5: ΔE2000 plot ────────────────────────────────────────────
        ax = axes[row, 5]
        ax.set_facecolor("#080818")
        bn = list(range(1, K + 1))
        de_bf    = [delta_e_2000(bf_lin[i],    ref_lin[i]) for i in range(K)]
        de_an    = [delta_e_2000(an_lin[i],    ref_lin[i]) for i in range(K)]
        de_hyb   = [delta_e_2000(hyb_lin[i],   ref_lin[i]) for i in range(K)]
        de_nai   = [delta_e_2000(nai_lin[i],   ref_lin[i]) for i in range(K)]
        de_ronly = [delta_e_2000(ronly_lin[i],  ref_lin[i]) for i in range(K)]
        mx = max(max(de_bf), max(de_an), max(de_hyb), max(de_nai), max(de_ronly), 0.1)
        ax.set_xlim(1, K); ax.set_ylim(0, mx * 1.1)
        ax.tick_params(colors="white", labelsize=6)
        ax.set_xlabel("bounce", color="white", fontsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor("#555577")
        ax.plot(bn, de_bf,    color="#4cc9f0", lw=1.8,          label="BF-reduced")
        ax.plot(bn, de_an,    color="#f72585", lw=1.8,          label="Analytic-red.")
        ax.plot(bn, de_hyb,   color="#ffaa00", lw=1.5,          label="Hybrid")
        ax.plot(bn, de_nai,   color="#aaaaaa", lw=1.2, ls=":",  label="Naive")
        ax.plot(bn, de_ronly, color="#a8ff78", lw=1.0, ls="--", label="R-only vs P-ref")
        ax.legend(fontsize=5.5, loc="upper left", facecolor="#080818",
                  edgecolor="#555577", labelcolor="white", framealpha=0.9)
        if row == 0:
            ax.set_title(col_titles[5], color="white", fontsize=7.5,
                         fontweight="bold", pad=5)

    plt.suptitle(
        f"Material: {name}  |  sRGB=({srgb_in[0]:.2f},{srgb_in[1]:.2f},{srgb_in[2]:.2f})"
        f"  |  mu_a={mu_a} mu_e={mu_e} s_a={sigma_a} s_e={sigma_e} a_bar={alpha_bar}"
        f"  |  XYZU [Method {method}]  |  {K} bounces",
        fontsize=10, fontweight="bold", color="white", y=1.005,
    )
    plt.tight_layout(pad=0.6)
    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, f"{name}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    print(f"  Saved: {out}")


# =============================================================================
# 12. Diagnostic plots  (2×3)
#     (0,0) CIE xy chromaticity   (0,1) Y luminance   (0,2) U channel over bounces
#     (1,0) Spectrum WITH fluor.  (1,1) R-only spec.  (1,2) J&H + U-bar overlay
# =============================================================================
def diagnostic_plots_xyzu(
    name, srgb_in, jakob_model,
    method=1,
    ill_name="D65", K=20,
    output_dir=RESULTS / "fluorescence", show_plot=False,
    alpha_bar=0.8, mu_a=430.0, mu_e=560.0, sigma_a=40.0, sigma_e=40.0,
):
    """2×3 diagnostic figure for a single material × illuminant × method."""
    (R_jh, xyz_alb, u_alb, rho_xyzu, R_red4,
     P_sp, _, Fbar_an4, _, P_an4, H4) = _build_material(
        srgb_in, jakob_model, method, alpha_bar, mu_a, sigma_a, mu_e, sigma_e)

    I_n  = normalize_illuminant(load_illuminant(ill_name))
    c_I4 = I_n @ S4

    # ── Spectral ref ─────────────────────────────────────────────────────────
    sp_spectra, sp_xyzu_std = [], []
    L = I_n.copy()
    for _ in range(K):
        L = P_sp @ L
        sp_spectra.append(L.copy())
        sp_xyzu_std.append(NORM_TO_STD @ (L @ S4)[:3])

    # ── R-only ref ───────────────────────────────────────────────────────────
    ro_spectra, ro_xyzu_std = [], []
    L = I_n.copy()
    for _ in range(K):
        L = np.diag(R_jh) @ L
        ro_spectra.append(L.copy())
        ro_xyzu_std.append(NORM_TO_STD @ (L @ S4)[:3])

    # ── Analytic-reduced ─────────────────────────────────────────────────────
    an_xyzu_std = []
    M_an = P_an4.copy()
    for _ in range(K):
        an_xyzu_std.append(NORM_TO_STD @ (M_an @ c_I4)[:3])
        M_an = M_an @ P_an4

    # ── Hybrid ───────────────────────────────────────────────────────────────
    hyb_xyzu_std = [NORM_TO_STD @ (H4 @ c_I4)[:3]]
    Ppow = P_an4.copy()
    for _ in range(1, K):
        hyb_xyzu_std.append(NORM_TO_STD @ (Ppow @ H4 @ c_I4)[:3])
        Ppow = Ppow @ P_an4

    # ── Naive ─────────────────────────────────────────────────────────────────
    nai_xyzu_std = []
    for n in range(1, K + 1):
        nai_xyzu_std.append(NORM_TO_STD @ ((rho_xyzu ** n) * c_I4)[:3])

    # ── U channel over bounces ────────────────────────────────────────────────
    U_sp  = [(L @ S4)[3] for L in sp_spectra]
    U_an  = []
    M_an2 = P_an4.copy()
    for _ in range(K):
        U_an.append((M_an2 @ c_I4)[3])
        M_an2 = M_an2 @ P_an4
    U_hyb = [(H4 @ c_I4)[3]]
    Ppow2 = P_an4.copy()
    for _ in range(1, K):
        U_hyb.append((Ppow2 @ H4 @ c_I4)[3])
        Ppow2 = Ppow2 @ P_an4

    def _to_xy(xyz_list):
        xs, ys = [], []
        for xyz in xyz_list:
            s = xyz.sum()
            xs.append(xyz[0] / s if s > 1e-10 else np.nan)
            ys.append(xyz[1] / s if s > 1e-10 else np.nan)
        return np.array(xs), np.array(ys)

    pipelines = [
        (sp_xyzu_std,  "#ff4444", "Spectral ref (P)",   5,  1.5, "-"),
        (an_xyzu_std,  "#f72585", "Analytic-reduced",   3,  1.0, "-"),
        (hyb_xyzu_std, "#ffaa00", "Hybrid",             3,  1.0, "-"),
        (nai_xyzu_std, "#aaaaaa", "Naive",              2,  0.8, ":"),
        (ro_xyzu_std,  "#a8ff78", "R-only (no fluor.)", 4,  1.2, "--"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(22, 13))
    fig.patch.set_facecolor("#0f0f1e")
    bn = np.arange(1, K + 1)

    # (0,0) CIE xy chromaticity
    ax = axes[0, 0]
    ax.set_facecolor("#080818")
    xyz_raw3  = _xyzu_raw[:, :3]
    locus_sum = xyz_raw3.sum(axis=1)
    mask      = locus_sum > 1e-6
    xl = xyz_raw3[mask, 0] / locus_sum[mask]
    yl = xyz_raw3[mask, 1] / locus_sum[mask]
    ax.fill(np.append(xl, xl[0]), np.append(yl, yl[0]), color="#181830", alpha=0.6)
    ax.plot(xl, yl, color="white", lw=0.8, alpha=0.5)
    ax.plot([xl[0], xl[-1]], [yl[0], yl[-1]], color="white", lw=0.5, ls="--", alpha=0.3)
    for wl_mark in [400, 430, 460, 480, 500, 520, 540, 560, 580, 600, 650, 700]:
        idx = wl_mark - 300
        if 0 <= idx < len(xl):
            ax.plot(xl[idx], yl[idx], "o", color="white", ms=3, alpha=0.4)
            ax.annotate(str(wl_mark), (xl[idx], yl[idx]), textcoords="offset points",
                        xytext=(5, 3), fontsize=5, color="#aaaacc", alpha=0.6)
    idx_em = int(mu_e) - 300
    if 0 <= idx_em < len(xl):
        ax.plot(xl[idx_em], yl[idx_em], "*", color="#f7dc6f", ms=14, zorder=10)
        ax.annotate(f"mu_e={int(mu_e)}nm", (xl[idx_em], yl[idx_em]),
                    textcoords="offset points", xytext=(8, 5), fontsize=7,
                    color="#f7dc6f", fontweight="bold")
    srgb_tri = np.array([[0.64, 0.33], [0.30, 0.60], [0.15, 0.06], [0.64, 0.33]])
    ax.plot(srgb_tri[:, 0], srgb_tri[:, 1], color="white", lw=0.5, alpha=0.15, ls=":")
    ill_std = NORM_TO_STD @ c_I4[:3]
    x_ill, y_ill = ill_std[0] / ill_std.sum(), ill_std[1] / ill_std.sum()
    ax.plot(x_ill, y_ill, "D", color="white", ms=7, zorder=10)
    ax.annotate(ill_name, (x_ill, y_ill), textcoords="offset points", xytext=(6, 5),
                fontsize=7, color="white", fontweight="bold")
    for xyz_list, color, label, ms, lw, ls in pipelines:
        x, y = _to_xy(xyz_list)
        ax.plot(x, y, marker="o", color=color, ms=ms, lw=lw, ls=ls, alpha=0.85, label=label)
        if len(x) > 0 and not np.isnan(x[0]):
            ax.annotate("1", (x[0], y[0]), fontsize=6, color=color, fontweight="bold",
                        textcoords="offset points", xytext=(-9, 4))
    ax.set_xlabel("x", color="white", fontsize=9); ax.set_ylabel("y", color="white", fontsize=9)
    ax.set_title("CIE xy chromaticity", color="white", fontsize=10, fontweight="bold")
    ax.legend(fontsize=6, loc="upper right", facecolor="#080818", edgecolor="#555577", labelcolor="white")
    ax.tick_params(colors="white", labelsize=7)
    ax.set_xlim(-0.05, 0.85); ax.set_ylim(-0.05, 0.95)
    for sp in ax.spines.values(): sp.set_edgecolor("#555577")
    ax.grid(color="#333355", lw=0.3, alpha=0.5)

    # (0,1) Y luminance decay
    ax = axes[0, 1]
    ax.set_facecolor("#080818")
    for xyz_list, color, label, _, lw, ls in pipelines:
        ax.plot(bn, [xyz[1] for xyz in xyz_list], color=color, lw=lw, ls=ls, label=label)
    ax.set_xlabel("bounce", color="white", fontsize=9)
    ax.set_ylabel("Y (luminance, std CIE)", color="white", fontsize=9)
    ax.set_title("Luminance decay", color="white", fontsize=10, fontweight="bold")
    ax.legend(fontsize=6, loc="upper right", facecolor="#080818", edgecolor="#555577", labelcolor="white")
    ax.tick_params(colors="white", labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor("#555577")
    ax.grid(color="#333355", lw=0.3, alpha=0.5)

    # (0,2) U channel over bounces
    ax = axes[0, 2]
    ax.set_facecolor("#080818")
    ax.plot(bn, U_sp,  color="#ff4444", lw=1.5, label="Spectral ref")
    ax.plot(bn, U_an,  color="#f72585", lw=1.2, label="Analytic-reduced")
    ax.plot(bn, U_hyb, color="#ffaa00", lw=1.2, label="Hybrid")
    ax.axhline(0, color="#555577", lw=0.5, ls="--")
    ax.set_xlabel("bounce", color="white", fontsize=9)
    ax.set_ylabel("U-channel value", color="white", fontsize=9)
    ax.set_title("UV channel (U) over bounces", color="white", fontsize=10, fontweight="bold")
    ax.legend(fontsize=6, facecolor="#080818", edgecolor="#555577", labelcolor="white")
    ax.tick_params(colors="white", labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor("#555577")
    ax.grid(color="#333355", lw=0.3, alpha=0.5)

    # (1,0) Output spectrum WITH fluorescence
    ax = axes[1, 0]
    ax.set_facecolor("#080818")
    selected = [0, 4, 9, K - 1]
    cmap = plt.cm.plasma
    for i, b in enumerate(selected):
        if b < len(sp_spectra):
            c = cmap(i / max(len(selected) - 1, 1))
            ax.plot(WL, sp_spectra[b], color=c, lw=1.2, label=f"bounce {b+1}", alpha=0.9)
    ax.axvline(mu_e, color="#f7dc6f", ls="--", lw=0.8, alpha=0.6)
    ax.axvline(mu_a, color="#5dade2", ls="--", lw=0.8, alpha=0.6)
    ax.axvline(380,  color="#888888", ls=":", lw=0.6, alpha=0.5)
    ymax = max(sp_spectra[0].max(), 0.1) if sp_spectra else 1.0
    ax.text(mu_e + 3, ymax * 0.93, f"mu_e={int(mu_e)}", fontsize=6, color="#f7dc6f")
    ax.text(mu_a + 3, ymax * 0.83, f"mu_a={int(mu_a)}", fontsize=6, color="#5dade2")
    ax.text(383, ymax * 0.73, "380nm", fontsize=5, color="#888888")
    ax.set_xlabel("wavelength (nm)", color="white", fontsize=9)
    ax.set_ylabel("spectral radiance", color="white", fontsize=9)
    ax.set_title("Output spectrum (WITH fluorescence, 300–799 nm)",
                 color="white", fontsize=10, fontweight="bold")
    ax.legend(fontsize=6, loc="upper right", facecolor="#080818", edgecolor="#555577", labelcolor="white")
    ax.tick_params(colors="white", labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor("#555577")
    ax.grid(color="#333355", lw=0.3, alpha=0.5)

    # (1,1) R-only output spectrum
    ax = axes[1, 1]
    ax.set_facecolor("#080818")
    for i, b in enumerate(selected):
        if b < len(ro_spectra):
            c = cmap(i / max(len(selected) - 1, 1))
            ax.plot(WL, ro_spectra[b], color=c, lw=1.2, label=f"bounce {b+1}", alpha=0.9)
    ax.axvline(380, color="#888888", ls=":", lw=0.6, alpha=0.5)
    ax.set_xlabel("wavelength (nm)", color="white", fontsize=9)
    ax.set_ylabel("spectral radiance", color="white", fontsize=9)
    ax.set_title("Output spectrum (WITHOUT fluorescence, R-only)",
                 color="white", fontsize=10, fontweight="bold")
    ax.legend(fontsize=6, loc="upper right", facecolor="#080818", edgecolor="#555577", labelcolor="white")
    ax.tick_params(colors="white", labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor("#555577")
    ax.grid(color="#333355", lw=0.3, alpha=0.5)

    # (1,2) J&H spectrum + U-bar overlay (shows what Method 1 integrates)
    ax   = axes[1, 2]
    ax2  = ax.twinx()
    ax.set_facecolor("#080818")
    ax.plot(WL, R_jh, color="#ffffff", lw=1.5, label="J&H reflectance (300–799 nm)")
    ax.axvline(380, color="#888888", ls=":", lw=0.8, alpha=0.5)
    u_bar_norm = _xyzu_raw[:, 3] / _xyzu_raw[:, 3].max()
    ax2.fill_between(WL, 0, u_bar_norm, color="#aa44ff", alpha=0.22, label="u-bar (normalised)")
    ax2.plot(WL, u_bar_norm, color="#aa44ff", lw=0.8, alpha=0.6)
    ax.set_xlabel("wavelength (nm)", color="white", fontsize=9)
    ax.set_ylabel("J&H reflectance", color="white", fontsize=9)
    ax2.set_ylabel("u-bar (normalised)", color="#aa44ff", fontsize=9)
    ax.set_title(f"J&H spectrum + U-bar  |  rho_U(M{method})={u_alb:.4f}",
                 color="white", fontsize=10, fontweight="bold")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=6, facecolor="#080818", edgecolor="#555577", labelcolor="white")
    ax.tick_params(colors="white", labelsize=7)
    ax2.tick_params(colors="#aa44ff", labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor("#555577")
    ax.grid(color="#333355", lw=0.3, alpha=0.5)

    plt.suptitle(
        f"Diagnostic: {name}  |  sRGB=({srgb_in[0]:.2f},{srgb_in[1]:.2f},{srgb_in[2]:.2f})"
        f"  |  {ill_name}  |  mu_a={mu_a} mu_e={mu_e} s_a={sigma_a} s_e={sigma_e}"
        f"  |  XYZU [Method {method}]  |  {K} bounces",
        fontsize=11, fontweight="bold", color="white", y=1.005,
    )
    plt.tight_layout(pad=1.5)
    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, f"{name}_diag_{ill_name}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    print(f"  Diagnostic saved: {out}")


# =============================================================================
# 13. U-channel comparison across all materials  (Method 1 vs Method 2)
# =============================================================================
def u_channel_comparison_plot(
    examples, jakob_model,
    output_dir=RESULTS / "fluorescence", show_plot=False,
):
    """Bar chart + scatter: rho_U (M1 spectral integral vs M2 linear projection)."""
    names = [e[0] for e in examples]
    N     = len(names)
    u_m1  = np.zeros(N)
    u_m2  = np.zeros(N)
    srgbs = [e[1] for e in examples]

    for i, (name, srgb) in enumerate(examples):
        lin  = srgb_to_linear(srgb)
        R_jh = jakob_model.evaluate_spectrum(lin, wavelengths=WL)
        xyz  = R_jh @ S3
        u_m1[i] = float(R_jh @ S4[:, 3])
        u_m2[i] = float(np.clip(T_U @ xyz, 0.0, 1.0))

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.patch.set_facecolor("#0f0f1e")

    x = np.arange(N)
    w = 0.35

    # Left: grouped bar
    ax = axes[0]
    ax.set_facecolor("#080818")
    ax.bar(x - w / 2, u_m1, w, label="Method 1 (J&H spectral integral)",
           color="#4cc9f0", alpha=0.85, edgecolor="#0f0f1e", linewidth=0.5)
    ax.bar(x + w / 2, u_m2, w, label="Method 2 (T_U linear projection)",
           color="#f72585", alpha=0.85, edgecolor="#0f0f1e", linewidth=0.5)
    # Material colour patches below x-axis
    for i, srgb in enumerate(srgbs):
        c = np.clip(srgb, 0, 1)
        ax.add_patch(plt.Rectangle((i - 0.48, -0.022), 0.96, 0.018,
                                    color=c, transform=ax.transData, clip_on=False))
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8, color="white")
    ax.set_ylabel("rho_U  (UV albedo component)", color="white", fontsize=9)
    ax.set_title("UV Albedo: Method 1 vs Method 2", color="white",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, facecolor="#080818", edgecolor="#555577", labelcolor="white")
    ax.tick_params(colors="white", labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor("#555577")
    ax.grid(color="#333355", lw=0.3, axis="y", alpha=0.5)
    ax.set_ylim(bottom=-0.03)

    # Right: M1 vs M2 scatter
    ax = axes[1]
    ax.set_facecolor("#080818")
    for i, (name, srgb) in enumerate(examples):
        c = np.clip(srgb, 0, 1)
        ax.scatter(u_m1[i], u_m2[i], color=c, s=140, zorder=5,
                   edgecolors="white", linewidths=0.6)
        ax.annotate(name, (u_m1[i], u_m2[i]), textcoords="offset points",
                    xytext=(6, 4), fontsize=7, color="white")
    lim = max(u_m1.max(), u_m2.max()) * 1.15 + 0.005
    ax.plot([0, lim], [0, lim], color="#555577", lw=1, ls="--", label="M1 = M2")
    ax.set_xlabel("rho_U  Method 1 (spectral integral)", color="white", fontsize=9)
    ax.set_ylabel("rho_U  Method 2 (linear proj.)", color="white", fontsize=9)
    ax.set_title("M1 vs M2  (identity = perfect agreement)",
                 color="white", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, facecolor="#080818", edgecolor="#555577", labelcolor="white")
    ax.tick_params(colors="white", labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor("#555577")
    ax.grid(color="#333355", lw=0.3, alpha=0.5)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)

    plt.suptitle("UV Channel Comparison: Method 1 (J&H Spectral) vs Method 2 (Linear T_U)",
                 fontsize=12, fontweight="bold", color="white")
    plt.tight_layout(pad=1.0)
    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, "u_channel_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    print(f"  U-channel comparison saved: {out}")


# =============================================================================
# 14. Parameter sweep  (mu_a × mu_e heatmap of ΔE2000 at a fixed bounce)
# =============================================================================
def parameter_sweep_plot(
    name, srgb_in, jakob_model,
    method=1,
    ill_name="D65",
    mu_a_vals=None, mu_e_vals=None,
    sigma_a=30.0, sigma_e=20.0, alpha_bar=0.95,
    bounce_eval=4,          # 0-based index (bounce_eval=4 → 5th bounce)
    output_dir=RESULTS / "fluorescence", show_plot=False,
):
    """Heatmaps of ΔE2000 (BF-reduced and Analytic-reduced vs Spectral) over (mu_a, mu_e) grid."""
    if mu_a_vals is None:
        mu_a_vals = np.array([310, 340, 370, 400, 430, 460])
    if mu_e_vals is None:
        mu_e_vals = np.array([450, 490, 530, 570, 610, 650])

    lin_srgb = srgb_to_linear(srgb_in)
    R_jh     = jakob_model.evaluate_spectrum(lin_srgb, wavelengths=WL)
    xyz_alb  = R_jh @ S3
    u_alb    = float(R_jh @ S4[:, 3]) if method == 1 else float(np.clip(T_U @ xyz_alb, 0, 1))
    rho_xyzu = np.append(xyz_alb, u_alb)
    R_red4   = np.einsum("ijk,k->ij", Rk4, rho_xyzu)

    I_n  = normalize_illuminant(load_illuminant(ill_name))
    c_I4 = I_n @ S4

    Na, Ne = len(mu_a_vals), len(mu_e_vals)
    de_bf  = np.full((Na, Ne), np.nan)
    de_an  = np.full((Na, Ne), np.nan)

    for ia, mu_a in enumerate(mu_a_vals):
        for ie, mu_e in enumerate(mu_e_vals):
            if mu_e <= mu_a + 10:
                continue
            Fbar_sp  = build_F_bar_spectral(alpha_bar, float(mu_a), sigma_a,
                                             float(mu_e), sigma_e)
            P_sp     = np.diag(R_jh) + Fbar_sp @ np.diag(1.0 - R_jh)
            # Spectral ref at bounce (bounce_eval+1 applications)
            L = I_n.copy()
            for _ in range(bounce_eval + 1):
                L = P_sp @ L
            ref_lin = xyzu_to_lin(L @ S4)

            # BF-reduced
            Fbar_bf4 = S4.T @ Fbar_sp @ Sb4
            P_bf4    = R_red4 + Fbar_bf4 @ (np.eye(4) - R_red4)
            M_bf     = P_bf4.copy()
            for _ in range(bounce_eval):
                M_bf = M_bf @ P_bf4
            de_bf[ia, ie] = delta_e_2000(xyzu_to_lin(M_bf @ c_I4), ref_lin)

            # Analytic-reduced
            Fbar_an4 = build_F_bar_analytic(alpha_bar, float(mu_a), sigma_a,
                                             float(mu_e), sigma_e)
            P_an4    = R_red4 + Fbar_an4 @ (np.eye(4) - R_red4)
            M_an     = P_an4.copy()
            for _ in range(bounce_eval):
                M_an = M_an @ P_an4
            de_an[ia, ie] = delta_e_2000(xyzu_to_lin(M_an @ c_I4), ref_lin)

    vmax = np.nanmax([de_bf, de_an])
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor("#0f0f1e")
    for ax, data, title in zip(axes,
                                [de_bf,          de_an],
                                ["BF-reduced",   "Analytic-reduced"]):
        ax.set_facecolor("#080818")
        im = ax.imshow(data, aspect="auto", origin="lower",
                       extent=[mu_e_vals[0], mu_e_vals[-1],
                                mu_a_vals[0], mu_a_vals[-1]],
                       cmap="plasma", vmin=0, vmax=vmax)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("ΔE2000", color="white")
        cbar.ax.tick_params(colors="white")
        ax.set_xlabel("mu_e  (emission center, nm)", color="white", fontsize=9)
        ax.set_ylabel("mu_a  (absorption center, nm)", color="white", fontsize=9)
        ax.set_title(f"{title} ΔE2000", color="white", fontsize=10, fontweight="bold")
        ax.tick_params(colors="white", labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor("#555577")

    plt.suptitle(
        f"Parameter Sweep: {name}  |  {ill_name}  |  bounce {bounce_eval+1}"
        f"  |  sigma_a={sigma_a} sigma_e={sigma_e} alpha_bar={alpha_bar}"
        f"  |  Method {method}",
        fontsize=11, fontweight="bold", color="white",
    )
    plt.tight_layout(pad=1.0)
    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, f"{name}_param_sweep_{ill_name}_M{method}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    print(f"  Parameter sweep saved: {out}")


# =============================================================================
# 15. Sigma sweep  (ΔE2000 curves for varying sigma_a / sigma_e)
# =============================================================================
def sigma_sweep_plot(
    name, srgb_in, jakob_model,
    method=1,
    ill_name="D65", K=10,
    mu_a=380.0, mu_e=520.0, alpha_bar=0.95,
    sigma_vals=None,
    output_dir=RESULTS / "fluorescence", show_plot=False,
):
    """ΔE curves (Analytic-reduced vs Spectral) sweeping sigma_a and sigma_e."""
    if sigma_vals is None:
        sigma_vals = [10.0, 25.0, 40.0, 60.0, 90.0]

    lin_srgb = srgb_to_linear(srgb_in)
    R_jh     = jakob_model.evaluate_spectrum(lin_srgb, wavelengths=WL)
    xyz_alb  = R_jh @ S3
    u_alb    = float(R_jh @ S4[:, 3]) if method == 1 else float(np.clip(T_U @ xyz_alb, 0, 1))
    rho_xyzu = np.append(xyz_alb, u_alb)
    R_red4   = np.einsum("ijk,k->ij", Rk4, rho_xyzu)
    I_n      = normalize_illuminant(load_illuminant(ill_name))
    c_I4     = I_n @ S4
    bn       = np.arange(1, K + 1)

    n_sv  = len(sigma_vals)
    fig, axes = plt.subplots(2, n_sv, figsize=(4.5 * n_sv, 10), sharey="row")
    fig.patch.set_facecolor("#0f0f1e")
    cmap   = plt.cm.viridis
    colors = [cmap(v / max(n_sv - 1, 1)) for v in range(n_sv)]

    row_configs = [
        ("sigma_a_fixed", "sigma_a fixed — varying sigma_e"),
        ("sigma_e_fixed", "sigma_e fixed — varying sigma_a"),
    ]

    for row_idx, (mode, row_label) in enumerate(row_configs):
        for col_idx, sv_fixed in enumerate(sigma_vals):
            ax = axes[row_idx, col_idx]
            ax.set_facecolor("#080818")

            for inner_idx, sv_vary in enumerate(sigma_vals):
                sa = sv_fixed if mode == "sigma_a_fixed" else sv_vary
                se = sv_vary  if mode == "sigma_a_fixed" else sv_fixed
                if mu_e <= mu_a or se < 1.0 or sa < 1.0:
                    continue

                Fbar_sp  = build_F_bar_spectral(alpha_bar, mu_a, sa, mu_e, se)
                P_sp     = np.diag(R_jh) + Fbar_sp @ np.diag(1.0 - R_jh)
                Fbar_an4 = build_F_bar_analytic(alpha_bar, mu_a, sa, mu_e, se)
                P_an4    = R_red4 + Fbar_an4 @ (np.eye(4) - R_red4)

                ref_list, an_list = [], []
                L    = I_n.copy()
                M_an = P_an4.copy()
                for _ in range(K):
                    L    = P_sp @ L
                    ref_list.append(xyzu_to_lin(L @ S4))
                    an_list.append(xyzu_to_lin(M_an @ c_I4))
                    M_an = M_an @ P_an4

                de = [delta_e_2000(an_list[i], ref_list[i]) for i in range(K)]
                lbl = (f"s_e={sv_vary:.0f}" if mode == "sigma_a_fixed"
                       else f"s_a={sv_vary:.0f}")
                ax.plot(bn, de, color=colors[inner_idx], lw=1.2, label=lbl)

            title = (f"sigma_a={sv_fixed:.0f}" if mode == "sigma_a_fixed"
                     else f"sigma_e={sv_fixed:.0f}")
            ax.set_title(title, color="white", fontsize=9, fontweight="bold")
            ax.set_xlabel("bounce", color="white", fontsize=7)
            if col_idx == 0:
                ax.set_ylabel(f"{row_label}\nΔE2000 (Analytic vs Spec.)",
                              color="white", fontsize=7)
            ax.tick_params(colors="white", labelsize=6)
            for sp in ax.spines.values(): sp.set_edgecolor("#555577")
            ax.grid(color="#333355", lw=0.3, alpha=0.5)
            if col_idx == n_sv - 1:
                ax.legend(fontsize=5.5, facecolor="#080818", edgecolor="#555577",
                          labelcolor="white")

    plt.suptitle(
        f"Sigma Sweep: {name}  |  {ill_name}  |  mu_a={mu_a} mu_e={mu_e}"
        f"  |  Method {method}",
        fontsize=11, fontweight="bold", color="white",
    )
    plt.tight_layout(pad=0.8)
    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, f"{name}_sigma_sweep_{ill_name}_M{method}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    print(f"  Sigma sweep saved: {out}")


# =============================================================================
# 16. UV illuminant comparison  (D65 vs UV Gaussian for all materials)
# =============================================================================
def uv_illuminant_plot(
    examples, jakob_model,
    method=1,
    mu_a=350.0, mu_e=500.0, sigma_a=30.0, sigma_e=40.0, alpha_bar=0.95,
    K=10,
    output_dir=RESULTS / "fluorescence", show_plot=False,
):
    """Side-by-side swatches under D65 vs UV Gaussian for all materials."""
    uv_ill = gaussian_illuminant(350.0, 40.0)
    d65    = normalize_illuminant(load_illuminant("D65"))
    Fbar_an4 = build_F_bar_analytic(alpha_bar, mu_a, sigma_a, mu_e, sigma_e)
    Fbar_sp  = build_F_bar_spectral( alpha_bar, mu_a, sigma_a, mu_e, sigma_e)

    n_mat = len(examples)
    fig, axes = plt.subplots(n_mat, 6, figsize=(26, 3.0 * n_mat))
    fig.patch.set_facecolor("#0f0f1e")
    if n_mat == 1:
        axes = axes[np.newaxis, :]

    col_titles = [
        "D65 — ref vs Analytic",
        "D65 — ref vs Hybrid",
        "UV — ref vs Analytic",
        "UV — ref vs Hybrid",
        "ΔE2000 D65",
        "ΔE2000 UV",
    ]
    bw, sw_h = 8, 60

    def _swatch2(ref_d, mdl_d):
        sw = np.zeros((sw_h, K * bw, 3))
        half = sw_h // 2
        for n in range(K):
            x0, x1 = n * bw, (n + 1) * bw
            sw[:half, x0:x1] = np.clip(linear_to_srgb(ref_d[n]), 0, 1)
            sw[half:, x0:x1] = np.clip(linear_to_srgb(mdl_d[n]), 0, 1)
        return sw

    for row, (mat_name, srgb) in enumerate(examples):
        lin_srgb = srgb_to_linear(srgb)
        R_jh     = jakob_model.evaluate_spectrum(lin_srgb, wavelengths=WL)
        xyz_alb  = R_jh @ S3
        u_alb    = float(R_jh @ S4[:, 3]) if method == 1 else float(np.clip(T_U @ xyz_alb, 0, 1))
        rho_xyzu = np.append(xyz_alb, u_alb)
        R_red4   = np.einsum("ijk,k->ij", Rk4, rho_xyzu)
        P_sp     = np.diag(R_jh) + Fbar_sp @ np.diag(1.0 - R_jh)
        P_an4    = R_red4 + Fbar_an4 @ (np.eye(4) - R_red4)
        D4       = np.diag(rho_xyzu)
        H4       = D4 + Fbar_an4 @ (np.eye(4) - D4)

        for ill_col, I_n in enumerate([d65, uv_ill]):
            c_I4 = I_n @ S4

            # Spectral ref
            ref_list = []
            L = I_n.copy()
            for _ in range(K):
                L = P_sp @ L
                ref_list.append(xyzu_to_lin(L @ S4))

            # Analytic
            an_list = []
            M_an = P_an4.copy()
            for _ in range(K):
                an_list.append(xyzu_to_lin(M_an @ c_I4))
                M_an = M_an @ P_an4

            # Hybrid
            hyb_list = [xyzu_to_lin(H4 @ c_I4)]
            Ppow = P_an4.copy()
            for _ in range(1, K):
                hyb_list.append(xyzu_to_lin(Ppow @ H4 @ c_I4))
                Ppow = Ppow @ P_an4

            # Analytic swatch  (col 0 or 2)
            ax = axes[row, ill_col * 2]
            ax.imshow(_swatch2(ref_list, an_list), aspect="auto")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_facecolor("#0f0f1e")
            for sp in ax.spines.values(): sp.set_edgecolor("#555577")
            if row == 0:
                ax.set_title(col_titles[ill_col * 2], color="white",
                             fontsize=7, fontweight="bold")
            if ill_col == 0:
                ax.set_ylabel(mat_name, color="white", fontsize=9,
                              rotation=0, labelpad=45, va="center")

            # Hybrid swatch  (col 1 or 3)
            ax = axes[row, ill_col * 2 + 1]
            ax.imshow(_swatch2(ref_list, hyb_list), aspect="auto")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_facecolor("#0f0f1e")
            for sp in ax.spines.values(): sp.set_edgecolor("#555577")
            if row == 0:
                ax.set_title(col_titles[ill_col * 2 + 1], color="white",
                             fontsize=7, fontweight="bold")

            # ΔE plot  (col 4 or 5)
            ax = axes[row, 4 + ill_col]
            ax.set_facecolor("#080818")
            de_an  = [delta_e_2000(an_list[i],  ref_list[i]) for i in range(K)]
            de_hyb = [delta_e_2000(hyb_list[i], ref_list[i]) for i in range(K)]
            ax.plot(range(1, K + 1), de_an,  color="#f72585", lw=1.3, label="Analytic")
            ax.plot(range(1, K + 1), de_hyb, color="#ffaa00", lw=1.3, label="Hybrid")
            ax.set_xlabel("bounce", color="white", fontsize=7)
            ax.set_ylabel("ΔE2000", color="white", fontsize=7)
            ax.tick_params(colors="white", labelsize=6)
            for sp in ax.spines.values(): sp.set_edgecolor("#555577")
            ax.grid(color="#333355", lw=0.3, alpha=0.5)
            if row == 0:
                ax.set_title(col_titles[4 + ill_col], color="white",
                             fontsize=7, fontweight="bold")
                ax.legend(fontsize=5, facecolor="#080818", edgecolor="#555577",
                          labelcolor="white")

    plt.suptitle(
        f"UV Illuminant Comparison  |  mu_a={mu_a} mu_e={mu_e}"
        f"  sigma_a={sigma_a} sigma_e={sigma_e}  Method {method}",
        fontsize=11, fontweight="bold", color="white",
    )
    plt.tight_layout(pad=0.8)
    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, f"uv_illuminant_comparison_M{method}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    print(f"  UV illuminant comparison saved: {out}")


# =============================================================================
# 17. Run
# =============================================================================
if __name__ == "__main__":
    COEFF_PATH = DATA / "srgb.coeff"
    if not COEFF_PATH.exists():
        print(f"Error: {COEFF_PATH} not found!"); exit(1)

    print("Loading Jakob & Hanika model...")
    jakob_model = RGB2SpecModel(COEFF_PATH)

    examples = [
        ("warm",   np.array([0.80, 0.50, 0.20])),
        ("red",    np.array([0.90, 0.10, 0.10])),
        ("green",  np.array([0.10, 0.80, 0.10])),
        ("blue",   np.array([0.10, 0.10, 0.90])),
        ("purple", np.array([0.70, 0.20, 0.60])),
        ("grey",   np.array([0.50, 0.50, 0.50])),
        ("cyan",   np.array([0.10, 0.80, 0.80])),
        ("white",  np.array([1.00, 1.00, 1.00])),
    ]

    # Fluorescence parameters — mu_a in UV to exercise the U channel
    FLUORO = dict(
        alpha_bar = 0.95,
        mu_a      = 380.0,   # UV absorption: excites the U Gaussian fully
        mu_e      = 520.0,   # visible emission (green)
        sigma_a   = 30.0,
        sigma_e   = 20.0,
    )

    for method in [1, 2]:
        OUT = RESULTS / "fluorescence" / f"method{method}"
        print(f"\n{'='*60}")
        print(f"  Running Method {method}  →  {OUT}")
        print(f"{'='*60}")

        for name, srgb in examples:
            validate_fluorescence_xyzu(
                name, srgb, jakob_model,
                method=method, K=20, output_dir=OUT, show_plot=False, **FLUORO
            )
            diagnostic_plots_xyzu(
                name, srgb, jakob_model,
                method=method, ill_name="D65", K=20,
                output_dir=OUT, show_plot=False, **FLUORO
            )

        # Parameter sweeps for representative materials
        for mat_name, srgb in [("warm", examples[0][1]), ("blue", examples[3][1])]:
            parameter_sweep_plot(
                mat_name, srgb, jakob_model, method=method,
                ill_name="D65", bounce_eval=4,
                sigma_a=FLUORO["sigma_a"], sigma_e=FLUORO["sigma_e"],
                alpha_bar=FLUORO["alpha_bar"],
                output_dir=OUT, show_plot=False,
            )
            sigma_sweep_plot(
                mat_name, srgb, jakob_model, method=method,
                ill_name="D65", K=10,
                mu_a=FLUORO["mu_a"], mu_e=FLUORO["mu_e"],
                alpha_bar=FLUORO["alpha_bar"],
                output_dir=OUT, show_plot=False,
            )

    # Cross-method figures (single output, root folder)
    OUT_ROOT = RESULTS / "fluorescence"
    u_channel_comparison_plot(examples, jakob_model,
                               output_dir=OUT_ROOT, show_plot=False)

    for method in [1, 2]:
        uv_illuminant_plot(
            examples, jakob_model, method=method,
            mu_a=350.0, mu_e=500.0, sigma_a=30.0, sigma_e=40.0, alpha_bar=0.95,
            K=10, output_dir=os.path.join(OUT_ROOT, f"uv_test_M{method}"),
            show_plot=False,
        )

    print(f"\nAll done.  Results in: {OUT_ROOT}/")
