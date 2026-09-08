"""
Diagonal vs. reduced ("upscaled") reflectance transport over multiple bounces.

Standalone reproduction of the supplementary figure script by Laurent Belcour
(Belcour et al. 2025, fig_upscale_diag_v2.py).  The authors' private
dependencies are replaced by plain numpy / matplotlib:
  - radiometry.rs_spectra  (CMF, Spectrum, SPEC_WL, SPEC_STEP)
  - radiometry.colortools  (sRGB_gamma_from_XYZ, XYZ_from_xyY, deltaE2000_XYZ)
  - tikz                   (all tikz output replaced by matplotlib)
The original blue-noise sampling of test colours is kept (commented out) and
replaced by eight fixed example XYZ inputs so the figure is deterministic.

Outputs (results/upscale_diag/):
  de_chroma_spectra.png   DeltaE vs bounce | chromaticity trajectory | spectra
  patches.png             sRGB patch grid: spectral ref / diagonal / upscaled
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches

from _paths import CMF, RESULTS

# =============================================================================
# 0. Configuration — point this to your CIE 2006 CSV
# =============================================================================
CMF_CSV = CMF / "ciexyz06_2deg.csv"
OUT_DIR = RESULTS / "upscale_diag"

# =============================================================================
# 1. Replacement for radiometry.rs_spectra
# =============================================================================

def _load_cie2006_csv(csv_path, wl_min=390, wl_max=830):
    """Load CIE 2006 CMF CSV (no header, columns: wl, xbar, ybar, zbar)."""
    data = np.loadtxt(csv_path, delimiter=",")
    wl   = data[:, 0].astype(int)
    mask = (wl >= wl_min) & (wl <= wl_max)
    return wl[mask], data[mask, 1], data[mask, 2], data[mask, 3]


class _CMF:
    """Minimal CMF class replacing radiometry.rs_spectra.CMF."""
    def __init__(self, csv_path, wl_min=390, wl_max=830):
        wl, xbar, ybar, zbar = _load_cie2006_csv(csv_path, wl_min, wl_max)
        self.wl      = wl                                   # (N,)
        self.xyz_bar = np.column_stack([xbar, ybar, zbar])  # (N, 3)
        self.step    = int(wl[1] - wl[0])                   # 1 nm

    def get_xyz_emissive(self, spectrum_data):
        """Integrate spectrum_data (N,) against CMFs → XYZ triplet."""
        return spectrum_data @ self.xyz_bar   # (3,)


# Load once globally so the rest of the script can use SPEC_WL / SPEC_STEP
_cmf_obj = _CMF(CMF_CSV)
SPEC_WL   = _cmf_obj.wl      # wavelength array  (N,)
SPEC_STEP = _cmf_obj.step     # 1


# =============================================================================
# 2. Replacement for radiometry.colortools
# =============================================================================

# Standard sRGB ← XYZ D65 matrix (IEC 61966-2-1)
_M_XYZ_TO_sRGB = np.array([
    [ 3.2404542, -1.5371385, -0.4985314],
    [-0.9692660,  1.8760108,  0.0415560],
    [ 0.0556434, -0.2040259,  1.0572252],
])

def sRGB_gamma_from_XYZ(xyz):
    """XYZ → sRGB with gamma (clipped to [0, 1])."""
    rgb_lin = _M_XYZ_TO_sRGB @ np.asarray(xyz)
    rgb_lin = np.clip(rgb_lin, 0.0, 1.0)
    # sRGB piecewise gamma
    gamma = np.where(
        rgb_lin <= 0.0031308,
        12.92 * rgb_lin,
        1.055 * np.power(np.maximum(rgb_lin, 0.0), 1.0 / 2.4) - 0.055,
    )
    return np.clip(gamma, 0.0, 1.0)


def XYZ_from_xyY(xy, Y=1.0):
    """chromaticity (x, y) + luminance Y → XYZ."""
    x, y = xy[0], xy[1]
    if y < 1e-8:
        return np.array([0.0, 0.0, 0.0])
    X = Y * x / y
    Z = Y * (1.0 - x - y) / y
    return np.array([X, Y, Z])


def _xyz_to_lab(xyz, xyz_n=None):
    """XYZ → CIELAB (D65 white by default)."""
    if xyz_n is None:
        xyz_n = np.array([0.95047, 1.00000, 1.08883])  # D65
    ratio = np.asarray(xyz) / xyz_n
    def f(t):
        delta = 6.0 / 29.0
        return np.where(t > delta**3,
                        np.cbrt(np.maximum(t, 0.0)),
                        t / (3.0 * delta**2) + 4.0 / 29.0)
    fx, fy, fz = f(ratio[0]), f(ratio[1]), f(ratio[2])
    L  = 116.0 * fy - 16.0
    a  = 500.0 * (fx - fy)
    b  = 200.0 * (fy - fz)
    return np.array([L, a, b])


def deltaE2000_XYZ(xyz1, xyz2):
    """CIEDE2000 colour difference between two XYZ triplets."""
    # Convert to Lab
    L1, a1, b1 = _xyz_to_lab(xyz1)
    L2, a2, b2 = _xyz_to_lab(xyz2)

    # CIEDE2000
    kL, kC, kH = 1.0, 1.0, 1.0

    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    Cb = (C1 + C2) / 2.0
    Cb7 = Cb**7
    G  = 0.5 * (1.0 - np.sqrt(Cb7 / (Cb7 + 25.0**7)))
    a1p = a1 * (1.0 + G)
    a2p = a2 * (1.0 + G)
    C1p = np.sqrt(a1p**2 + b1**2)
    C2p = np.sqrt(a2p**2 + b2**2)

    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0

    dLp = L2 - L1
    dCp = C2p - C1p

    if C1p * C2p < 1e-10:
        dhp = 0.0
    elif abs(h2p - h1p) <= 180.0:
        dhp = h2p - h1p
    elif h2p - h1p > 180.0:
        dhp = h2p - h1p - 360.0
    else:
        dhp = h2p - h1p + 360.0

    dHp = 2.0 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp / 2.0))

    Lbp = (L1 + L2) / 2.0
    Cbp = (C1p + C2p) / 2.0

    if C1p * C2p < 1e-10:
        hbp = h1p + h2p
    elif abs(h1p - h2p) <= 180.0:
        hbp = (h1p + h2p) / 2.0
    elif h1p + h2p < 360.0:
        hbp = (h1p + h2p + 360.0) / 2.0
    else:
        hbp = (h1p + h2p - 360.0) / 2.0

    T  = (1.0
          - 0.17 * np.cos(np.radians(hbp - 30.0))
          + 0.24 * np.cos(np.radians(2.0 * hbp))
          + 0.32 * np.cos(np.radians(3.0 * hbp + 6.0))
          - 0.20 * np.cos(np.radians(4.0 * hbp - 63.0)))

    SL = 1.0 + 0.015 * (Lbp - 50.0)**2 / np.sqrt(20.0 + (Lbp - 50.0)**2)
    SC = 1.0 + 0.045 * Cbp
    SH = 1.0 + 0.015 * Cbp * T

    Cbp7 = Cbp**7
    RC  = 2.0 * np.sqrt(Cbp7 / (Cbp7 + 25.0**7))
    d_theta = 30.0 * np.exp(-((hbp - 275.0) / 25.0)**2)
    RT  = -np.sin(np.radians(2.0 * d_theta)) * RC

    dE = np.sqrt(
        (dLp / (kL * SL))**2 +
        (dCp / (kC * SC))**2 +
        (dHp / (kH * SH))**2 +
        RT * (dCp / (kC * SC)) * (dHp / (kH * SH))
    )
    return float(dE)


# =============================================================================
# 3. Build CMF matrices (Laurent's exact method)
# =============================================================================

cmf = _cmf_obj

S     = cmf.xyz_bar.copy()           # (N, 3)  raw CMFs
Snorm = S.sum(axis=0, keepdims=True) # column sums
S     = S / Snorm                    # normalised so each column sums to 1
Sb    = S @ np.linalg.inv(S.T @ S)  # dual basis  (N, 3)

Rxyz = np.concatenate([
    (S.T @ np.diag(Sb[:, 0]) @ Sb)[..., None],
    (S.T @ np.diag(Sb[:, 1]) @ Sb)[..., None],
    (S.T @ np.diag(Sb[:, 2]) @ Sb)[..., None],
], axis=-1)   # (3, 3, 3)

print('S.shape    = ', S.shape)
print('Sb.shape   = ', Sb.shape)
print('Rxyz.shape = ', Rxyz.shape)


# =============================================================================
# 4. Example XYZ inputs  (replaces blue-noise sampling)
#    These are raw XYZ values — they will be projected onto the spectral
#    subspace the same way Laurent does with the blue-noise samples.
# =============================================================================
example_xyz_raw = np.array([
    [0.8, 0.5, 0.2],   # warm
    [0.2, 0.7, 0.3],   # greenish
    [0.3, 0.3, 0.9],   # bluish
    [0.7, 0.2, 0.6],   # purple
    [0.4, 0.4, 0.4],   # grey
    [0.9, 0.1, 0.1],   # red
    [0.1, 0.9, 0.1],   # green
    [0.1, 0.1, 0.9],   # blue
])

# Project each XYZ into the spectral subspace (same pipeline as blue-noise)
def _xyz_to_valid_spec(xyz_in, S, Sb, scale=0.8):
    """Project XYZ → spectrum → clip → back to XYZ, matching Laurent's pipeline."""
    spec = Sb @ xyz_in
    spec = scale * spec / spec.max()
    spec_clip = spec.clip(0.0, 1.0)
    xyz_out = spec_clip @ S
    return spec_clip, xyz_out

# Build all_xyz by projecting the example inputs
all_xyz_spec   = []
all_xyz_proj   = []
for xyz_raw in example_xyz_raw:
    sc, xyz_p = _xyz_to_valid_spec(xyz_raw, S, Sb)
    all_xyz_spec.append(sc)
    all_xyz_proj.append(xyz_p)

all_xyz_spec = np.array(all_xyz_spec)   # (N, SPEC_N) — spectra
all_xyz      = np.array(all_xyz_proj)   # (N, 3)      — XYZ after projection


# Blue-noise alternative (commented out — uncomment to use):
# def sample_bluenoise(nbSamples, radius=1.0, nbTry=1_000_000):
#     samples = [np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])]
#     samplesChroma = []
#     nbConsecutiveTry = 0
#     k = 0
#     while k < nbSamples:
#         print(f'Bluenoise: {k} valid, {nbConsecutiveTry} tries', end='\r')
#         xyz1 = np.random.rand(3)
#         spec = Sb @ xyz1
#         spec = 0.8 * spec / spec.max()
#         spec_clip = spec.clip(0, 1)
#         xyz1 = spec_clip @ S
#         xy1  = xyz1[0:2] / xyz1.sum()
#         sampleValid = all(np.linalg.norm(xy1 - xyk) > radius for xyk in samplesChroma)
#         if sampleValid:
#             samples += [xyz1]; samplesChroma += [xy1]; k += 1; nbConsecutiveTry = 0
#         else:
#             radius *= 0.9999; nbConsecutiveTry += 1
#         if nbConsecutiveTry > nbTry:
#             return np.array(samples)[3:]
#     print()
#     return np.array(samples)[3:]
# all_xyz = sample_bluenoise(100)[:N]


# =============================================================================
# 5. Main loop — identical to Laurent's
# =============================================================================

K = 10
N = len(all_xyz)

deltaEAll_diag    = np.zeros((K,))
deltaEAll_upscale = np.zeros((K,))
deltaE_diag       = np.zeros((K,))
deltaE_upscale    = np.zeros((K,))
diffY_diag        = np.zeros((K,))
diffY_upscale     = np.zeros((K,))

patches_colors_spec    = np.zeros((N, K, 3))
patches_colors_diag    = np.zeros((N, K, 3))
patches_colors_upscale = np.zeros((N, K, 3))

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[1].axis([0, 1, 0, 1])

# Spectral locus in chromaticity
xy_curve = cmf.xyz_bar[:, 0:2] / (cmf.xyz_bar.sum(axis=-1, keepdims=True) + 1e-8)
ax[1].plot(xy_curve[:, 0], xy_curve[:, 1], 'k-', lw=1.5, label='spectral locus')

# Dual basis functions (background)
ax[2].plot(SPEC_WL, Sb[:, 0], 'r', alpha=0.15)
ax[2].plot(SPEC_WL, Sb[:, 1], 'g', alpha=0.15)
ax[2].plot(SPEC_WL, Sb[:, 2], 'b', alpha=0.15)

# Colour cycle for samples
prop_cycle = plt.rcParams['axes.prop_cycle']
colors_cycle = [c['color'] for c in prop_cycle]

for i, (xyz, spec_clip) in enumerate(zip(all_xyz, all_xyz_spec)):
    # spec_clip already computed above; use it directly
    R = Rxyz @ xyz

    spec_k = spec_clip.copy()
    xyz_k  = xyz.copy()
    R_k    = R.copy()

    col = colors_cycle[i % len(colors_cycle)]

    if i == 0:
        ax[2].plot(SPEC_WL, spec_clip, 'k',    lw=1.5,  label='spectrum (sample 0)')
        ax[2].plot(SPEC_WL, spec_clip, color='gray', linestyle='dashed', alpha=0.7)

    ax[1].scatter([xyz[0] / xyz.sum()], [xyz[1] / xyz.sum()],
                  color=col, zorder=5, s=40)

    xy_diag_curve    = np.zeros((K, 2))
    xy_spec_curve    = np.zeros((K, 2))
    xy_upscale_curve = np.zeros((K, 2))

    for k in range(K):
        # — Spectral reference —
        xyz_spec = spec_k @ S
        sum_spec = xyz_spec.sum()
        xy_spec  = xyz_spec[0:2] / sum_spec if sum_spec > 1e-10 else np.array([0.33, 0.33])
        xy_spec_curve[k] = xy_spec
        xyz_Y_spec = XYZ_from_xyY(xy_spec, 0.5)
        spec_k *= spec_clip

        # — Naive / diagonal —
        xyz_diag = xyz_k.copy()
        sum_diag = xyz_diag.sum()
        xy_diag  = xyz_diag[0:2] / sum_diag if sum_diag > 1e-10 else np.array([0.33, 0.33])
        xyz_Y_diag = XYZ_from_xyY(xy_diag, 0.5)
        xy_diag_curve[k] = xy_diag
        xyz_k *= xyz

        # — Upscaled / reduced matrix —
        xyz_upscale = R_k @ np.ones(3)
        sum_up = xyz_upscale.sum()
        xy_upscale = xyz_upscale[0:2] / sum_up if sum_up > 1e-10 else np.array([0.33, 0.33])
        xyz_Y_upscale = XYZ_from_xyY(xy_upscale, 0.5)
        xy_upscale_curve[k] = xy_upscale
        R_k = R_k @ R

        # Accumulate delta-E averages
        deltaEAll_diag[k]    += deltaE2000_XYZ(xyz_diag,   xyz_spec) / N
        deltaEAll_upscale[k] += deltaE2000_XYZ(xyz_upscale, xyz_spec) / N
        deltaE_diag[k]    += deltaE2000_XYZ(xyz_Y_diag,   xyz_Y_spec) / N
        deltaE_upscale[k] += deltaE2000_XYZ(xyz_Y_upscale, xyz_Y_spec) / N
        diffY_diag[k]    += deltaE2000_XYZ(
            XYZ_from_xyY([0.33, 0.33], xyz_spec[1]),
            XYZ_from_xyY([0.33, 0.33], xyz_diag[1])) / N
        diffY_upscale[k] += deltaE2000_XYZ(
            XYZ_from_xyY([0.33, 0.33], xyz_spec[1]),
            XYZ_from_xyY([0.33, 0.33], xyz_upscale[1])) / N

        if i == 0:
            ax[2].plot(SPEC_WL, spec_k, 'gray',  alpha=(1.0 / (k + 1.0)))
            ax[2].plot(SPEC_WL, Sb @ np.clip(xyz_upscale, 0, None),
                       'green', alpha=(1.0 / (k + 1.0)))
            rect = matplotlib.patches.Rectangle(
                (k / K, 0.0), 1/K, 1/K, facecolor=np.clip(xyz_spec, 0, 1))
            ax[1].add_patch(rect)
            rect = matplotlib.patches.Rectangle(
                (k / K, 1/K), 1/K, 1/K, facecolor=np.clip(xyz_diag, 0, 1))
            ax[1].add_patch(rect)
            rect = matplotlib.patches.Rectangle(
                (k / K, 2/K), 1/K, 1/K, facecolor=np.clip(xyz_upscale, 0, 1))
            ax[1].add_patch(rect)

        patches_colors_spec[i][k]    = sRGB_gamma_from_XYZ(xyz_spec)
        patches_colors_diag[i][k]    = sRGB_gamma_from_XYZ(xyz_diag)
        patches_colors_upscale[i][k] = sRGB_gamma_from_XYZ(xyz_upscale)

    # Plot chromaticity trajectories
    if i == 0:
        h, = ax[1].plot(xy_spec_curve[:, 0],    xy_spec_curve[:, 1],    color=col,
                        label='Spectral ref.')
        ax[1].plot(xy_diag_curve[:, 0],    xy_diag_curve[:, 1],    color=col,
                   linestyle='dotted',  label='Diagonal R')
        ax[1].plot(xy_upscale_curve[:, 0], xy_upscale_curve[:, 1], color=col,
                   linestyle='dashed',  label='Upscaled R')
    else:
        ax[1].plot(xy_spec_curve[:, 0],    xy_spec_curve[:, 1],    color=col)
        ax[1].plot(xy_diag_curve[:, 0],    xy_diag_curve[:, 1],    color=col, linestyle='dotted')
        ax[1].plot(xy_upscale_curve[:, 0], xy_upscale_curve[:, 1], color=col, linestyle='dashed')

    # Starting-point markers
    ax[1].plot(xy_spec_curve[0, 0],    xy_spec_curve[0, 1],    color=col, marker='o', ms=5)
    ax[1].plot(xy_diag_curve[0, 0],    xy_diag_curve[0, 1],    color=col, marker='o', ms=5,
               linestyle='dotted')
    ax[1].plot(xy_upscale_curve[0, 0], xy_upscale_curve[0, 1], color=col, marker='o', ms=5,
               linestyle='dashed')
    ax[1].text(xy_spec_curve[0, 0], xy_spec_curve[0, 1] + 0.025,
               f'({i})', fontsize=7, ha='center', color=col)

    print(f'  sample {i:2d}  xyz = {xyz}')


# =============================================================================
# 6. ax[0] — delta-E vs. bounce count
# =============================================================================
h, = ax[0].plot(deltaE_diag,    label='dE xy  (diagonal)')
ax[0].plot(deltaE_upscale,   color=h.get_color(), linestyle='dashed',  label='dE xy  (upscale)')
h, = ax[0].plot(diffY_diag,     label='dE Y   (diagonal)')
ax[0].plot(diffY_upscale,    color=h.get_color(), linestyle='dashed',  label='dE Y   (upscale)')
h, = ax[0].plot(deltaEAll_diag,  label='dE XYZ (diagonal)')
ax[0].plot(deltaEAll_upscale, color=h.get_color(), linestyle='dashed',  label='dE XYZ (upscale)')
ax[0].legend(fontsize=7)
ax[0].set_title('ΔE vs. number of bounces')
ax[0].set_xlabel('bounce index')
ax[0].set_ylabel('ΔE 2000')
ax[0].axis([0, K, 0, 100])

ax[1].set_title('chromaticity vs. number of bounces')
ax[1].set_xlabel('x'); ax[1].set_ylabel('y')
ax[1].legend(fontsize=7, loc='upper right')

ax[2].set_title('spectral upsampling & saturation vs. bounces')
ax[2].set_xlabel('wavelength (nm)'); ax[2].set_ylabel('reflectance')

plt.tight_layout()


# =============================================================================
# 7. Extra figure: colour patch grid (replaces tikz colour patches)
# =============================================================================
fig2, axes2 = plt.subplots(N, 3, figsize=(K * 0.6 + 1, N * 0.6 + 1))
labels = ['Spectral ref.', 'Diagonal', 'Upscaled']
for i in range(N):
    for row, (patches, lbl) in enumerate(zip(
        [patches_colors_spec[i], patches_colors_diag[i], patches_colors_upscale[i]],
        labels
    )):
        ax2 = axes2[i, row]
        ax2.imshow(patches[np.newaxis, :, :], aspect='auto')
        ax2.set_yticks([])
        ax2.set_xticks(range(K))
        ax2.set_xticklabels([str(k) for k in range(K)], fontsize=6)
        if row == 0:
            ax2.set_ylabel(f'({i})', fontsize=7, rotation=0, labelpad=20)
        if i == 0:
            ax2.set_title(lbl, fontsize=8)

fig2.suptitle('sRGB colour patches per sample / bounce / pipeline', fontsize=9)
plt.tight_layout()

os.makedirs(OUT_DIR, exist_ok=True)
fig.savefig(OUT_DIR / "de_chroma_spectra.png", dpi=150, bbox_inches="tight")
fig2.savefig(OUT_DIR / "patches.png", dpi=150, bbox_inches="tight")
print(f"Saved figures to {OUT_DIR}/")
plt.close("all")
