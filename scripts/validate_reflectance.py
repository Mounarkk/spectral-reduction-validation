"""
Reflectance-only validation in the 3-channel XYZ basis (380-780 nm).

Establishes the two modelling decisions that the fluorescence pipeline and
the Godot port build on:

  * Alpha-scaling of the albedo.  An sRGB colour reconstructed to a
    reflectance spectrum through the dual basis (Sb @ xyz) can exceed 1.
    The albedo is scaled in linear sRGB by alpha = 0.9 / max(Sb @ xyz) so the
    reconstructed spectrum stays physically valid (<= ~0.9).
  * Hybrid transport.  The first (directly lit) bounce uses the diagonal
    reflectance diag(rho); every later bounce uses the reduced matrix
    R = sum_k rho_k R_k.  Compared against pure-reduced and naive transport.

The spectral reference uses a *smoothed* illuminant I_smooth = Sb (S^T I),
re-normalised to Y = 1, so that reference and models share the same reduced
representation of the light (fair for spiky illuminants such as HP1).

Outputs (results/reflectance/<material>.png): 6 illuminant rows x 7 columns
  Illuminant | Hybrid | Reduced | Naive | DeltaE2000 vs bounce | I_smooth | R(lambda)

Converted from an exploratory notebook.  The XYZ->linear-sRGB constant was
corrected to the IEC 61966-2-1 value (0.0556434).
"""
import os

import numpy as np
import matplotlib.pyplot as plt
import colour

from _paths import CMF, RESULTS

# ============================================================
# 1. Wavelength grid
# ============================================================
WL = np.arange(380, 781, 1)   # 401 points, 1 nm steps

# ============================================================
# 2. Laurent's normalised CMF basis
# ============================================================
def load_cie2006(csv_path, wl_min=380, wl_max=780, step=1):
    data = np.loadtxt(csv_path, delimiter=",")
    wl_raw = data[:, 0]
    cmfs_raw = data[:, 1:4]
    wl_grid = np.arange(wl_min, wl_max + 1, step)
    cmfs_interp = np.zeros((len(wl_grid), 3))
    for i in range(3):
        cmfs_interp[:, i] = np.interp(wl_grid, wl_raw, cmfs_raw[:, i],
                                      left=0.0, right=0.0)
    return cmfs_interp[:, 0], cmfs_interp[:, 1], cmfs_interp[:, 2]

xbar, ybar, zbar = load_cie2006(CMF / "ciexyz06_2deg.csv")
S_raw = np.column_stack([xbar, ybar, zbar])
S     = S_raw / S_raw.sum(axis=0, keepdims=True)      # column sums = 1
Sb    = S @ np.linalg.pinv(S.T @ S)                    # dual basis
Rxyz  = np.stack([S.T @ np.diag(Sb[:, i]) @ Sb for i in range(3)], axis=-1)  # (3,3,3)

# ============================================================
# 3. Illuminant helpers
# ============================================================
def load_illuminant(name: str) -> np.ndarray:
    ill = colour.SDS_ILLUMINANTS[name]
    return ill.copy().align(colour.SpectralShape(380, 780, 1)).values

def normalize_illuminant(I_raw: np.ndarray) -> np.ndarray:
    """Make Y=1 for a perfect white under the normalised ybar."""
    Y_white = float(I_raw @ S[:, 1])
    return I_raw / Y_white

def smooth_illuminant(I_norm: np.ndarray) -> np.ndarray:
    """
    Project illuminant to XYZ and reconstruct a smooth spectrum,
    then re‑normalize so that a perfect white reflector still gives Y=1.
    """
    xyz = I_norm @ S
    I_smooth = Sb @ xyz
    # Remove tiny negatives that may arise
    I_smooth = np.clip(I_smooth, 0.0, None)
    # Re‑normalize after the projection so that Y remains 1 for a perfect white
    Y_smooth = float(I_smooth @ S[:, 1])
    if Y_smooth > 1e-10:
        I_smooth = I_smooth / Y_smooth
    return I_smooth

# ============================================================
# 4. Colour helpers (sRGB <-> linear RGB <-> XYZ)
# ============================================================
XYZ_TO_LINEAR_SRGB = np.array([
    [ 3.2404542, -1.5371385, -0.4985314],
    [-0.9692660,  1.8760108,  0.0415560],
    [ 0.0556434, -0.2040259,  1.0572252],
])
LINEAR_SRGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])

def srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    srgb = np.clip(srgb, 0.0, 1.0)
    return np.where(srgb <= 0.04045,
                    srgb / 12.92,
                    ((srgb + 0.055) / 1.055) ** 2.4)

def linear_to_srgb(lin: np.ndarray) -> np.ndarray:
    c = np.clip(lin, 0.0, None)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)

def xyz_to_lin_srgb(xyz: np.ndarray) -> np.ndarray:
    return XYZ_TO_LINEAR_SRGB @ xyz

def lin_srgb_to_xyz(lin: np.ndarray) -> np.ndarray:
    return LINEAR_SRGB_TO_XYZ @ lin

_D65_WHITE = np.array([0.95047, 1.00000, 1.08883])

def delta_e_2000(rgb_a: np.ndarray, rgb_b: np.ndarray) -> float:
    xyz_a = LINEAR_SRGB_TO_XYZ @ np.clip(rgb_a, 0, 1)
    xyz_b = LINEAR_SRGB_TO_XYZ @ np.clip(rgb_b, 0, 1)
    lab_a = colour.XYZ_to_Lab(xyz_a, illuminant=_D65_WHITE)
    lab_b = colour.XYZ_to_Lab(xyz_b, illuminant=_D65_WHITE)
    return float(colour.delta_E(lab_a, lab_b, method='CIE 2000'))

# ============================================================
# 5. Core validation for a single sRGB material
# ============================================================
def validate_srgb_material(
    name: str,
    srgb_in: np.ndarray,
    illuminant_names=None,
    K: int = 20,
    output_dir=RESULTS / "reflectance",
    show_plot: bool = False
):
    if illuminant_names is None:
        illuminant_names = ["D65", "D50", "HP1", "E", "A", "LED-B1"]

    # ---- Albedo: from sRGB to physically valid reflectance spectrum ----
    lin_srgb_orig = srgb_to_linear(srgb_in)
    xyz_orig = lin_srgb_to_xyz(lin_srgb_orig)

    spec_raw = Sb @ xyz_orig
    alpha = 0.9 / spec_raw.max()

    # Effective linear sRGB albedo (scaled by alpha) — this is the "physically realisable" colour
    effective_lin_srgb = lin_srgb_orig * alpha
    effective_srgb = linear_to_srgb(effective_lin_srgb)   # for display only
    effective_xyz = lin_srgb_to_xyz(effective_lin_srgb)   # used for all reduced models

    # Physical reflectance spectrum 
    R_spectrum = np.clip(Sb @ effective_xyz, 0.0, 1.0)

    xyz_check = R_spectrum @ S
    print(f"  [{name}] alpha = {alpha:.4f}, "
          f"effective sRGB = {effective_srgb.round(3)}  "
          f"xyz albedo = {effective_xyz.round(4)}  "
          f"spectrum XYZ diff = {np.linalg.norm(effective_xyz - xyz_check):.2e}")

    # ---- Build reduced reflectance matrix ----
    xyz_albedo = effective_xyz.copy()
    R_rxyz_mat = Rxyz @ xyz_albedo

    # ---- Figure layout (7 columns) ----
    col_titles = [
        "Illuminant",
        "Hybrid\n(R^(n-1)·diag·I)",
        "Reduced\n(Rxyz)",
        "Naive\n(diagonal)",
        "ΔE 2000\nvs bounce",
        "I_smooth\nspectrum",
        "R(λ)\nmaterial spectrum"
    ]
    N_rows = len(illuminant_names)
    fig, axes = plt.subplots(
        N_rows, 7,
        figsize=(32, 3.5 * N_rows),
        gridspec_kw={"width_ratios": [0.8, 1, 1, 1, 1.6, 1.2, 1.2]},
    )
    fig.patch.set_facecolor("#1a1a2e")
    if N_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, ill_name in enumerate(illuminant_names):
        I_raw = load_illuminant(ill_name)
        I_norm = normalize_illuminant(I_raw)          # Y=1 for perfect white
        I_smooth = smooth_illuminant(I_norm)          # Y-normalized smooth illuminant
        white_I = I_smooth @ S                        # light vector (Y≈1) for reduced models

        # ---- Spectral reference (bounces) using smooth illuminant ----
        ref_lin, ref_disp = [], []
        spec_cur = R_spectrum.copy()
        for _ in range(K):
            xyz = (spec_cur * I_smooth) @ S
            lin = xyz_to_lin_srgb(xyz)               # may be >1 — that's physically fine
            ref_lin.append(lin)
            ref_disp.append(linear_to_srgb(lin))     # will be clamped for display
            spec_cur = spec_cur * R_spectrum

        # ---- Hybrid ----
        hybrid_lin, hybrid_disp = [], []
        Diag = np.diag(xyz_albedo)
        vec1 = Diag @ white_I
        M_cur = np.eye(3)
        for _ in range(K):
            xyz = M_cur @ vec1
            lin = xyz_to_lin_srgb(xyz)
            hybrid_lin.append(lin)
            hybrid_disp.append(linear_to_srgb(lin))
            M_cur = M_cur @ R_rxyz_mat

        # ---- Rxyz‑reduced ----
        rred_lin, rred_disp = [], []
        N_cur = R_rxyz_mat.copy()
        for _ in range(K):
            xyz = N_cur @ white_I
            lin = xyz_to_lin_srgb(xyz)
            rred_lin.append(lin)
            rred_disp.append(linear_to_srgb(lin))
            N_cur = N_cur @ R_rxyz_mat

        # ---- Naive diagonal ----
        naive_lin, naive_disp = [], []
        for n in range(1, K + 1):
            xyz = (xyz_albedo ** n) * white_I
            lin = xyz_to_lin_srgb(xyz)
            naive_lin.append(lin)
            naive_disp.append(linear_to_srgb(lin))

        # ---- Swatches (bounces left→right) ----
        band_w = 8
        swatch_h = 100
        swatch_w = K * band_w
        half_h = swatch_h // 2

        def make_swatch_rotated(model_disp):
            sw = np.zeros((swatch_h, swatch_w, 3))
            for n in range(K):
                x0 = n * band_w
                x1 = x0 + band_w
                sw[:half_h, x0:x1] = np.clip(ref_disp[n],   0, 1)
                sw[half_h:,  x0:x1] = np.clip(model_disp[n], 0, 1)
            return sw

        illum_color = np.clip(linear_to_srgb(xyz_to_lin_srgb(white_I)), 0, 1)
        swatch_illum = np.ones((100, 100, 3)) * illum_color

        swatches = [
            swatch_illum,
            make_swatch_rotated(hybrid_disp),
            make_swatch_rotated(rred_disp),
            make_swatch_rotated(naive_disp),
        ]

        # Columns 0-3: swatches
        for col, sw in enumerate(swatches):
            ax = axes[row, col]
            ax.set_facecolor("#1a1a2e")
            ax.imshow(np.clip(sw, 0, 1), aspect='auto')
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("#444466")
                spine.set_linewidth(1.5)
            if row == 0:
                ax.set_title(col_titles[col], color="white",
                             fontsize=9, fontweight="bold", pad=6)
            if col == 0:
                ax.text(0.5, 0.5, ill_name, transform=ax.transAxes,
                        ha="center", va="center", fontsize=10,
                        fontweight="bold", color="white",
                        bbox=dict(boxstyle="round,pad=0.25",
                                  fc="#1a1a2e", alpha=0.75))
            if row == 0 and col > 0 and col < 4:
                for n in range(K):
                    x_center = (n + 0.5) * band_w / swatch_w
                    ax.text(x_center, -0.05, f'{n+1}', transform=ax.transAxes,
                            ha='center', va='bottom', fontsize=5, color='white')

        # ---- ΔE vs bounce plot (column 4) ----
        ax = axes[row, 4]
        ax.set_facecolor("#12122a")
        de_hybrid = [delta_e_2000(hybrid_lin[i], ref_lin[i]) for i in range(K)]
        de_rxyz   = [delta_e_2000(rred_lin[i],   ref_lin[i]) for i in range(K)]
        de_naive  = [delta_e_2000(naive_lin[i],  ref_lin[i]) for i in range(K)]
        max_de = max(max(de_hybrid), max(de_rxyz), max(de_naive), 0.1)
        ax.set_xlim(1, K)
        ax.set_ylim(0, max_de * 1.1)
        ax.tick_params(colors='white', labelsize=7)
        ax.set_xlabel("bounce", color='white', fontsize=8)
        if row == 0:
            ax.set_title(col_titles[4], color="white",
                         fontsize=9, fontweight="bold", pad=6)
        ax.plot(range(1, K+1), de_hybrid, color="#ffaa00", lw=1.5, label="Hybrid")
        ax.plot(range(1, K+1), de_rxyz,   color="#f72585", lw=1.5, label="Rxyz")
        ax.plot(range(1, K+1), de_naive,  color="#aaaaaa", lw=1.5, label="Naive")
        ax.legend(fontsize=6, loc='upper left', facecolor='#12122a',
                  edgecolor='white', labelcolor='white')
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")

        # ---- Column 5: I_smooth spectrum plot ----
        ax = axes[row, 5]
        ax.set_facecolor("#12122a")
        ax.plot(WL, I_smooth, color='#88ccff', lw=1.0)
        ax.set_xlim(380, 780)
        ax.set_ylim(0, max(I_smooth.max() * 1.1, 0.01))
        ax.tick_params(colors='white', labelsize=6)
        if row == 0:
            ax.set_title(col_titles[5], color="white", fontsize=8, fontweight="bold", pad=6)
        ax.set_xlabel("λ (nm)", color='white', fontsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")

        # ---- Column 6: R_spectrum (material) ----
        ax = axes[row, 6]
        ax.set_facecolor("#12122a")
        ax.plot(WL, R_spectrum, color='#ff8888', lw=1.0)
        ax.set_xlim(380, 780)
        ax.set_ylim(0, 1.0)
        ax.tick_params(colors='white', labelsize=6)
        if row == 0:
            ax.set_title(col_titles[6], color="white", fontsize=8, fontweight="bold", pad=6)
        ax.set_xlabel("λ (nm)", color='white', fontsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")

    plt.suptitle(
        f"Material: {name}  |  sRGB input = ({srgb_in[0]:.2f}, {srgb_in[1]:.2f}, {srgb_in[2]:.2f})  "
        f"|  effective sRGB = ({effective_srgb[0]:.2f}, {effective_srgb[1]:.2f}, {effective_srgb[2]:.2f})"
        f"  |  α = {alpha:.4f}  |  {K} bounces",
        fontsize=13, fontweight="bold", color="white", y=1.01,
    )
    plt.tight_layout(pad=0.8)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{name}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    print(f"  → Saved: {out_path}")

# ============================================================
# 6. Run with example sRGB colours
# ============================================================
if __name__ == "__main__":
    example_srgb = [
        ("warm",     np.array([0.8, 0.5, 0.2])),
        ("greenish", np.array([0.2, 0.7, 0.3])),
        ("bluish",   np.array([0.3, 0.3, 0.9])),
        ("purple",   np.array([0.7, 0.2, 0.6])),
        ("grey",     np.array([0.4, 0.4, 0.4])),
        ("red",      np.array([0.9, 0.1, 0.1])),
        ("green",    np.array([0.1, 0.9, 0.1])),
        ("blue",     np.array([0.1, 0.1, 0.9])),
    ]

    for name, srgb in example_srgb:
        validate_srgb_material(name, srgb, K=20,
                               output_dir=RESULTS / "reflectance",
                               show_plot=False)