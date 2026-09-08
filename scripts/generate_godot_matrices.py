"""
Regenerate the FIXED reduction matrices that the Godot fluorescence addon
hardcodes, from the *current* col-sum-normalised XYZU basis on the 300-799 nm
grid (matching validate_fluorescence_v4.py). Emits GDScript / GLSL literals.

Fixed matrices produced (all pure functions of the CMF basis + Table-1 fit):
  - C4           (4x4)   : C = (S4_gauss^T S4_gauss)^-1        (paper Eq. 15)
  - Rx,Ry,Rz,Ru (4x4 ea) : Rk[:,:,k] = S4^T diag(Sb4[:,k]) Sb4 (paper Eq. 7)
  - T_U          (3,)    : S4[:,3] @ Sb3   -> UV_WEIGHTS         (Method 2)
  - LIN_TO_NORM  (3x3)   : linear sRGB   -> normalised XYZ
  - NORM_TO_LIN  (3x3)   : normalised XYZ -> linear sRGB
  - T_G          (5x4)   : canonical 0/1 grouping (paper Eq. 29) -- for reference

Usage:  python scripts/generate_godot_matrices.py
"""
import numpy as np

from _paths import CMF

np.set_printoptions(suppress=True, precision=8, floatmode="fixed")

# ---------------------------------------------------------------------------
# 1. Col-sum-normalised XYZU basis on 300-799 nm  (matches v4)
# ---------------------------------------------------------------------------
WL = np.arange(300, 800, 1)                          # 500 points
_data = np.loadtxt(CMF / "xyzu.csv", delimiter=",")
assert _data.shape == (500, 5), f"unexpected xyzu.csv shape {_data.shape}"
_raw = _data[:, 1:5]                                 # (500, 4)  raw XYZU CMF

col_sums4 = _raw.sum(axis=0)                          # (4,)
S4  = _raw / col_sums4                                # (500, 4) normalised
Sb4 = S4 @ np.linalg.pinv(S4.T @ S4)                  # (500, 4) dual, S4^T Sb4 = I4

S3  = S4[:, :3]                                       # (500, 3) XYZ sub-basis
Sb3 = S3 @ np.linalg.pinv(S3.T @ S3)                  # (500, 3) dual

# ---------------------------------------------------------------------------
# 2. Gaussian basis (paper Table 1) + transfer matrix T_G
#    IMPORTANT: the paper's canonical 0/1 grouping (Eq. 29) assumes S is the
#    *standard*-scale CMF the Gaussians were fit to. Our pipeline works in the
#    col-sum-NORMALISED S4, so T_G must be the dense least-squares fit that
#    reconstructs the normalised basis (S4 ~= G @ T_G), matching v4.
# ---------------------------------------------------------------------------
_aG = np.array([0.35087, 1.141263, 1.024335, 1.915863, 1.0])
_mG = np.array([443.412226, 596.813847, 560.186336, 447.268188, 382.535501])
_sG = np.array([20.838149, 33.276659, 43.898132, 23.542626, 57.432550])

def _g1d(wl, a, m, s):
    return a * np.exp(-0.5 * ((wl - m) / s) ** 2)

G = np.column_stack([_g1d(WL, _aG[i], _mG[i], _sG[i]) for i in range(5)])  # (500,5)

T_G, _, _, _ = np.linalg.lstsq(G, S4, rcond=None)     # (5,4) dense fit to normalised S4
T_G_T = T_G.T                                         # (4,5)

S4_gauss = G @ T_G                                    # (500,4) Gaussian approx of S4
C4 = np.linalg.pinv(S4_gauss.T @ S4_gauss)            # (4,4)   Eq. 15

# ---------------------------------------------------------------------------
# 3. Reduced reflectance basis  Rk[:,:,k] = S4^T diag(Sb4[:,k]) Sb4   (Eq. 7)
# ---------------------------------------------------------------------------
Rk = np.stack([S4.T @ np.diag(Sb4[:, k]) @ Sb4 for k in range(4)], axis=-1)  # (4,4,4)
Rx, Ry, Rz, Ru = Rk[:, :, 0], Rk[:, :, 1], Rk[:, :, 2], Rk[:, :, 3]

# ---------------------------------------------------------------------------
# 4. U-channel albedo, Method 2 (overlap projection)  T_U = S4[:,3] @ Sb3
# ---------------------------------------------------------------------------
T_U = S4[:, 3] @ Sb3                                  # (3,)

# ---------------------------------------------------------------------------
# 5. sRGB <-> normalised-XYZ conversions (D65 sRGB primaries)
# ---------------------------------------------------------------------------
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
col_sums3   = _raw[:, :3].sum(axis=0)                 # xyz col sums (300-799)
col_y       = col_sums3[1]
NORM_TO_STD = np.diag(col_sums3 / col_y)             # normalised -> standard XYZ
STD_TO_NORM = np.diag(col_y / col_sums3)             # standard   -> normalised XYZ
LIN_TO_NORM = STD_TO_NORM @ LIN_TO_XYZ               # linear sRGB   -> normalised XYZ
NORM_TO_LIN = XYZ_TO_LIN  @ NORM_TO_STD              # normalised XYZ -> linear sRGB

# ---------------------------------------------------------------------------
# 6. Self-checks
# ---------------------------------------------------------------------------
print("=== SELF-CHECKS ===")
print(f"S4 col sums (should be ~1)         : {S4.sum(axis=0).round(6)}")
print(f"max |S4^T Sb4 - I4|                : {np.abs(S4.T @ Sb4 - np.eye(4)).max():.2e}")
print(f"max |S3^T Sb3 - I3|                : {np.abs(S3.T @ Sb3 - np.eye(3)).max():.2e}")
print(f"Gaussian fit max |S4 - G@T_G|      : {np.abs(S4 - S4_gauss).max():.6f}  (dense lstsq, want small)")
print(f"C4 symmetric? max |C4 - C4^T|      : {np.abs(C4 - C4.T).max():.2e}")
# sum_k Rk applied to a flat rho=1: reduced R vs I4. Non-zero error is the
# intrinsic reduction error (a flat spectrum is not exactly in the 4D dual span).
R_white = Rk.sum(axis=-1)
print(f"sum_k Rk vs I4 (info: reduction err): max {np.abs(R_white - np.eye(4)).max():.2e}")
print(f"T_U (UV_WEIGHTS)                   : {T_U.round(7)}")
print(f"LIN_TO_NORM @ NORM_TO_LIN = I3?    : max err {np.abs(LIN_TO_NORM @ NORM_TO_LIN - np.eye(3)).max():.2e}")

# ---------------------------------------------------------------------------
# 7. Emit literals
# ---------------------------------------------------------------------------
def gd_flat(mat, name, per_row=None):
    """GDScript row-major flat float array."""
    flat = mat.flatten()
    per_row = per_row or mat.shape[1]
    lines = []
    for i in range(0, len(flat), per_row):
        chunk = ", ".join(f"{v: .8f}" for v in flat[i:i + per_row])
        lines.append("\t" + chunk + ",")
    body = "\n".join(lines).rstrip(",")
    return f"const {name}: Array[float] = [\n{body}\n]"

def glsl_flat(mat, name):
    flat = mat.flatten()
    n = len(flat)
    vals = ",\n\t".join(", ".join(f"{v: .8f}" for v in flat[i:i+mat.shape[1]])
                        for i in range(0, n, mat.shape[1]))
    return f"const float {name}[{n}] = float[{n}](\n\t{vals}\n);"

def glsl_mat4_colmajor(mat, name):
    """GLSL mat4 literal: each vec4 is a COLUMN (column-major)."""
    cols = []
    for c in range(4):
        col = mat[:, c]
        cols.append("\tvec4(" + ", ".join(f"{v: .8e}" for v in col) + ")")
    return f"const mat4 {name} = mat4(\n" + ",\n".join(cols) + "\n);"

print("\n\n=== GLSL: reduced reflectance basis (column-major mat4) ===")
for M, nm in [(Rx, "Rx"), (Ry, "Ry"), (Rz, "Rz"), (Ru, "Ru")]:
    print(glsl_mat4_colmajor(M, nm))

print("\n=== GLSL: correction matrix C (row-major float[16]) ===")
print(glsl_flat(C4, "C"))

print("\n=== GLSL: UV_WEIGHTS (Method 2 overlap) ===")
print("const vec3 UV_WEIGHTS = vec3(" + ", ".join(f"{v: .7f}" for v in T_U) + ");")

print("\n=== GLSL: sRGB <-> normalised-XYZ (row-major float[9]) ===")
print(glsl_flat(LIN_TO_NORM, "LIN_SRGB_TO_NORM_XYZ"))
print(glsl_flat(NORM_TO_LIN, "NORM_XYZ_TO_LIN_SRGB"))

print("\n=== GDScript: for fluorescent_controller.gd (row-major flat arrays) ===")
print(gd_flat(T_G_T, "TG_T", per_row=5) + "  # (4x5)")
print()
print(gd_flat(T_G,   "TG",   per_row=4) + "  # (5x4)")
print()
print(gd_flat(C4,    "C_MAT", per_row=4) + "  # (4x4)")
