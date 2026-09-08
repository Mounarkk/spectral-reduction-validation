"""
Resolve the Fbar index / transpose convention between the validated Python
pipeline (validate_fluorescence.py) and the Godot controller
(fluorescent_controller.gd), by comparing the EFFECTIVE linear operator each
one applies to an incident colour.

Four operators are built for one test material:
  1. Fbar_bf       : S4^T F_spec Sb4                  (spectral ground truth)
  2. Fbar_v4       : build_F_bar_analytic            (validated analytic model)
  3. ORIGINAL    : the controller as first written ; F_circ indices absorption<->j,
                   emission<->k, uploaded through a Projection whose columns are
                   Fbar rows (an implicit transpose).
  4. CORRECTED   : F_circ indices swapped to the paper convention (absorption<->k,
                   emission<->j), uploaded directly.  This is what ships.

Two real bugs were found this way and fixed in fluorescent_controller.gd:
the flipped F_circ indices and the transposed Projection upload.  The asserts
at the end encode the outcome: CORRECTED reproduces Fbar_v4 to machine
precision, ORIGINAL does not.
"""
import numpy as np
from scipy.special import erf

from _paths import CMF

WL = np.arange(300, 800, 1)
_data = np.loadtxt(CMF / "xyzu.csv", delimiter=",")
_raw = _data[:, 1:5]
S4  = _raw / _raw.sum(axis=0)
Sb4 = S4 @ np.linalg.pinv(S4.T @ S4)

_aG = np.array([0.35087, 1.141263, 1.024335, 1.915863, 1.0])
_mG = np.array([443.412226, 596.813847, 560.186336, 447.268188, 382.535501])
_sG = np.array([20.838149, 33.276659, 43.898132, 23.542626, 57.432550])
def _g1d(wl, a, m, s): return a * np.exp(-0.5 * ((wl - m) / s) ** 2)
G = np.column_stack([_g1d(WL, _aG[i], _mG[i], _sG[i]) for i in range(5)])
T_G, *_ = np.linalg.lstsq(G, S4, rcond=None)
C4 = np.linalg.pinv((G @ T_G).T @ (G @ T_G))

# --- test material ---
mu_a, sa, mu_e, se, alpha_bar = 430.0, 40.0, 560.0, 40.0, 0.8

def _alpha_max(mu_e, se):
    return 1.0 / (np.sqrt(np.pi/2)*se*(1 + erf(mu_e/(np.sqrt(2)*se))))

# ---------------------------------------------------------------------------
# 1. Brute-force spectral F_bar (ground truth)
# ---------------------------------------------------------------------------
li, lo = WL[np.newaxis, :], WL[:, np.newaxis]
Ga = np.exp(-0.5*((li-mu_a)/sa)**2)
Ge = np.exp(-0.5*((lo-mu_e)/se)**2)
Fsp = Ga*Ge*(lo > li)
amax = 1.0/Fsp.sum(axis=0).max()
Fbar_spec = alpha_bar*amax*Fsp
Fbar_bf = S4.T @ Fbar_spec @ Sb4

# ---------------------------------------------------------------------------
# 2. v4 analytic (reference convention: absorption<->k, emission<->j)
# ---------------------------------------------------------------------------
def _gprod(a1,m1,s1,a2,m2,s2):
    s1s,s2s = s1**2,s2**2; d=s1s+s2s
    return (a1*a2*np.exp(-0.5*(m1-m2)**2/d),(m1*s2s+m2*s1s)/d,s1*s2/np.sqrt(d))
def _Fcirc_v4(alpha,aK,mK,sK,aJ,mJ,sJ):
    aa,ma,ssa=_gprod(1.0,mu_a,sa,aK,mK,sK)
    ae,me,sse=_gprod(1.0,mu_e,se,aJ,mJ,sJ)
    return np.pi*alpha*aa*ae*ssa*sse*(1-erf((ma-me)/np.sqrt(2*(ssa**2+sse**2))))
alpha = alpha_bar*_alpha_max(mu_e,se)
Fc_v4 = np.zeros((5,5))
for j in range(5):
    for k in range(5):
        Fc_v4[j,k]=_Fcirc_v4(alpha,_aG[k],_mG[k],_sG[k],_aG[j],_mG[j],_sG[j])
Fbar_v4 = T_G.T @ Fc_v4 @ T_G @ C4

# ---------------------------------------------------------------------------
# 3. Controller F-circle (CURRENT: absorption<->j, emission<->k)  -> row-major F_bar
# ---------------------------------------------------------------------------
def controller_fbar(swap):
    Fc = np.zeros((5,5))
    for j in range(5):
        for k in range(5):
            if swap:  # match v4: absorption<->k, emission<->j
                ja, ka = k, j
            else:     # current .gd: absorption<->j, emission<->k
                ja, ka = j, k
            a_jk = alpha*_aG[j]*_aG[k]*np.exp(-(mu_a-_mG[ja])**2/(2*(sa**2+_sG[ja]**2)))\
                                      *np.exp(-(mu_e-_mG[ka])**2/(2*(se**2+_sG[ka]**2)))
            mj=(mu_a*_sG[ja]**2+_mG[ja]*sa**2)/(sa**2+_sG[ja]**2); sj=np.sqrt((sa**2*_sG[ja]**2)/(sa**2+_sG[ja]**2))
            mk=(mu_e*_sG[ka]**2+_mG[ka]*se**2)/(se**2+_sG[ka]**2); sk=np.sqrt((se**2*_sG[ka]**2)/(se**2+_sG[ka]**2))
            Fc[j,k]=np.pi*a_jk*sj*sk*(1-erf((mj-mk)/np.sqrt(2*(sj**2+sk**2))))
    return T_G.T @ Fc @ T_G @ C4   # row-major F_bar (as controller stores f_bar_total)

# The original controller built a Projection with col_k = Fbar row_k; in the
# shader `f_bar_matrix * v` then applied the TRANSPOSE of the row-major Fbar.
results = {}
for swap in (False, True):
    Fbar_ctrl = controller_fbar(swap)
    shader_op = Fbar_ctrl.T          # effect of the col_k = row_k upload
    tag = "CORRECTED (paper idx, direct upload)" if swap else "ORIGINAL (.gd idx, transposed upload)"
    print(f"\n--- {tag} ---")
    print(f"  ||shader_op   - Fbar_v4|| = {np.linalg.norm(shader_op - Fbar_v4):.4e}")
    print(f"  ||Fbar_ctrl      - Fbar_v4|| = {np.linalg.norm(Fbar_ctrl - Fbar_v4):.4e}")
    print(f"  ||shader_op^T  - Fbar_v4|| = {np.linalg.norm(shader_op.T - Fbar_v4):.4e}")
    results[swap] = (Fbar_ctrl, shader_op)

err_model = np.linalg.norm(Fbar_v4 - Fbar_bf)
print(f"\n||Fbar_v4 - Fbar_bf|| = {err_model:.4e}  ; analytic Gaussian model vs. spectral "
      f"ground truth.  This is the model's intrinsic approximation error, not a bug; "
      f"its perceptual cost is what validate_fluorescence.py measures in DeltaE2000.")
print(f"Fbar_v4 =\n{np.array2string(Fbar_v4, precision=5, suppress_small=True)}")

Fc_corr, op_orig = results[True][0], results[False][1]
assert np.linalg.norm(Fc_corr - Fbar_v4) < 1e-12   # corrected controller reproduces Fbar_v4
assert np.linalg.norm(op_orig - Fbar_v4) > 1e-3    # the original convention did not
assert err_model < 0.5                              # regression guard on the Gaussian fit
print("\nchecks passed")
