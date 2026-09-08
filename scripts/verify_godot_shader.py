"""
End-to-end check of the Godot port.  The direct-lighting shader evaluates
    out = R*c + Fbar*((I - R)*c)
with the baked reflectance basis R_k and the controller's Fbar.  For identical
inputs (a 4-D albedo rho_xyzu and a 4-D light c) it must reproduce the
validated single-bounce output (R + Fbar(I - R))*c of validate_fluorescence.py,
for both the reduced reflectance R = sum rho_k R_k and the diagonal diag(rho).

Every residual is expected at machine precision (~1e-16); the asserts allow
1e-12.
"""
import numpy as np
from scipy.special import erf

from _paths import CMF

WL = np.arange(300, 800, 1)
_raw = np.loadtxt(CMF / "xyzu.csv", delimiter=",")[:, 1:5]
S4  = _raw / _raw.sum(axis=0)
Sb4 = S4 @ np.linalg.pinv(S4.T @ S4)
Sb3 = S4[:, :3] @ np.linalg.pinv(S4[:, :3].T @ S4[:, :3])
Rk  = np.stack([S4.T @ np.diag(Sb4[:, k]) @ Sb4 for k in range(4)], axis=-1)

_aG = np.array([0.35087, 1.141263, 1.024335, 1.915863, 1.0])
_mG = np.array([443.412226, 596.813847, 560.186336, 447.268188, 382.535501])
_sG = np.array([20.838149, 33.276659, 43.898132, 23.542626, 57.432550])
G = np.column_stack([_aG[i]*np.exp(-0.5*((WL-_mG[i])/_sG[i])**2) for i in range(5)])
T_G, *_ = np.linalg.lstsq(G, S4, rcond=None)
C4 = np.linalg.pinv((G @ T_G).T @ (G @ T_G))

def _alpha_max(mu_e, se):
    return 1.0/(np.sqrt(np.pi/2)*se*(1+erf(mu_e/(np.sqrt(2)*se))))

# ---- v4 analytic F_bar (reference) ----
def _gprod(a1,m1,s1,a2,m2,s2):
    s1s,s2s=s1**2,s2**2; d=s1s+s2s
    return (a1*a2*np.exp(-0.5*(m1-m2)**2/d),(m1*s2s+m2*s1s)/d,s1*s2/np.sqrt(d))
def fbar_v4(alpha_bar,mu_a,sa,mu_e,se):
    alpha=alpha_bar*_alpha_max(mu_e,se); Fc=np.zeros((5,5))
    for j in range(5):
        for k in range(5):
            aa,ma,ssa=_gprod(1.0,mu_a,sa,_aG[k],_mG[k],_sG[k])
            ae,me,sse=_gprod(1.0,mu_e,se,_aG[j],_mG[j],_sG[j])
            Fc[j,k]=np.pi*alpha*aa*ae*ssa*sse*(1-erf((ma-me)/np.sqrt(2*(ssa**2+sse**2))))
    return T_G.T @ Fc @ T_G @ C4

# ---- fixed-controller F_bar (swapped indices, as edited in fluorescent_controller.gd) ----
def fbar_controller(alpha_bar,mu_a,sa,mu_e,se):
    alpha=alpha_bar*_alpha_max(mu_e,se); Fc=np.zeros((5,5))
    def gpm(m1,s1,m2,s2): return (m1*s2**2+m2*s1**2)/(s1**2+s2**2)
    def gps(s1,s2): return np.sqrt((s1**2*s2**2)/(s1**2+s2**2))
    for j in range(5):
        for k in range(5):
            a_jk=alpha*_aG[k]*_aG[j]*np.exp(-(mu_a-_mG[k])**2/(2*(sa**2+_sG[k]**2)))\
                                   *np.exp(-(mu_e-_mG[j])**2/(2*(se**2+_sG[j]**2)))
            mu_abs=gpm(mu_a,sa,_mG[k],_sG[k]); s_abs=gps(sa,_sG[k])
            mu_em =gpm(mu_e,se,_mG[j],_sG[j]); s_em =gps(se,_sG[j])
            Fc[j,k]=np.pi*a_jk*s_abs*s_em*(1-erf((mu_abs-mu_em)/np.sqrt(2*(s_abs**2+s_em**2))))
    return T_G.T @ Fc @ T_G @ C4   # row-major; new upload applies it directly

# ---- inputs ----
rho = np.array([0.30, 0.55, 0.20, 0.12])     # some 4D albedo
c_light = np.array([1.00, 1.00, 1.00, 0.60]) # some XYZU light (with UV)
params = dict(alpha_bar=0.8, mu_a=430.0, sa=40.0, mu_e=560.0, se=40.0)

Fb_v4  = fbar_v4(**params)
Fb_ctl = fbar_controller(**params)
I4 = np.eye(4)

e_fbar = np.linalg.norm(Fb_ctl-Fb_v4)
print(f"||Fbar_controller - Fbar_v4||                 : {e_fbar:.3e}")

# REDUCED reflectance
R_red = np.einsum("ijk,k->ij", Rk, rho)
out_v4_red    = (R_red + Fb_v4 @ (I4 - R_red)) @ c_light
out_godot_red = R_red @ c_light + Fb_ctl @ ((I4 - R_red) @ c_light)
e_red = np.linalg.norm(out_godot_red-out_v4_red)
print(f"REDUCED : ||out_godot - out_v4||         : {e_red:.3e}")

# DIAGONAL reflectance (v9 hybrid first bounce)
R_diag = np.diag(rho)
out_v4_diag    = (R_diag + Fb_v4 @ (I4 - R_diag)) @ c_light
out_godot_diag = R_diag @ c_light + Fb_ctl @ ((I4 - R_diag) @ c_light)
e_diag = np.linalg.norm(out_godot_diag-out_v4_diag)
print(f"DIAGONAL: ||out_godot - out_v4||         : {e_diag:.3e}")

print(f"\nsample outgoing XYZU (reduced) : {out_godot_red.round(4)}")
print(f"sample outgoing XYZU (diagonal): {out_godot_diag.round(4)}")

assert e_fbar < 1e-12
assert e_red < 1e-12
assert e_diag < 1e-12
print("\nchecks passed")
