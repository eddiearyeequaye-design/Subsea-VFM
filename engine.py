"""
engine.py  —  Subsea VFM Pro  |  Commercial Hybrid Engine  v4.1
================================================================
Generic field-validated architecture.  Target MAPE < 10%.
No field-specific hardcoding — all parameters supplied via config.

Modules:
  1  PVTEngine       DAK z-factor, Standing Bo/Rs, Beggs-Robinson, Lee-Kesler
  2  TubingModel     B&B BHP→WHP traverse, in-situ gas correction, friction_mult
  3  IPRModel        Vogel IPR, calibrated J, time-varying productivity
  4  ChokeModel      Gilbert critical + Sachdeva subcritical, Cd calibrated
  5  KalmanFilter    Sage-Husa AEKF, warm-start
  6  HydrateModel    Makogon + Hammerschmidt
  7  HybridML        Bootstrap GBR, physics-seeded synthetic training
  8  AutoCalibrator  Nelder-Mead, tunes friction_mult + pi_mult vs well tests
  9  WellAgent       Single-well integrator
 10  FieldController Multi-well manager + allocation

Primary VFM inputs (required at runtime):
  p_res    [psia]   reservoir pressure
  whp_psi  [psia]   wellhead pressure
  wht_k    [K]      wellhead temperature
  wc       [0-1]    water cut
  gor      [SCF/STB] producing GOR

Optional (improve accuracy):
  bhp_psi  [psia]   measured BHP
  p_choke_dn [psia] downstream choke P
  choke_size_64 [int] bean size 1/64 in
"""

import logging, warnings
import numpy as np
import pandas as pd
from copy import deepcopy
from scipy.optimize import brentq, minimize
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING,
                    format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("SubseaVFM")

PSI_TO_PA = 6894.757
PA_TO_PSI = 1.0 / PSI_TO_PA
STB_TO_M3 = 0.158987
G         = 9.80665


# ================================================================
# MODULE 1: PVT ENGINE
# ================================================================
class PVTEngine:
    """Black-oil PVT. API 20-55. DAK z-factor (valid to 15 000 psia)."""

    def __init__(self, api=35.0, sg_gas=0.65, wc=0.0, gor=500.0,
                 rho_brine=1025.0):
        self.api       = float(api)
        self.sg_gas    = float(sg_gas)
        self.wc        = float(np.clip(wc, 0.0, 0.999))
        self.gor       = float(max(gor, 1.0))
        self.rho_brine = float(rho_brine)
        self._sg_oil   = 141.5 / (131.5 + self.api)

    def z_factor(self, p_psi, t_k):
        ppc = 677.0 + 15.0*self.sg_gas - 37.5*self.sg_gas**2
        tpc = 168.0 + 325.0*self.sg_gas - 12.5*self.sg_gas**2
        ppr = p_psi / ppc
        tpr = (t_k * 1.8) / tpc
        def dak(rr):
            t1 = (0.3265-1.0700/tpr-0.5339/tpr**3+0.01569/tpr**4-0.05165/tpr**5)*rr
            t2 = (0.5475-0.7361/tpr+0.1844/tpr**2)*rr**2
            t3 = -0.1056*(0.7361/tpr-0.1844/tpr**2)*rr**5
            t4 = 0.6134*(1+0.7210*rr**2)*(rr**2/tpr**3)*np.exp(-0.7210*rr**2)
            return rr-0.27*ppr/(tpr*(1+t1+t2+t3+t4))
        try:
            rr = brentq(dak, 1e-6, 0.98, xtol=1e-6, maxiter=60)
            return max(float(0.27*ppr/(tpr*rr)), 0.05)
        except Exception:
            z = 1.0-(3.52*ppr)/(10**(0.9813*tpr))+(0.274*ppr**2)/(10**(0.8157*tpr))
            return max(float(z), 0.05)

    def gas_density(self, p_psi, t_k):
        mw = 28.97*self.sg_gas
        return (p_psi*PSI_TO_PA*mw*1e-3)/(self.z_factor(p_psi,t_k)*8.314*t_k)

    def gas_viscosity(self, p_psi, t_k):
        rho = self.gas_density(p_psi, t_k)
        mw  = 28.97*self.sg_gas; tr = t_k*1.8
        k   = (9.4+0.02*mw)*tr**1.5/(209.0+19.0*mw+tr)
        x   = 3.5+986.0/tr+0.01*mw; y = 2.4-0.2*x
        return max(float(1e-4*k*np.exp(x*(rho/1000)**y)), 0.005)

    def solution_gor(self, p_psi, t_k):
        tf = (t_k-273.15)*9/5+32
        rs = self.sg_gas*((p_psi/18.2+1.4)*10**(0.0125*self.api-0.00091*tf))**1.205
        return float(min(rs, self.gor))

    def oil_fvf(self, p_psi, t_k):
        tf = (t_k-273.15)*9/5+32; rs = self.solution_gor(p_psi,t_k)
        f  = rs*(self.sg_gas/self._sg_oil)**0.5+1.25*tf
        return float(max(0.972+1.47e-4*f**1.175, 1.0))

    def oil_density(self, p_psi, t_k):
        return float((self._sg_oil*1000+0.0136*self.solution_gor(p_psi,t_k)*self.sg_gas*1.225)
                     / self.oil_fvf(p_psi,t_k))

    def oil_viscosity(self, p_psi, t_k):
        tf   = (t_k-273.15)*9/5+32
        x    = 10**(3.0324-0.02023*self.api)*tf**(-1.163)
        mu_d = 10**x-1.0; rs = self.solution_gor(p_psi,t_k)
        return float(max(10.715*(rs+100)**(-0.515)*mu_d**(5.44*(rs+150)**(-0.338)), 0.1))

    def liquid_density(self, p_psi, t_k):
        return float((1-self.wc)*self.oil_density(p_psi,t_k)+self.wc*self.rho_brine)

    def liquid_viscosity(self, p_psi, t_k):
        return float((1-self.wc)*self.oil_viscosity(p_psi,t_k)+self.wc*0.9)

    def free_gor(self, p_psi, t_k):
        return float(max(self.gor-self.solution_gor(p_psi,t_k), 0.0))

    def mixture_density_column(self, p_avg_psi, t_k):
        rho_l = self.liquid_density(p_avg_psi, t_k)
        rho_g = self.gas_density(p_avg_psi, t_k)
        gor_at_p = self.free_gor(p_avg_psi, t_k) * (14.7/max(p_avg_psi,14.7))
        gvf = gor_at_p / max(gor_at_p + 5.615, 1e-6)
        gvf = float(np.clip(gvf, 0.0, 0.92))
        return float(gvf*rho_g + (1-gvf)*rho_l)

    def water_rate(self, q_liq):
        return float(q_liq*self.wc/max(1-self.wc, 1e-6))

    def gas_volume_fraction(self, p_psi, t_k, q_liq):
        bo    = self.oil_fvf(p_psi, t_k)
        q_l   = q_liq*STB_TO_M3/86400.0*bo
        free  = self.free_gor(p_psi,t_k)*q_liq
        z     = self.z_factor(p_psi,t_k)
        q_g   = free*0.028317/86400.0*(14.7/max(p_psi,1.0))*(t_k/288.7)*z
        return float(q_g/max(q_g+q_l, 1e-12))


# ================================================================
# MODULE 2: TUBING MODEL  (Beggs & Brill, BHP → WHP)
# ================================================================
class TubingModel:
    """B&B traverse BHP→WHP. friction_mult is primary calibration param."""

    SEGMENTS = 30

    # WC-adaptive friction multiplier lookup (default values, should be
    # calibrated per-field using AutoCalibrator for best accuracy).
    _WC_FM_TABLE = [(0.20, 0.99), (0.50, 0.83), (0.75, 0.81), (1.01, 1.44)]

    def __init__(self, depth, diameter, t_res=360.0, t_sea=277.0,
                 u_overall=8.0, deviation_survey=None):
        self.L     = float(depth)
        self.D     = float(diameter)
        self.T_res = float(t_res)
        self.T_sea = float(t_sea)
        self.U     = float(u_overall)
        self.area  = np.pi*(self.D/2)**2
        self.friction_mult = 1.0

        # ── Deviation survey ──────────────────────────────────
        self._survey = self._parse_survey(deviation_survey)
        self.tvd     = self._compute_tvd()
        self.dz      = self.tvd / self.SEGMENTS
        self._mean_inc_deg = (
            float(np.mean([inc for _, inc in self._survey]))
            if self._survey else 0.0
        )
        # Expected B&B accuracy penalty from deviation (0%→vertical, 25%→horizontal)
        self.deviation_penalty_pct = round(
            25.0 * np.sin(np.radians(self._mean_inc_deg)), 1
        )

    def _parse_survey(self, survey):
        if survey is None or len(survey) == 0:
            return None
        try:
            parsed = [(float(md), float(inc)) for md, inc in survey]
            return sorted(parsed, key=lambda x: x[0])
        except Exception:
            logger.warning("TubingModel: invalid deviation_survey — using vertical")
            return None

    def _compute_tvd(self):
        """Minimum curvature TVD from survey."""
        if self._survey is None:
            return self.L
        tvd = 0.0
        prev_md, prev_inc = self._survey[0]
        for md, inc in self._survey[1:]:
            ds = md - prev_md
            tvd += ds * (np.cos(np.radians(prev_inc)) + np.cos(np.radians(inc))) / 2.0
            prev_md, prev_inc = md, inc
        return max(float(tvd), 1.0)

    def _cos_inc_at(self, seg_idx):
        """Interpolated cos(inclination) at traverse segment."""
        if self._survey is None:
            return 1.0
        total_md = self._survey[-1][0] - self._survey[0][0]
        frac     = seg_idx / max(self.SEGMENTS - 1, 1)
        md_at    = self._survey[0][0] + frac * total_md
        mds  = [s[0] for s in self._survey]
        incs = [s[1] for s in self._survey]
        return float(np.cos(np.radians(np.interp(md_at, mds, incs))))

    def _flow_regime(self, lam, nfr):
        lam = np.clip(lam, 1e-6, 1-1e-6)
        l1=316.0*lam**0.302; l2=9.252e-4*lam**-2.4684
        l3=0.10*lam**-1.4516; l4=0.5*lam**6.738
        if lam<0.01 and nfr<l1:               return "seg"
        if lam>=0.01 and nfr<l2:              return "seg"
        if 0.01<=lam<0.4 and l2<=nfr<=l3:    return "trans"
        if lam<0.4 and l3<nfr<=l1:           return "int"
        if lam>=0.4 and l3<nfr<=l4:          return "int"
        return "dist"

    def _holdup(self, regime, lam, nfr):
        lam=np.clip(lam,1e-6,1-1e-6); nfr=max(nfr,1e-6)
        p={"seg":(0.980,0.4846,0.0868),"int":(0.845,0.5351,0.0173),
           "dist":(1.065,0.5824,0.0609),"trans":(0.845,0.5351,0.0173)}
        a,b,c=p[regime]
        hl=np.clip(a*lam**b/nfr**c, lam, 1.0)
        cc={"seg":(0.011,-3.768,3.539,-1.614),"int":(2.96,0.305,-0.4473,0.0978),
            "trans":(2.96,0.305,-0.4473,0.0978)}
        if regime in cc:
            d,e,f,g=cc[regime]; lnl=np.log(max(lam,1e-9))
            hl=np.clip(hl*max(1+d*(1.8-1/3)*(lnl-e*lnl**2-f)*g,0.0), lam, 1.0)
        return float(hl)

    def _ff(self, re, lam, hl):
        re=max(re,1.0)
        fn=0.3164/re**0.25 if re<1e5 else 0.0032+0.221/re**0.237
        y=lam/max(hl**2,1e-9)
        s=(np.log(2.2*y-1.2) if 1.0<y<1.2 else
           (lambda lny: lny/(-0.0523+3.182*lny-0.8725*lny**2+0.01853*lny**4))(np.log(max(y,1e-9))))
        return float(fn*np.exp(np.clip(s,-5.0,5.0))*self.friction_mult)

    def _dpdz(self, p_psi, t_k, q_liq, pvt, cos_inc=1.0):
        bo=pvt.oil_fvf(p_psi,t_k)
        ql=q_liq*STB_TO_M3/86400.0*bo
        z=pvt.z_factor(p_psi,t_k)
        qg=pvt.free_gor(p_psi,t_k)*q_liq*0.028317/86400.0*(14.7/max(p_psi,1))*(t_k/288.7)*z
        vsl=ql/self.area; vsg=qg/self.area; vm=vsl+vsg
        lam=vsl/max(vm,1e-9); nfr=vm**2/(G*self.D)
        reg=self._flow_regime(lam,nfr)
        hl=self._holdup(reg,lam,nfr)
        rl=pvt.liquid_density(p_psi,t_k); rg2=pvt.gas_density(p_psi,t_k)
        rm=hl*rl+(1-hl)*rg2
        mul=pvt.liquid_viscosity(p_psi,t_k); mug=pvt.gas_viscosity(p_psi,t_k)
        mum=hl*mul+(1-hl)*mug
        re=max(rm*vm*self.D/(mum*1e-3),1.0)
        ftp=self._ff(re,lam,hl)
        return float(rm*G*cos_inc+ftp*rm*vm**2/(2.0*self.D))

    def _dtdz(self, t_k, depth_m, m_dot):
        te=self.T_sea+(self.T_res-self.T_sea)*(depth_m/self.L)
        return float(-self.U*np.pi*self.D*(t_k-te)/max(m_dot*2200.0,500.0))

    def traverse(self, q_liq, bhp_psi, bht_k, pvt):
        p=bhp_psi*PSI_TO_PA; t=float(bht_k)
        m=max(q_liq*STB_TO_M3/86400.0*pvt.liquid_density(bhp_psi*0.85,bht_k*0.97),0.1)
        for i in range(self.SEGMENTS):
            cos_inc = self._cos_inc_at(i)
            dpdz=self._dpdz(p*PA_TO_PSI,t,q_liq,pvt,cos_inc=cos_inc)
            dtdz=self._dtdz(t, self.tvd*(1-i/self.SEGMENTS), m)
            p-=dpdz*self.dz; t+=dtdz*self.dz
            p=max(p,14.7*PSI_TO_PA); t=max(t,self.T_sea)
        return float(p*PA_TO_PSI)

    def solve_rate(self, bhp_psi, bht_k, whp_psi, pvt, q_min=50.0, q_max=100000.0):
        def res(q): return self.traverse(q,bhp_psi,bht_k,pvt)-whp_psi
        sq=np.logspace(np.log10(max(q_min,1)),np.log10(q_max),80)
        sr=[]
        for qs in sq:
            try: sr.append(float(res(qs)))
            except: sr.append(np.nan)
        for i in range(len(sq)-1):
            ra,rb=sr[i],sr[i+1]
            if np.isfinite(ra) and np.isfinite(rb) and ra*rb<0:
                return float(brentq(res,sq[i],sq[i+1],xtol=0.5,maxiter=150))
        finite=[(sq[i],abs(sr[i])) for i in range(len(sr)) if np.isfinite(sr[i])]
        if finite: return float(min(finite,key=lambda x:x[1])[0])
        raise VFMError("Traverse failed for all test flow rates.")

    def hydrostatic_whp(self, bhp_psi, pvt, t_k=None):
        t_k   = t_k or self.T_res
        rho_l = pvt.liquid_density(bhp_psi*0.7, t_k)
        whp   = max(bhp_psi*0.25, 50.0)
        for _ in range(8):
            p_avg = (bhp_psi+whp)/2.0
            rho_m = pvt.mixture_density_column(p_avg, t_k)
            dp    = rho_m*G*self.tvd*PA_TO_PSI
            whp_new = max(bhp_psi-dp, 14.7)
            if abs(whp_new-whp) < 2.0:
                whp = whp_new; break
            whp = whp_new
        return float(max(whp, 14.7))


# ================================================================
# MODULE 3: IPR MODEL  (Vogel)
# ================================================================
class IPRModel:
    def __init__(self, productivity_index=10.0):
        self.J       = float(productivity_index)
        self.pi_mult = 1.0

    @property
    def J_eff(self): return self.J * self.pi_mult

    def pwf_from_qoil(self, q_oil, p_res):
        q_max = self.J_eff * p_res / 1.8
        if q_oil >= q_max*0.999: return p_res*0.001
        ratio = q_oil/max(q_max,1.0)
        disc  = 0.04+3.2*ratio
        if disc<0: return p_res*0.001
        x = np.clip((-0.2+np.sqrt(disc))/1.6, 0.0, 1.0)
        return float(p_res*x)

    def qoil_from_pwf(self, pwf, p_res):
        q_max = self.J_eff*p_res/1.8
        x = np.clip(pwf/max(p_res,1.0), 0.0, 1.0)
        return float(max(q_max*(1-0.2*x-0.8*x**2), 0.0))

    def calibrate(self, q_oil, p_res, pwf):
        x = np.clip(pwf/max(p_res,1.0), 0.0, 1.0)
        vf = max(1-0.2*x-0.8*x**2, 0.01)
        self.J = (q_oil/vf)*1.8/max(p_res,1.0)
        logger.info(f"IPR calibrated: J={self.J:.2f} STB/D/psi")

    def aof(self, p_res):
        return float(self.J_eff * p_res / 1.8)


# ================================================================
# MODULE 4: CHOKE MODEL
# ================================================================
class ChokeModel:
    CRIT_RATIO = 0.546; C_GILBERT = 0.1

    def __init__(self, choke_size_64=32.0, cd=0.75):
        self.size_64=float(choke_size_64); self.cd=float(np.clip(cd,0.3,1.2))

    @property
    def _area_m2(self):
        return np.pi*((self.size_64/64.0)*0.0254/2)**2

    def has_drop(self, p_up, p_dn, min_dp=20.0):
        return p_dn is not None and (p_up-p_dn)>min_dp

    def estimate(self, p_up, pvt, t_up_k, p_dn=None):
        glr=pvt.gor+pvt.free_gor(p_up,t_up_k)
        if p_dn is None or (p_dn/max(p_up,1))<self.CRIT_RATIO:
            q=max(self.C_GILBERT*self.cd*p_up*self.size_64**1.89/max(glr,1)**0.546,0.0)
            return {"q_choke_stbd":round(q,1),"regime":"critical","cd":round(self.cd,3)}
        dp=max(p_up-p_dn,1.0)*PSI_TO_PA
        rl=pvt.liquid_density(p_up,t_up_k); rg=pvt.gas_density(p_up,t_up_k)
        free=pvt.free_gor(p_up,t_up_k)
        lam=1.0/(1.0+free*rl/max(rg*5.615,1.0))
        rm=lam*rl+(1-lam)*rg
        vm=self.cd*np.sqrt(2.0*dp/max(rm,1.0))
        q=max(vm*self._area_m2*lam*86400.0/STB_TO_M3,0.0)
        return {"q_choke_stbd":round(q,1),"regime":"subcritical","cd":round(self.cd,3)}


# ================================================================
# MODULE 5: KALMAN FILTER
# ================================================================
class KalmanFilter:
    def __init__(self, q0=0.5, r0=5.0, window=30):
        self._x=None; self.P=10.0; self.Q=float(q0); self.R=float(r0)
        self.b=0.95; self._w=window; self._inno=[]

    def update(self, z):
        if self._x is None: self._x=float(z); return self._x
        xp=self._x; Pp=self.P+self.Q; inn=z-xp
        K=Pp/(Pp+self.R); self._x=xp+K*inn; self.P=(1-K)*Pp
        self._inno.append(inn)
        if len(self._inno)>self._w: self._inno.pop(0)
        if len(self._inno)>=5:
            ev=float(np.var(self._inno))
            self.R=max(self.b*self.R+(1-self.b)*(ev+self.P-Pp),0.1)
            self.Q=max(self.b*self.Q+(1-self.b)*K*ev*K,0.01)
        return float(self._x)

    @property
    def status(self):
        if len(self._inno)<5: return {"healthy":True,"flag":"INIT","std":0.0,"bias":0.0}
        std=float(np.std(self._inno)); bias=float(np.mean(self._inno))
        ok=std<3*np.sqrt(self.R) and abs(bias)<2*np.sqrt(self.R)
        flag="OK" if ok else ("DRIFT" if abs(bias)>2*np.sqrt(self.R) else "NOISY")
        return {"healthy":ok,"flag":flag,"std":round(std,3),"bias":round(bias,3)}


# ================================================================
# MODULE 6: HYDRATE MODEL
# ================================================================
class HydrateModel:
    """
    Hydrate risk:  Makogon equilibrium + Hammerschmidt inhibitor dose.
    Wax onset:     Lindeloff (2001) WAT from API gravity.

    New in v4.1: assess() returns wax_status, wat_k, wax_delta_t_k.
    """
    MARGIN_K     = 3.0   # Hydrate subcooling safety margin [K]
    WAX_MARGIN_K = 5.0   # Wax proximity warning margin [K]

    def __init__(self, api=35.0, wax_onset_api_threshold=30.0):
        self.api           = float(api)
        self.wax_threshold = float(wax_onset_api_threshold)

    def equilibrium_temp(self,p): return float(283.15+0.036*np.sqrt(max(p,0))+0.0002*p)

    def wax_appearance_temp(self) -> float:
        """Lindeloff (2001): WAT_C = 65 - 0.8*(API-20). Valid API 15-50."""
        api = float(np.clip(self.api, 15.0, 50.0))
        return float((65.0 - 0.8*(api - 20.0)) + 273.15)

    def assess(self,p_psi,t_k):
        th  = self.equilibrium_temp(p_psi); d = t_k - th
        wat_k       = self.wax_appearance_temp()
        wax_delta_t = t_k - wat_k
        api_risk    = 1.0 if self.api < self.wax_threshold else 0.7
        if wax_delta_t < 0:
            wax_status = "RISK"
        elif wax_delta_t < self.WAX_MARGIN_K * api_risk:
            wax_status = "MONITOR"
        else:
            wax_status = "SAFE"
        return {
            "status":            "CRITICAL" if d<0 else "WARNING" if d<self.MARGIN_K else "SAFE",
            "t_hydrate_k":       round(th,2),
            "delta_t_k":         round(float(d),2),
            "meg_dose_vol_pct":  self._dose(p_psi,t_k,62.07,1297.0),
            "meoh_dose_vol_pct": self._dose(p_psi,t_k,32.04,2335.0),
            "wax_status":        wax_status,
            "wat_k":             round(wat_k,2),
            "wax_delta_t_k":     round(float(wax_delta_t),2),
        }

    def _dose(self,p,t,mw,kh):
        dT=max(self.equilibrium_temp(p)-t+self.MARGIN_K,0.0)
        return 0.0 if dT==0 else round((kh*dT)/(mw+kh*dT)*100,1)


# ================================================================
# MODULE 7: HYBRID ML LAYER
# ================================================================
class HybridML:
    N_BOOT = 50
    N_EST  = 100
    MIN_N  = 100
    FEATS  = ["bhp", "whp", "wht_k", "wc", "gor", "choke_size_64", "depth", "t_res"]

    def __init__(self):
        self.models=[]; self.scaler=StandardScaler()
        self.is_trained=False; self.meta={}

    def _new_m(self):
        return GradientBoostingRegressor(n_estimators=self.N_EST,
            learning_rate=0.05,max_depth=4,subsample=0.8,
            min_samples_leaf=3,random_state=np.random.randint(0,99999))

    def generate_training_data(self, tubing: TubingModel, ipr: IPRModel,
                               pvt: PVTEngine,
                               p_res_range=(3500,5500),
                               wc_range=(0.0,0.90),
                               gor_range=(500,1000),
                               n=3000) -> pd.DataFrame:
        pvt_c = deepcopy(pvt)
        records = []
        logger.info(f"Generating {n} synthetic training records...")
        for _ in range(n):
            p_res   = np.random.uniform(*p_res_range)
            wc      = np.random.uniform(*wc_range)
            gor     = np.random.uniform(*gor_range)
            pvt_c.wc=wc; pvt_c.gor=gor
            pwf_frac = np.random.uniform(0.30, 0.90)
            pwf      = p_res * pwf_frac
            q_oil    = ipr.qoil_from_pwf(pwf, p_res)
            q_liq    = q_oil / max(1-wc, 0.001)
            if q_liq < 50: continue
            bht      = tubing.T_sea + (tubing.T_res-tubing.T_sea)*0.95
            wht      = tubing.T_sea + (tubing.T_res-tubing.T_sea)*0.12
            choke_64 = float(np.random.choice([8,12,16,20,24,28,32,40,48,56,64]))
            try:
                whp = tubing.traverse(q_liq, pwf, bht, pvt_c)
                whp += np.random.normal(0, whp*0.02)
                if 14.7 < whp < p_res*0.85:
                    records.append({
                        "bhp": pwf + np.random.normal(0, pwf*0.01),
                        "whp": whp,
                        "wht_k": wht + np.random.normal(0, 2),
                        "wc":  wc, "gor": gor,
                        "depth": tubing.L, "t_res": tubing.T_res,
                        "choke_size_64": choke_64, "q_true": q_liq,
                    })
            except Exception:
                pass
        df = pd.DataFrame(records)
        logger.info(f"Generated {len(df)} valid records from {n} attempts")
        return df

    def train(self, df: pd.DataFrame, source="synthetic"):
        missing=set(self.FEATS+["q_true"])-set(df.columns)
        if missing: raise ValueError(f"Missing: {missing}")
        df=df.dropna(subset=self.FEATS+["q_true"])
        if len(df)<self.MIN_N: raise ValueError(f"Need ≥{self.MIN_N} records")
        X=self.scaler.fit_transform(df[self.FEATS].values); y=df["q_true"].values; n=len(df)
        self.models=[]
        for _ in range(self.N_BOOT):
            idx=np.random.choice(n,n,replace=True)
            m=self._new_m(); m.fit(X[idx],y[idx]); self.models.append(m)
        self.is_trained=True; self.meta={"source":source,"n":n,"n_boot":self.N_BOOT}
        logger.info(f"ML trained | {self.meta}")

    def predict(self, bhp, whp, wht_k, wc, gor, choke_size_64, depth, t_res):
        if not self.is_trained: return None
        x=self.scaler.transform([[bhp, whp, wht_k, wc, gor, choke_size_64, depth, t_res]])
        preds=np.clip([m.predict(x)[0] for m in self.models],0,None)
        return {"p10":round(float(np.percentile(preds,10)),1),
                "p50":round(float(np.percentile(preds,50)),1),
                "p90":round(float(np.percentile(preds,90)),1),
                "mean":round(float(np.mean(preds)),1),
                "cv":round(float(np.std(preds)/max(np.mean(preds),1)*100),2)}


# ================================================================
# MODULE 8: AUTO-CALIBRATOR
# ================================================================
class AutoCalibrator:
    BOUNDS={"friction_mult":(0.3,3.0),"pi_mult":(0.2,5.0)}

    def __init__(self):
        self.friction_mult=1.0; self.pi_mult=1.0
        self.calibrated=False; self.calibration_mape=None; self.n_tests=0

    def calibrate(self, well, well_tests: pd.DataFrame) -> dict:
        req={"whp","wht_k","wc","gor","q_liq"}
        missing=req-set(well_tests.columns)
        if missing: raise ValueError(f"Missing: {missing}")
        if len(well_tests)<2: raise ValueError("Need ≥2 well tests")
        recs=well_tests.to_dict("records"); self.n_tests=len(recs)

        def obj(params):
            fm,pm=params
            fm=np.clip(fm,*self.BOUNDS["friction_mult"])
            pm=np.clip(pm,*self.BOUNDS["pi_mult"])
            well.tubing.friction_mult=fm; well.ipr.pi_mult=pm
            errs=[]
            for rec in recs:
                try:
                    r=well.run_vfm(
                        p_res        = rec.get("p_res", rec.get("bhp",3500)*1.25),
                        whp_psi      = rec["whp"],
                        wht_k        = rec["wht_k"],
                        wc           = rec["wc"],
                        gor          = rec["gor"],
                        bhp_psi      = rec.get("bhp"),
                        p_choke_dn   = rec.get("p_choke_dn"),
                        choke_size_64= rec.get("choke_size_64"),
                    )
                    q=r["q_total_stbd"]
                    if q>0: errs.append(abs(q-rec["q_liq"])/max(rec["q_liq"],1.0))
                except: errs.append(1.0)
            return float(np.mean(errs)) if errs else 1.0

        res=minimize(obj,x0=[1.0,1.0],method="Nelder-Mead",
                     options={"xatol":0.005,"fatol":0.005,"maxiter":600,"disp":False})
        self.friction_mult=float(np.clip(res.x[0],*self.BOUNDS["friction_mult"]))
        self.pi_mult=float(np.clip(res.x[1],*self.BOUNDS["pi_mult"]))
        self.calibrated=True; self.calibration_mape=round(res.fun*100,2)
        well.tubing.friction_mult=self.friction_mult; well.ipr.pi_mult=self.pi_mult
        logger.info(f"Calibrated | MAPE={self.calibration_mape}% | fm={self.friction_mult:.3f} | pm={self.pi_mult:.3f}")
        return {"mape_pct":self.calibration_mape,"friction_mult":round(self.friction_mult,4),
                "pi_mult":round(self.pi_mult,4),"n_tests":self.n_tests,"converged":bool(res.success)}


# ================================================================
# MODULE 9: WELL AGENT
# ================================================================
class WellAgent:
    """
    Single-well integrator.  Zero hardcoded field values.

    Required config keys:
      depth [m], diameter [m], t_res [K], t_sea [K],
      api, sg_gas, productivity_index [STB/D/psi]
    Optional:
      rho_brine, u_overall, wc, gor, choke_size_64, choke_cd
      deviation_survey          : list of [md_m, inclination_deg] pairs
      wax_onset_api_threshold   : default 30.0

    New in v4.1 — run_vfm() returns:
      physics_uncertainty       : WC+deviation-adaptive ±% bounds on physics estimate
      deviation_penalty_pct     : expected accuracy loss from well deviation
      tvd_m                     : computed true vertical depth
      wax_status / wat_k / wax_delta_t_k
    """

    def __init__(self, name: str, config: dict):
        self.name=name; self._cfg=config
        self.pvt=PVTEngine(api=config["api"],sg_gas=config["sg_gas"],
                           wc=config.get("wc",0.0),gor=config.get("gor",500.0),
                           rho_brine=config.get("rho_brine",1025.0))
        self.tubing=TubingModel(
            depth=config["depth"],diameter=config["diameter"],
            t_res=config["t_res"],t_sea=config["t_sea"],
            u_overall=config.get("u_overall",8.0),
            deviation_survey=config.get("deviation_survey",None),
        )
        self.ipr=IPRModel(productivity_index=config["productivity_index"])
        self.choke=ChokeModel(choke_size_64=config.get("choke_size_64",32.0),
                              cd=config.get("choke_cd",0.75))
        self.kf_whp=KalmanFilter(); self.kf_bhp=KalmanFilter()
        self.kf_pr=KalmanFilter(q0=0.1,r0=2.0)
        self.hydrate=HydrateModel(
            api=config["api"],
            wax_onset_api_threshold=config.get("wax_onset_api_threshold",30.0),
        )
        self.ml=HybridML(); self.calibrator=AutoCalibrator()
        self._ml_ready=False
        self._depth=self.tubing.tvd   # always TVD
        self._t_res=config["t_res"]

    def _physics_uncertainty(self, q_physics: float, wc: float) -> dict:
        """
        WC-adaptive ±% bounds from B&B correlation error studies.
          WC < 0.30  → ±8%  |  WC 0.30-0.60 → ±12%  |  WC > 0.60 → ±18%
        Deviation adds additional % based on mean well inclination.
        """
        wc        = float(np.clip(wc, 0.0, 0.999))
        base_pct  = 8.0 if wc < 0.30 else (12.0 if wc < 0.60 else 18.0)
        dev_pct   = self.tubing.deviation_penalty_pct
        total_pct = base_pct + dev_pct
        p10 = round(max(q_physics*(1-total_pct/100), 0), 1)
        p50 = round(q_physics, 1)
        p90 = round(q_physics*(1+total_pct/100), 1)
        return {
            "p10": p10, "p50": p50, "p90": p90,
            "half_range_pct":          round(total_pct, 1),
            "wc_component_pct":        round(base_pct, 1),
            "deviation_component_pct": round(dev_pct, 1),
        }

    def initialize_ml(self, p_res_range=None, wc_range=None,
                      gor_range=None, n=3000):
        p_res_range = p_res_range or (self.ipr.J*1.8*0.6, self.ipr.J*1.8*1.3)
        p_res_range = (max(p_res_range[0],500), min(p_res_range[1],15000))
        wc_range    = wc_range  or (0.0, 0.90)
        gor_range   = gor_range or (max(self.pvt.gor*0.5,100),
                                    min(self.pvt.gor*2.0,5000))
        df = self.ml.generate_training_data(
            self.tubing, self.ipr, self.pvt,
            p_res_range=p_res_range, wc_range=wc_range,
            gor_range=gor_range, n=n
        )
        if len(df) >= HybridML.MIN_N:
            self.ml.train(df, source="physics_synthetic")
            self._ml_ready=True
            logger.info(f"[{self.name}] ML initialized | {self.ml.meta}")
        else:
            logger.warning(f"[{self.name}] Insufficient training data ({len(df)})")

    def train_ml_on_field_data(self, df: pd.DataFrame, source: str = "field_data"):
        """
        Train ML on real field data.
        Expects columns: bhp, whp, wht_k, wc, gor, choke_size_64, q_true
        depth and t_res are added from well config automatically.
        """
        df = df.copy()
        df["depth"] = self._depth
        df["t_res"] = self._t_res
        self.ml.train(df, source=source)
        self._ml_ready = True
        logger.info(f"[{self.name}] ML trained on {len(df)} field records")

    def calibrate(self, well_tests: pd.DataFrame) -> dict:
        return self.calibrator.calibrate(self, well_tests)

    def calibrate_ipr(self, q_oil, p_res, pwf):
        self.ipr.calibrate(q_oil, p_res, pwf)

    def run_vfm(self, p_res, whp_psi, wht_k, wc, gor,
                bhp_psi=None, p_choke_dn=None, choke_size_64=None) -> dict:
        self.pvt.wc=float(np.clip(wc,0.0,0.999))
        self.pvt.gor=float(max(gor,1.0))
        if choke_size_64 is not None: self.choke.size_64=float(choke_size_64)

        pr_c  = self.kf_pr.update(p_res)
        whp_c = self.kf_whp.update(whp_psi)
        whp_s = self.kf_whp.status
        bhp_c = None; bhp_s={"healthy":True,"flag":"NOT_PROVIDED"}
        if bhp_psi is not None:
            bhp_c=self.kf_bhp.update(bhp_psi); bhp_s=self.kf_bhp.status

        bht_k = wht_k + self.tubing.tvd*0.025

        if bhp_c is not None and bhp_c > whp_c:
            q_nodal = self.tubing.solve_rate(bhp_c, bht_k, whp_c, self.pvt)
            bhp_used = bhp_c
        else:
            def nodal_res(q_liq):
                q_oil = q_liq*max(1-self.pvt.wc,0.001)
                pwf   = self.ipr.pwf_from_qoil(q_oil, pr_c)
                if pwf <= whp_c: return -(whp_c-pwf)
                return self.tubing.traverse(q_liq, pwf, bht_k, self.pvt)-whp_c

            sq=np.logspace(np.log10(50),np.log10(100000),30)
            sr=[]
            for qs in sq:
                try: sr.append(float(nodal_res(qs)))
                except: sr.append(np.nan)
            bracket=None
            for i in range(len(sq)-1):
                ra,rb=sr[i],sr[i+1]
                if np.isfinite(ra) and np.isfinite(rb) and ra*rb<0:
                    bracket=(float(sq[i]),float(sq[i+1])); break
            if bracket:
                q_nodal=float(brentq(nodal_res,bracket[0],bracket[1],xtol=0.5,maxiter=150))
            else:
                bhp_est=max(whp_c*2.5, pr_c*0.45)
                q_nodal=self.ipr.qoil_from_pwf(bhp_est,pr_c)/max(1-self.pvt.wc,0.001)
                logger.debug(f"[{self.name}] Nodal fallback q={q_nodal:.0f}")
            bhp_used=self.ipr.pwf_from_qoil(q_nodal*max(1-self.pvt.wc,0.001),pr_c)

        q_choke=None; choke_result=None
        if self.choke.has_drop(whp_c,p_choke_dn):
            choke_result=self.choke.estimate(whp_c,self.pvt,wht_k,p_choke_dn)
            q_choke=choke_result["q_choke_stbd"]

        if q_choke is not None:
            div=abs(q_nodal-q_choke)/max(q_nodal,1.0)
            agr="GOOD" if div<0.10 else "MARGINAL" if div<0.25 else "POOR"
            wn=0.60 if agr=="GOOD" else 0.75 if agr=="MARGINAL" else 1.0
            q_combined=wn*q_nodal+(1-wn)*q_choke
        else:
            agr="CHOKE_OPEN"; q_combined=q_nodal; wn=1.0

        q_final=q_combined; w_phys=1.0; ml_result=None
        if self._ml_ready:
            ml_result=self.ml.predict(
                bhp          = bhp_c if bhp_c else bhp_used,
                whp          = whp_c,
                wht_k        = wht_k,
                wc           = self.pvt.wc,
                gor          = self.pvt.gor,
                choke_size_64= self.choke.size_64,
                depth        = self._depth,
                t_res        = self._t_res,
            )
            if ml_result:
                q_ml   = ml_result["p50"]
                div_ml = abs(q_combined-q_ml)/max(q_combined,1.0)
                if self.ml.meta.get("source","") == "field_data":
                    w_max = 0.25
                else:
                    wc_pen = float(np.clip((self.pvt.wc-0.30)/0.40, 0.0, 1.0))
                    w_max  = 0.85 - 0.50*wc_pen
                w_phys  = float(np.clip(w_max - 0.15*div_ml, 0.05, w_max))
                q_final = w_phys*q_combined + (1-w_phys)*q_ml

        phys_unc = self._physics_uncertainty(q_combined, self.pvt.wc)
        q_water=self.pvt.water_rate(q_final)
        q_oil=max(q_final-q_water,0.0)
        q_gas=q_oil*self.pvt.gor/1000.0
        gvf=self.pvt.gas_volume_fraction(whp_c,wht_k,q_final)
        hydrate=self.hydrate.assess(whp_c,wht_k)

        return {
            "well":              self.name,
            "q_total_stbd":      round(q_final,1),
            "q_oil_stbd":        round(q_oil,1),
            "q_water_stbd":      round(q_water,1),
            "q_gas_mscfd":       round(q_gas,2),
            "q_nodal_stbd":      round(q_nodal,1),
            "q_choke_stbd":      round(q_choke,1) if q_choke else None,
            "choke_agreement":   agr,
            "gvf_insitu":        round(gvf,4),
            "bhp_used_psi":      round(bhp_used,1),
            "whp_filtered_psi":  round(whp_c,1),
            "p_res_filtered_psi":round(pr_c,1),
            "wc":                round(self.pvt.wc,3),
            "gor_scf_stb":       round(self.pvt.gor,0),
            "wht_k":             round(wht_k,2),
            "wht_c":             round(wht_k-273.15,2),
            "whp_sensor_flag":   whp_s["flag"],
            "bhp_sensor_flag":   bhp_s["flag"],
            "hydrate_status":    hydrate["status"],
            "hydrate_delta_t_k": hydrate["delta_t_k"],
            "meg_dose_vol_pct":  hydrate["meg_dose_vol_pct"],
            "meoh_dose_vol_pct": hydrate["meoh_dose_vol_pct"],
            "calibrated":        self.calibrator.calibrated,
            "calibration_mape":  self.calibrator.calibration_mape,
            "physics_uncertainty":  phys_unc,
            "physics_weight_pct":  round(w_phys*100,1),
            "ml_uncertainty":      ml_result,
            "wax_status":          hydrate.get("wax_status","SAFE"),
            "wat_k":               hydrate.get("wat_k",0),
            "wax_delta_t_k":       hydrate.get("wax_delta_t_k",0),
            "tvd_m":               round(self.tubing.tvd,1),
            "deviation_penalty_pct": self.tubing.deviation_penalty_pct,
        }


# ================================================================
# MODULE 10: FIELD CONTROLLER
# ================================================================
class FieldController:
    def __init__(self):
        self.wells: dict[str,WellAgent]={}

    def add_well(self, name: str, config: dict):
        self.wells[name]=WellAgent(name,config)

    def run_field(self, telemetry: dict) -> list:
        results=[]
        for name,data in telemetry.items():
            if name not in self.wells: continue
            try: results.append(self.wells[name].run_vfm(**data))
            except VFMError as e:
                logger.error(str(e))
                results.append({"well":name,"error":str(e),
                                "q_total_stbd":0,"q_oil_stbd":0,
                                "q_water_stbd":0,"q_gas_mscfd":0})
        return results

    def allocate(self, results: list, fiscal_oil_stbd: float=None) -> dict:
        valid=[r for r in results if "error" not in r]
        to=sum(r["q_oil_stbd"] for r in valid)
        tw=sum(r["q_water_stbd"] for r in valid)
        tg=sum(r["q_gas_mscfd"] for r in valid)
        recon=fiscal_oil_stbd/to if fiscal_oil_stbd and to>0 else 1.0
        return {"total_oil_stbd":round(to,0),"total_water_stbd":round(tw,0),
                "total_gas_mscfd":round(tg,2),
                "field_wc":round(tw/max(to+tw,1),3),
                "reconciliation_factor":round(recon,4),
                "wells_online":len(valid),"wells_faulted":len(results)-len(valid),
                "hydrate_alerts":[r["well"] for r in valid if r.get("hydrate_status")!="SAFE"],
                "wax_alerts":    [r["well"] for r in valid if r.get("wax_status") in ("RISK","MONITOR")],
                "well_allocation":[{"well":r["well"],
                                    "q_oil_alloc":round(r["q_oil_stbd"]*recon,1),
                                    "q_water_alloc":round(r["q_water_stbd"]*recon,1),
                                    "contribution_pct":round(r["q_oil_stbd"]/max(to,1)*100,1)}
                                   for r in valid]}


# ================================================================
# EXCEPTION
# ================================================================
class VFMError(Exception):
    pass
