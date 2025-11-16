# -*- coding: utf-8 -*-
"""
compare.py
----------
Simulation du modèle SCS-like HSM/Guinot à partir d'une série de pluie
issue du fichier ../02_Data/PQ_BV_Cloutasse.csv, en intégrant un terme
de déplétion s_a(t) sur le réservoir d'abstraction h_a, lié à l'ETP
SAFRAN journalière.

L'ETP agit UNIQUEMENT sur h_a (réservoir d'abstraction).
s_a(t) = ETP_effective(t) = min(ETP_pot(t), h_a(t))

On calcule :
  - états h_a, h_s, h_r (cumul de runoff)
  - flux instantanés : q(t), infiltration(t), r(t)
  - un bilan de masse global
  - Q_mod(t) = r(t) * A_BV_M2

On suppose que Q_ls dans le CSV est en L/s -> converti en m³/s.
"""

from __future__ import annotations
from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# SciPy pour le calage
try:
    from scipy.optimize import minimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ======================================================================
# 1. Modèle SCS-like HSM/Guinot avec ETP agissant sur h_a (s_a)
# ======================================================================

def run_scs_hsm(
    dt: float,
    p_rate: np.ndarray,
    etp_rate: np.ndarray | None = None,
    i_a: float = 2e-3,
    s: float = 0.02,
    k_infiltr: float = 1e-5,
    k_seepage: float = 1e-6,
    h_a_init: float = 0.0,
    h_s_init: float = 0.0,
) -> dict:
    """
    Modèle SCS-like HSM continu avec états h_a, h_s et ruissellement r(t).
    """
    p_rate = np.nan_to_num(np.asarray(p_rate, dtype=float), nan=0.0)
    nt = len(p_rate)

    if etp_rate is None:
        etp_rate = np.zeros(nt, dtype=float)
    else:
        etp_rate = np.nan_to_num(np.asarray(etp_rate, dtype=float), nan=0.0)
        if len(etp_rate) != nt:
            raise ValueError("etp_rate doit avoir la même longueur que p_rate")

    t_vector = np.array([i * dt for i in range(nt + 1)], dtype=float)

    # Etats
    h_a = np.zeros(nt + 1, dtype=float)
    h_s = np.zeros(nt + 1, dtype=float)
    h_r_cum = np.zeros(nt + 1, dtype=float)
    h_a[0] = h_a_init
    h_s[0] = h_s_init

    # Flux
    p_store = np.zeros(nt + 1, dtype=float)
    q = np.zeros(nt, dtype=float)
    infil = np.zeros(nt, dtype=float)
    r_rate = np.zeros(nt, dtype=float)
    seep_loss = np.zeros(nt, dtype=float)
    sa_loss = np.zeros(nt, dtype=float)

    for n in range(nt):
        p = float(p_rate[n])
        p_store[n] = p

        # 1) ETP sur h_a
        h_a_0 = h_a[n]
        etp_pot = float(etp_rate[n]) * dt
        etp_eff = min(etp_pot, h_a_0)
        h_a_after_etp = h_a_0 - etp_eff
        sa_loss[n] = etp_eff

        # 2) Abstraction Ia + pluie nette q
        h_a_temp = h_a_after_etp + p * dt
        if h_a_temp < i_a:
            q_n = 0.0
            h_a_next = h_a_temp
        else:
            q_n = (h_a_temp - i_a) / dt
            h_a_next = i_a

        q[n] = q_n
        h_a[n + 1] = h_a_next

        # 3) Infiltration potentielle HSM
        h_s_begin = h_s[n]
        X_begin = 1.0 - h_s_begin / s
        if X_begin <= 0.0:
            X_begin = 1e-12
        X_end = 1.0 / (1.0 / X_begin + k_infiltr * dt / s)
        h_s_end = (1.0 - X_end) * s
        infil_pot = (h_s_end - h_s_begin) / dt

        # 4) Limitation par l'eau dispo
        infil_n = max(0.0, min(infil_pot, q_n))
        r_n = max(0.0, q_n - infil_n)

        infil[n] = infil_n
        r_rate[n] = r_n

        # 5) Sol avant seepage
        h_s_temp = h_s_begin + infil_n * dt

        # 6) Seepage profond
        if k_seepage > 0.0:
            h_s_after_seep = h_s_temp * math.exp(-k_seepage * dt)
            seep = h_s_temp - h_s_after_seep
        else:
            h_s_after_seep = h_s_temp
            seep = 0.0

        seep_loss[n] = seep
        h_s[n + 1] = h_s_after_seep

        # 7) Ruissellement cumulé
        h_r_cum[n + 1] = h_r_cum[n] + r_n * dt

    # ------------------------------------------------------------------
    # Bilan de masse
    # ------------------------------------------------------------------
    P_tot = float(np.nansum(p_rate) * dt)          # m
    R_tot = float(np.nansum(r_rate) * dt)          # m
    Seep_tot = float(np.nansum(seep_loss))         # m
    ET_tot = float(np.nansum(sa_loss))             # m
    d_h_a = h_a[-1] - h_a[0]
    d_h_s = h_s[-1] - h_s[0]
    delta_storage = d_h_a + d_h_s

    mb_error = P_tot - (R_tot + Seep_tot + ET_tot + delta_storage)

    mass_balance = {
        "P_tot_m": P_tot,
        "R_tot_m": R_tot,
        "Seep_tot_m": Seep_tot,
        "ET_tot_m": ET_tot,
        "Delta_storage_m": delta_storage,
        "Closure_error_m": mb_error,
        "Closure_error_mm": mb_error * 1000.0,
        "Relative_error_%": 100.0 * mb_error / P_tot if P_tot > 0 else np.nan,
    }

    return {
        "t": t_vector,
        "p": p_store,
        "h_a": h_a,
        "h_s": h_s,
        "h_r": h_r_cum,
        "q": q,
        "infil": infil,
        "r_rate": r_rate,
        "seep_loss": seep_loss,
        "sa_loss": sa_loss,
        "mass_balance": mass_balance,
    }


# ======================================================================
# 2. Lecture pluie + Q_ls (en L/s) -> Q_ls en m³/s
# ======================================================================

def read_rain_series_from_csv(csv_name: str, dt: float = 300.0):
    """
    Lit ../02_Data/csv_name (séparateur ';'), extrait :
      - dateP   : datetimes
      - P_mm    : pluie (mm / 5 min)  (NA -> 0)
      - Q_ls    : débit mesuré (supposé en L/s -> converti en m³/s)

    Renvoie (time_index, P_mm, p_rate_m_per_s, q_ls_m3s)
    """
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir.parent / "02_Data"
    csv_path = data_dir / csv_name

    df = pd.read_csv(
        csv_path,
        sep=";",
        na_values=["NA", "NaN", "", -9999, -9999.0],
    )

    time_series = pd.to_datetime(df["dateP"])
    time_index = pd.DatetimeIndex(time_series)

    rain_5min_mm = df["P_mm"].astype(float).fillna(0.0).to_numpy()
    p_rate = rain_5min_mm * 1e-3 / dt  # mm/5min -> m/s

    q_ls = None
    if "Q_ls" in df.columns:
        q_raw = df["Q_ls"].astype(float).to_numpy()
        q_ls = q_raw / 1000.0  # L/s -> m³/s

    return time_index, rain_5min_mm, p_rate, q_ls


# ======================================================================
# 3. Lecture ETP SAFRAN journalière + mise à l'échelle 5 min
# ======================================================================

def read_etp_series_for_time_index(
    etp_csv_name: str,
    time_index: pd.DatetimeIndex,
) -> np.ndarray:
    """
    Lit ../02_Data/etp_csv_name (séparateur ';') contenant :
      - DATE : AAAAMMJJ
      - ETP  : mm/jour

    Renvoie etp_rate (m/s) sur la grille temporelle time_index (NA -> 0).
    """
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir.parent / "02_Data"
    etp_path = data_dir / etp_csv_name

    df_etp = pd.read_csv(
        etp_path,
        sep=";",
        na_values=["NA", "NaN", "", -9999, -9999.0],
    )

    date_col = None
    for c in ["DATE", "Date", "date"]:
        if c in df_etp.columns:
            date_col = c
            break
    if date_col is None:
        raise ValueError("Impossible de trouver la colonne date dans le fichier ETP.")

    etp_col = None
    for c in ["ETP", "etp", "Etp"]:
        if c in df_etp.columns:
            etp_col = c
            break
    if etp_col is None:
        raise ValueError("Impossible de trouver la colonne ETP dans le fichier ETP.")

    df_etp[date_col] = pd.to_datetime(df_etp[date_col].astype(str), format="%Y%m%d")
    df_etp[date_col] = df_etp[date_col].dt.normalize()
    df_etp[etp_col] = df_etp[etp_col].astype(float).fillna(0.0)

    etp_dict = dict(zip(df_etp[date_col].values, df_etp[etp_col].values))

    dates_only = time_index.normalize()
    etp_mm_per_day = np.array([etp_dict.get(d, 0.0) for d in dates_only], dtype=float)

    etp_rate = etp_mm_per_day * 1e-3 / 86400.0  # mm/j -> m/s
    return etp_rate


# ======================================================================
# 4. Bilan de masse – impression
# ======================================================================

def print_mass_balance(mb: dict):
    print("\n=== Bilan de masse sur la période ===")
    print(f"P_tot          = {mb['P_tot_m']*1000:.1f} mm")
    print(f"Ruissellement  = {mb['R_tot_m']*1000:.1f} mm")
    print(f"Seepage profond= {mb['Seep_tot_m']*1000:.1f} mm")
    print(f"ETP effective (s_a sur h_a) = {mb['ET_tot_m']*1000:.1f} mm")
    print(f"ΔStock (Ia+sol)            = {mb['Delta_storage_m']*1000:.2f} mm")
    print(f"Erreur de fermeture         = {mb['Closure_error_mm']:.3f} mm "
          f"({mb['Relative_error_%']:.3f} %)")


# ======================================================================
# 5. Calage des paramètres sur Q_ls (RMSE)
# ======================================================================

def calibrate_scs_hsm(
    dt: float,
    p_rate: np.ndarray,
    etp_rate: np.ndarray,
    q_obs: np.ndarray,
    A_bv_m2: float,
    i_a_init: float,
    s_init: float,
    k_infiltr_init: float,
    k_seepage_init: float,
):
    if not HAS_SCIPY:
        raise RuntimeError("SciPy n'est pas disponible : calage impossible.")

    p_rate = np.asarray(p_rate, dtype=float)
    etp_rate = np.asarray(etp_rate, dtype=float)
    q_obs = np.asarray(q_obs, dtype=float)

    mask = ~np.isnan(q_obs)
    if mask.sum() == 0:
        raise ValueError("q_obs ne contient aucune valeur valide (toutes NaN ?)")

    def objective(theta: np.ndarray) -> float:
        i_a, s, k_inf, k_seep = theta
        if i_a <= 0 or s <= 0 or k_inf <= 0 or k_seep < 0:
            return 1e6

        res = run_scs_hsm(
            dt=dt,
            p_rate=p_rate,
            etp_rate=etp_rate,
            i_a=i_a,
            s=s,
            k_infiltr=k_inf,
            k_seepage=k_seep,
            h_a_init=0.0,
            h_s_init=0.0,
        )

        r_rate = res["r_rate"]          # <<< plus de [: -1]
        q_mod = r_rate * A_bv_m2        # m³/s

        diff = q_mod[mask] - q_obs[mask]
        return float(np.sqrt(np.mean(diff**2)))

    x0 = np.array([i_a_init, s_init, k_infiltr_init, k_seepage_init], dtype=float)
    bounds = [
        (1e-4, 0.05),   # i_a
        (1e-3, 0.20),   # s
        (1e-7, 1e-3),   # k_infiltr
        (0.0,  1e-3),   # k_seepage
    ]

    result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
    return {
        "opt_params": {
            "i_a": result.x[0],
            "s": result.x[1],
            "k_infiltr": result.x[2],
            "k_seepage": result.x[3],
        },
        "rmse_opt": float(result.fun),
        "success": bool(result.success),
        "message": result.message,
    }


# ======================================================================
# 6. Script principal
# ======================================================================

def main():
    dt = 300.0  # 5 min
    csv_rain = "PQ_BV_Cloutasse.csv"
    csv_etp = "ETP_SAFRAN_J.csv"

    # Aire du bassin : 0.8 km²
    A_BV_M2 = 0.8 * 1e6

    time_index, rain_5min_mm, p_rate, q_obs = read_rain_series_from_csv(csv_rain, dt)
    etp_rate = read_etp_series_for_time_index(csv_etp, time_index)

    # Paramètres initiaux
    i_a = 2e-3
    s = 0.02
    k_infiltr = 1e-5
    k_seepage = 1e-6

    if q_obs is not None and HAS_SCIPY:
        print("Lancement du calage des paramètres sur Q_ls (RMSE)...")
        calib = calibrate_scs_hsm(
            dt, p_rate, etp_rate, q_obs, A_BV_M2,
            i_a, s, k_infiltr, k_seepage,
        )
        print("\n=== Résultats du calage ===")
        print(f"Succès      : {calib['success']}")
        print(f"Message     : {calib['message']}")
        print(f"RMSE_opt    : {calib['rmse_opt']:.3f} m³/s")
        for k, v in calib["opt_params"].items():
            print(f"  {k} = {v:.6g}")

        i_a = calib["opt_params"]["i_a"]
        s = calib["opt_params"]["s"]
        k_infiltr = calib["opt_params"]["k_infiltr"]
        k_seepage = calib["opt_params"]["k_seepage"]

    res = run_scs_hsm(dt, p_rate, etp_rate,
                      i_a=i_a, s=s,
                      k_infiltr=k_infiltr, k_seepage=k_seepage)

    # Etats : nt+1 -> alignés sur nt par [: -1]
    h_a = res["h_a"][:-1]
    h_s = res["h_s"][:-1]
    h_r = res["h_r"][:-1]
    # Flux : déjà de longueur nt
    r_rate = res["r_rate"]

    runoff_5min_mm = r_rate * dt * 1000.0

    base_dir = Path(__file__).resolve().parent
    plots_dir = base_dir.parent / "03_Plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path_main = plots_dir / "runoff_scs_hsm_PQ_BV_Cloutasse.png"
    out_path_Q = plots_dir / "runoff_vs_Qls_PQ_BV_Cloutasse.png"

    # Figure principale
    fig, ax1 = plt.subplots(figsize=(12, 6))
    line_ha, = ax1.plot(time_index, h_a, color="grey", label="h in i_a")
    line_hs, = ax1.plot(time_index, h_s, color="green", label="h in soil")
    line_hr, = ax1.plot(time_index, h_r, color="red", label="h runoff (cum.)")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("h (m)")
    ax1.grid(True)

    ax2 = ax1.twinx()
    line_P, = ax2.plot(time_index, rain_5min_mm,
                       color="blue", alpha=0.5, label="P (mm / 5 min)")
    line_Ri, = ax2.plot(time_index, runoff_5min_mm,
                        color="orange", alpha=0.8, linestyle="--",
                        label="runoff (mm / 5 min)")
    ax2.set_ylabel("P, runoff (mm / 5 min)")

    lines = [line_ha, line_hs, line_hr, line_P, line_Ri]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left")

    fig.suptitle(
        "SCS-like runoff-infiltration model (HSM)\n"
        "(données PQ_BV_Cloutasse + ETP SAFRAN, ETP sur h_a uniquement)"
    )
    fig.tight_layout()
    fig.savefig(out_path_main, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Figure principale enregistrée dans : {out_path_main.resolve()}")

    # Comparaison Q_mod vs Q_obs
    if q_obs is not None:
        q_mod = r_rate * A_BV_M2  # m³/s

        fig2, ax_q = plt.subplots(figsize=(12, 4))
        ax_q.plot(time_index, q_mod, label="Q modèle (ruissellement)", linewidth=1.0)
        ax_q.plot(time_index, q_obs, label="Q_ls observé", linewidth=1.0, alpha=0.7)
        ax_q.set_xlabel("Date")
        ax_q.set_ylabel("Débit (m³/s)")
        ax_q.grid(True)
        ax_q.legend()
        fig2.suptitle("Comparaison débit modèle SCS-like vs Q_ls observé")
        fig2.tight_layout()
        fig2.savefig(out_path_Q, dpi=300, bbox_inches="tight")
        plt.show()
        print(f"Figure Q_mod vs Q_ls enregistrée dans : {out_path_Q.resolve()}")

        mask = ~np.isnan(q_obs)
        if mask.sum() > 0:
            diff = q_mod[mask] - q_obs[mask]
            rmse = float(np.sqrt(np.mean(diff**2)))
            denom = np.sum((q_obs[mask] - np.mean(q_obs[mask]))**2)
            nse = 1.0 - np.sum(diff**2) / denom if denom > 0 else np.nan
            print(f"RMSE (Q_mod, Q_obs) = {rmse:.2f} m³/s")
            print(f"NSE  (Q_mod, Q_obs) = {nse:.3f}")
    else:
        print("Pas de colonne Q_ls : pas de comparaison débit.")

    print_mass_balance(res["mass_balance"])


if __name__ == "__main__":
    main()
