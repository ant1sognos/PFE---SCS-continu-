# -*- coding: utf-8 -*-
"""
compare_guinot.py
-----------------
Simulation du modèle SCS-like HSM/Guinot à partir :
  - d'une série P(t), Q_ls(t) issue de ../02_Data/PQ_BV_Cloutasse.csv
  - d'une série ETP journalière SAFRAN issue de ../02_Data/ETP_SAFRAN_J.csv

Hypothèses :
  - L'ETP agit UNIQUEMENT sur h_a (réservoir d'abstraction).
    s_a(t) = ETP_effective(t) = min(ETP_pot(t), h_a(t)).
  - Infiltration potentielle = loi HSM de Guinot (X_begin / X_end).
  - Infiltration limitée par l'eau disponible à la surface :
        v <= q + h_r/dt
  - h_r(t) = lame de ruissellement cumulée à la surface (comme h_r_vector chez Guinot).

On calcule :
  - états : h_a(t), h_s(t), h_r(t)
  - flux instantanés : p(t), q(t), infiltration(t), r(t)
  - bilan de masse global
  - Q_mod(t) = r(t) * A_BV_M2 (débit m³/s) pour comparaison visuelle avec Q_ls.

AUCUN CALAGE DE PARAMÈTRES : on fixe (Ia, S, k_infiltr, k_seepage) à la main.

Created on Tue Nov 18 15:41:17 2025
@author: asognos
"""

from __future__ import annotations
from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ======================================================================
# 1. Modèle SCS-like HSM/Guinot avec ETP sur h_a
# ======================================================================

def run_scs_hsm_guinot_numba(
    dt: float,
    p_rate: np.ndarray,
    etp_rate: np.ndarray,
    i_a: float,
    s: float,
    k_infiltr: float,
    k_seepage: float,
    h_a_init: float,
    h_s_init: float,
    h_r_init: float,
) -> tuple:
    """
    Version Numba JIT (compiled). Retourne tous les vecteurs en tuple.
    """
    nt = len(p_rate)
    
    # États (nt+1)
    h_a = np.zeros(nt + 1, dtype=np.float64)
    h_s = np.zeros(nt + 1, dtype=np.float64)
    h_r = np.zeros(nt + 1, dtype=np.float64)
    
    h_a[0] = h_a_init
    h_s[0] = h_s_init
    h_r[0] = h_r_init
    
    # Flux (nt)
    p_store = np.zeros(nt, dtype=np.float64)
    q = np.zeros(nt, dtype=np.float64)
    infil = np.zeros(nt, dtype=np.float64)
    r_rate = np.zeros(nt, dtype=np.float64)
    sa_loss = np.zeros(nt, dtype=np.float64)
    seep_loss = np.zeros(nt, dtype=np.float64)
    
    for n in range(nt):
        p = p_rate[n]
        etp = etp_rate[n]
        p_store[n] = p
        
        # 1) ETP sur h_a
        h_a_0 = h_a[n]
        etp_pot = etp * dt
        etp_eff = min(etp_pot, h_a_0)
        h_a_after_etp = h_a_0 - etp_eff
        sa_loss[n] = etp_eff
        
        # 2) Réservoir Ia
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
        h_r_begin = h_r[n]
        water_avail_rate = q_n + h_r_begin / dt
        if water_avail_rate < 0.0:
            water_avail_rate = 0.0
        
        infil_n = max(0.0, min(infil_pot, water_avail_rate))
        infil[n] = infil_n
        
        # 5) Mise à jour du sol
        h_s_temp = h_s_begin + infil_n * dt
        
        # 6) Seepage profond
        if k_seepage > 0.0:
            h_s_after_seep = h_s_temp * np.exp(-k_seepage * dt)
            seep = h_s_temp - h_s_after_seep
        else:
            h_s_after_seep = h_s_temp
            seep = 0.0
        
        h_s[n + 1] = h_s_after_seep
        seep_loss[n] = seep
        
        # 7) Réservoir de surface h_r
        h_r[n + 1] = h_r_begin + (q_n - infil_n) * dt
        if h_r[n + 1] < 0.0:
            h_r[n + 1] = 0.0
    
    # 8) Ruissellement instantané
    for n in range(nt):
        dh = h_r[n + 1] - h_r[n]
        if dh < 0.0:
            dh = 0.0
        r_rate[n] = dh / dt
    
    return h_a, h_s, h_r, p_store, q, infil, r_rate, sa_loss, seep_loss


def run_scs_hsm_guinot(
    dt: float,
    p_rate: np.ndarray,
    etp_rate: np.ndarray | None = None,
    i_a: float = 2e-3,
    s: float = 0.02,
    k_infiltr: float = 1,
    k_seepage: float = 1e-3,
    h_a_init: float = 0.0,
    h_s_init: float = 0.0,
    h_r_init: float = 0.0,
) -> dict:
    """
    Wrapper appelant la version Numba.
    """
    p_rate = np.nan_to_num(np.asarray(p_rate, dtype=float), nan=0.0)
    nt = len(p_rate)

    if etp_rate is None:
        etp_rate = np.zeros(nt, dtype=float)
    else:
        etp_rate = np.nan_to_num(np.asarray(etp_rate, dtype=float), nan=0.0)
        if len(etp_rate) != nt:
            raise ValueError("etp_rate doit avoir la même longueur que p_rate")

    h_a, h_s, h_r, p_store, q, infil, r_rate, sa_loss, seep_loss = run_scs_hsm_guinot_numba(
        dt=dt,
        p_rate=p_rate,
        etp_rate=etp_rate,
        i_a=i_a,
        s=s,
        k_infiltr=k_infiltr,
        k_seepage=k_seepage,
        h_a_init=h_a_init,
        h_s_init=h_s_init,
        h_r_init=h_r_init,
    )
    
    t = np.array([i * dt for i in range(nt + 1)], dtype=float)
    
    P_tot = float(np.nansum(p_rate) * dt)
    R_tot = float(h_r[-1])
    Seep_tot = float(np.nansum(seep_loss))
    ET_tot = float(np.nansum(sa_loss))

    d_h_a = h_a[-1] - h_a[0]
    d_h_s = h_s[-1] - h_s[0]
    delta_storage = d_h_a + d_h_s

    closure = P_tot - (R_tot + Seep_tot + ET_tot + delta_storage)

    mass_balance = {
        "P_tot_m": P_tot,
        "R_tot_m": R_tot,
        "Seep_tot_m": Seep_tot,
        "ET_tot_m": ET_tot,
        "Delta_storage_m": delta_storage,
        "Closure_error_m": closure,
        "Closure_error_mm": closure * 1000.0,
        "Relative_error_%": 100.0 * closure / P_tot if P_tot > 0 else np.nan,
    }

    return {
        "t": t,
        "p": p_store,
        "h_a": h_a,
        "h_s": h_s,
        "h_r": h_r,
        "q": q,
        "infil": infil,
        "r_rate": r_rate,
        "sa_loss": sa_loss,
        "seep_loss": seep_loss,
        "mass_balance": mass_balance,
    }


# ======================================================================
# 2. Lecture pluie + Q_ls (L/s) -> Q_ls (m³/s)
# ======================================================================

def read_rain_series_from_csv(csv_name: str, dt: float = 300.0):
    """
    Lit ../02_Data/csv_name (séparateur ';'), extrait :
      - dateP   : datetimes
      - P_mm    : pluie (mm / 5 min)  (NA -> 0)
      - Q_ls    : débit mesuré (L/s) -> m³/s

    Renvoie : (time_index, P_mm, p_rate_m_per_s, q_ls_m3s)
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
        # si tu veux vraiment des m³/s, décommente :
        # q_ls = q_raw / 1000.0
        q_ls = q_raw

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
        raise ValueError("Colonne date absente dans le fichier ETP.")

    etp_col = None
    for c in ["ETP", "etp", "Etp"]:
        if c in df_etp.columns:
            etp_col = c
            break
    if etp_col is None:
        raise ValueError("Colonne ETP absente dans le fichier ETP.")

    df_etp[date_col] = pd.to_datetime(df_etp[date_col].astype(str), format="%Y%m%d")
    df_etp[date_col] = df_etp[date_col].dt.normalize()
    df_etp[etp_col] = df_etp[etp_col].astype(float).fillna(0.0)

    etp_dict = dict(zip(df_etp[date_col].values, df_etp[etp_col].values))

    dates_only = time_index.normalize()
    etp_mm_per_day = np.array([etp_dict.get(d, 0.0) for d in dates_only], dtype=float)

    etp_rate = etp_mm_per_day * 1e-3 / 86400.0  # mm/j -> m/s
    return etp_rate


# ======================================================================
# 4. Impression bilan de masse
# ======================================================================

def print_mass_balance(mb: dict):
    print("\n=== Bilan de masse sur la période ===")
    print(f"P_tot          = {mb['P_tot_m']*1000:.1f} mm")
    print(f"Ruissellement  = {mb['R_tot_m']*1000:.1f} mm")
    print(f"Seepage profond= {mb['Seep_tot_m']*1000:.1f} mm")
    print(f"ETP effective (s_a sur h_a) = {mb['ET_tot_m']*1000:.1f} mm")
    print(f"ΔStock (Ia+sol)            = {mb['Delta_storage_m']*1000:.2f} mm")
    print(
        f"Erreur de fermeture         = {mb['Closure_error_mm']:.3f} mm "
        f"({mb['Relative_error_%']:.3f} %)"
    )


def main():
    dt = 300.0  # pas de temps = 5 min
    csv_rain = "PQ_BV_Cloutasse.csv"
    csv_etp = "ETP_SAFRAN_J.csv"

    # Aire du bassin pour construire Q_mod
    A_BV_M2 = 800000 

    # -- Lecture des données
    time_index, rain_5min_mm, p_rate_input, q_obs = read_rain_series_from_csv(csv_rain, dt)
    etp_rate = read_etp_series_for_time_index(csv_etp, time_index)

    # -- Paramètres 
    i_a = 0.04  # pertes initiales (m)
    s = 0.5     # capacité sol (m)
    k_infiltr = 5e-7
    k_seepage = 5e-7

    # -- Simulation simple du modèle
    res = run_scs_hsm_guinot(
        dt=dt,
        p_rate=p_rate_input,
        etp_rate=etp_rate,
        i_a=i_a,
        s=s,
        k_infiltr=k_infiltr,
        k_seepage=k_seepage,
        h_a_init=0.0,
        h_s_init=0.0,
        h_r_init=0.0,
    )

    # États (nt+1), alignés sur time_index (nt) en coupant le dernier point
    h_a = res["h_a"][:-1]
    h_s = res["h_s"][:-1]
    h_r = res["h_r"][:-1]

    # ==========================
    #  A. Préparation des séries
    # ==========================

    p_rate    = res["p"]         # [m/s]
    q_rate    = res["q"]         # [m/s]
    infil     = res["infil"]     # [m/s]
    r_rate    = res["r_rate"]    # [m/s]
    sa_loss   = res["sa_loss"]   # [m] par pas
    seep_loss = res["seep_loss"] # [m] par pas

    # 1) Flux instantanés en mm / 5 min
    factor_mm_5min = dt * 1000.0   # m/s * dt -> m, *1000 -> mm

    P_mm_5      = p_rate   * factor_mm_5min
    Qeff_mm_5   = q_rate   * factor_mm_5min
    Infil_mm_5  = infil    * factor_mm_5min
    Runoff_mm_5 = r_rate   * factor_mm_5min

    # 2) Cumuls en mm
    P_cum_mm      = np.cumsum(p_rate  * dt * 1000.0)
    R_cum_mm      = np.cumsum(r_rate  * dt * 1000.0)
    Infil_cum_mm  = np.cumsum(infil   * dt * 1000.0)
    Seep_cum_mm   = np.cumsum(seep_loss * 1000.0)   # m par pas -> mm
    ET_cum_mm     = np.cumsum(sa_loss   * 1000.0)   # m par pas -> mm

    # ==========================================
    #  B. DÉCIMATION POUR L'AFFICHAGE UNIQUEMENT
    # ==========================================

    step_plot = 12  # 1 point sur 12 -> 1 h (12 * 5 min)
    idx = slice(0, len(time_index), step_plot)

    t_plot           = time_index[idx]
    P_mm_5_plot     = P_mm_5[idx]
    Qeff_mm_5_plot  = Qeff_mm_5[idx]
    Infil_mm_5_plot = Infil_mm_5[idx]
    Runoff_mm_5_plot= Runoff_mm_5[idx]

    P_cum_plot      = P_cum_mm[idx]
    R_cum_plot      = R_cum_mm[idx]
    Infil_cum_plot  = Infil_cum_mm[idx]
    Seep_cum_plot   = Seep_cum_mm[idx]
    ET_cum_plot     = ET_cum_mm[idx]

    h_a_plot = h_a[idx]
    h_s_plot = h_s[idx]
    h_r_plot = h_r[idx]

    if q_obs is not None:
        q_obs_m3s = np.asarray(q_obs, dtype=float) / 1000.0  # si Q_ls en L/s
        q_obs_plot = q_obs_m3s[idx]
    else:
        q_obs_plot = None

    q_mod_full = r_rate * A_BV_M2
    q_mod_plot = q_mod_full[idx]

    # ==============================
    #  C. FIGURES 
    # ==============================

    base_dir = Path(__file__).resolve().parent
    plots_dir = base_dir.parent / "03_Plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ==============================
    #  FIGURE 1 : flux instantanés
    # ==============================

    fig1, ax = plt.subplots(figsize=(12, 5))

    ax.plot(t_plot, Qeff_mm_5_plot,   label="Pluie nette q (mm / 5 min)")
    ax.plot(t_plot, Infil_mm_5_plot,  label="Infiltration (mm / 5 min)")
    ax.plot(t_plot, Runoff_mm_5_plot, label="Ruissellement r (mm / 5 min)")

    ax.bar(t_plot, P_mm_5_plot, width=0.02, label="P (mm / 5 min)",
           alpha=0.3, align="center")

    ax.set_xlabel("Date")
    ax.set_ylabel("Flux (mm / 5 min)")
    ax.grid(True)
    ax.legend(loc="upper right")
    fig1.suptitle("Flux instantanés (P, q, infiltration, ruissellement)")
    fig1.tight_layout()
    fig1.savefig(plots_dir / "flux_instantanes_P_q_infil_r.png", dpi=100)

    # ==============================
    #  FIGURE 2 : cumuls (mm)
    # ==============================

    fig2, ax2 = plt.subplots(figsize=(12, 5))

    ax2.plot(t_plot, P_cum_plot,      label="P cumulée", linewidth=1.5)
    ax2.plot(t_plot, R_cum_plot,      label="Ruissellement cumulé", linewidth=1.5)
    ax2.plot(t_plot, Infil_cum_plot,  label="Infiltration cumulée", linewidth=1.0)
    ax2.plot(t_plot, Seep_cum_plot,   label="Seepage cumulé", linestyle="--")
    ax2.plot(t_plot, ET_cum_plot,     label="ET sur Ia cumulée", linestyle=":")

    ax2.set_xlabel("Date")
    ax2.set_ylabel("Lame cumulée (mm)")
    ax2.grid(True)
    ax2.legend(loc="upper left")
    fig2.suptitle("Cumuls P / ruissellement / infiltration / ET / seepage")
    fig2.tight_layout()
    fig2.savefig(plots_dir / "cumuls_P_R_infil_ET_seep.png", dpi=100)

    # ===================================
    #  FIGURE 3 : États des réservoirs
    # ===================================

    fig3, ax3 = plt.subplots(figsize=(12, 5))

    ax3.plot(t_plot, h_a_plot, label="h_a (Ia)",  color="grey")
    ax3.plot(t_plot, h_s_plot, label="h_s (sol)", color="green")
    ax3.plot(t_plot, h_r_plot, label="h_r",       color="red")

    ax3.set_xlabel("Date")
    ax3.set_ylabel("Hauteurs (m)")
    ax3.grid(True)
    ax3.legend(loc="upper left")
    fig3.suptitle("États des réservoirs (Ia, sol, runoff)")
    fig3.tight_layout()
    fig3.savefig(plots_dir / "etats_reservoirs_Ia_sol_runoff.png", dpi=100)

    # ==========================================
    #  FIGURE 4 : Hydrogramme Q_mod vs Q_obs
    # ==========================================

    if q_obs_plot is not None:
        fig4, ax4 = plt.subplots(figsize=(12, 4))
        ax4.plot(t_plot, q_mod_plot, label="Q_mod (r_rate * A)", linewidth=1.0)
        ax4.plot(t_plot, q_obs_plot, label="Q_obs (Q_ls)",       linewidth=1.0, alpha=0.7)

        ax4.set_xlabel("Date")
        ax4.set_ylabel("Débit (m³/s)")
        ax4.grid(True)
        ax4.legend(loc="upper right")
        fig4.suptitle("Comparaison Q_mod vs Q_obs (m³/s)")
        fig4.tight_layout()
        fig4.savefig(plots_dir / "Q_mod_vs_Q_obs.png", dpi=100)
    else:
        print("Pas de Q_obs : pas de figure Q_mod vs Q_obs.")

    # --------------------------------------------------------------
    # Bilan de masse
    # --------------------------------------------------------------
    print_mass_balance(res["mass_balance"])



if __name__ == "__main__":
    main()
