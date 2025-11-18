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
    Version "Guinot" adaptée à une série temporelle :

      - Réservoir d'accumulation Ia (h_a), avec ETP sa(t) uniquement sur h_a.
      - Pluie nette q(t) = (h_a_after_etp + p*dt - Ia)/dt si ha dépasse Ia.
      - Réservoir sol h_s alimenté par infiltration potentielle de type HSM :

            X = 1 - h_s / S
            dX/dt = - (k_infiltr / S) * X^2   (Guinot)
        intégrée par :

            X_end = 1 / (1 / X_begin + k_infiltr * dt / S)

        ⇒ infiltration_pot = (h_s_end - h_s_begin) / dt.

      - Infiltration limitée par l'eau dispo à la surface :
            infil <= q + h_r/dt
        (comme dans le code de Guinot: min(infil_pot, h_r/dt + q)).
      - Réservoir de surface h_r : lame de ruissellement cumulée (m) :
            dh_r/dt = q - infil
      - Seepage profond sur h_s via un terme exponentiel (k_seepage).

    Sorties :
      - t           : temps (s)
      - p           : pluie instantanée (m/s)
      - h_a, h_s, h_r : états (m)
      - q           : pluie nette (m/s)
      - infil       : infiltration effective (m/s)
      - r_rate      : ruissellement instantané (m/s) = dh_r/dt ≥ 0
      - sa_loss     : ETP effective sur h_a (m) par pas de temps
      - seep_loss   : seepage profond (m) par pas de temps
      - mass_balance: bilan de masse global (dict)
    """
    # -- Forçages
    p_rate = np.nan_to_num(np.asarray(p_rate, dtype=float), nan=0.0)
    nt = len(p_rate)

    if etp_rate is None:
        etp_rate = np.zeros(nt, dtype=float)
    else:
        etp_rate = np.nan_to_num(np.asarray(etp_rate, dtype=float), nan=0.0)
        if len(etp_rate) != nt:
            raise ValueError("etp_rate doit avoir la même longueur que p_rate")

    # -- Temps
    t = np.array([i * dt for i in range(nt + 1)], dtype=float)

    # -- Etats (nt+1)
    h_a = np.zeros(nt + 1, dtype=float)
    h_s = np.zeros(nt + 1, dtype=float)
    h_r = np.zeros(nt + 1, dtype=float)

    h_a[0] = float(h_a_init)
    h_s[0] = float(h_s_init)
    h_r[0] = float(h_r_init)

    # -- Flux (nt)
    p_store = np.zeros(nt, dtype=float)
    q = np.zeros(nt, dtype=float)
    infil = np.zeros(nt, dtype=float)
    r_rate = np.zeros(nt, dtype=float)
    sa_loss = np.zeros(nt, dtype=float)
    seep_loss = np.zeros(nt, dtype=float)

    # === Boucle temporelle ===
    for n in range(nt):
        p = float(p_rate[n])       # [m/s]
        etp = float(etp_rate[n])   # [m/s]
        p_store[n] = p

        # 1) ETP sur h_a (sa)
        h_a_0 = h_a[n]
        etp_pot = etp * dt              # [m]
        etp_eff = min(etp_pot, h_a_0)   # on ne peut évaporer plus que le stock
        h_a_after_etp = h_a_0 - etp_eff
        sa_loss[n] = etp_eff

        # 2) Réservoir Ia -> pluie nette q
        h_a_temp = h_a_after_etp + p * dt
        if h_a_temp < i_a:
            q_n = 0.0
            h_a_next = h_a_temp
        else:
            q_n = (h_a_temp - i_a) / dt   # [m/s]
            h_a_next = i_a

        q[n] = q_n
        h_a[n + 1] = h_a_next

        # 3) Infiltration potentielle HSM comme dans le code Guinot
        h_s_begin = h_s[n]
        X_begin = 1.0 - h_s_begin / s
        if X_begin <= 0.0:
            X_begin = 1e-12  # éviter division par zéro
        X_end = 1.0 / (1.0 / X_begin + k_infiltr * dt / s)
        h_s_end = (1.0 - X_end) * s
        infil_pot = (h_s_end - h_s_begin) / dt   # [m/s]

        # 4) Limitation par l'eau dispo à la surface : q + h_r/dt
        h_r_begin = h_r[n]
        water_avail_rate = q_n + h_r_begin / dt  # [m/s]
        if water_avail_rate < 0.0:
            water_avail_rate = 0.0

        infil_n = max(0.0, min(infil_pot, water_avail_rate))  # [m/s]
        infil[n] = infil_n

        # 5) Mise à jour du sol AVANT seepage
        h_s_temp = h_s_begin + infil_n * dt

        # 6) Seepage profond
        if k_seepage > 0.0:
            h_s_after_seep = h_s_temp * math.exp(-k_seepage * dt)
            seep = h_s_temp - h_s_after_seep
        else:
            h_s_after_seep = h_s_temp
            seep = 0.0

        h_s[n + 1] = h_s_after_seep
        seep_loss[n] = seep

        # 7) Réservoir de surface h_r (lame de ruissellement cumulée)
        #    dh_r/dt = q - infil
        h_r[n + 1] = h_r_begin + (q_n - infil_n) * dt
        if h_r[n + 1] < 0.0:
            h_r[n + 1] = 0.0

    # 8) Ruissellement instantané r_rate = dh_r/dt (en m/s)
    for n in range(nt):
        dh = h_r[n + 1] - h_r[n]
        if dh < 0.0:
            dh = 0.0
        r_rate[n] = dh / dt

    # 9) Bilan de masse global
    P_tot = float(np.nansum(p_rate) * dt)      # [m]
    R_tot = float(h_r[-1])
    Seep_tot = float(np.nansum(seep_loss))     # [m]
    ET_tot = float(np.nansum(sa_loss))         # [m]

    d_h_a = h_a[-1] - h_a[0]
    d_h_s = h_s[-1] - h_s[0]
    delta_storage = d_h_a + d_h_s             # [m]

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
    print(f"Erreur de fermeture         = {mb['Closure_error_mm']:.3f} mm "
          f"({mb['Relative_error_%']:.3f} %)")


# ======================================================================
# 5. Script principal (AUCUN CALAGE)
# ======================================================================

def main():
    dt = 300.0  # pas de temps = 5 min
    csv_rain = "PQ_BV_Cloutasse.csv"
    csv_etp = "ETP_SAFRAN_J.csv"

    # Aire du bassin pour construire Q_mod (0.8 km² par ex.)
    A_BV_M2 = 800000 

    # -- Lecture des données
    time_index, rain_5min_mm, p_rate, q_obs = read_rain_series_from_csv(csv_rain, dt)
    etp_rate = read_etp_series_for_time_index(csv_etp, time_index)

    # -- Paramètres 
    i_a = 0.01
    s = 0.15
    k_infiltr = 5e-7
    k_seepage = 1e-7

    # -- Simulation simple du modèle .
    res = run_scs_hsm_guinot(
        dt=dt,
        p_rate=p_rate,
        etp_rate=etp_rate,
        i_a=i_a,
        s=s,
        k_infiltr=k_infiltr,
        k_seepage=k_seepage,
        h_a_init=0.0,
        h_s_init=0.0,
        h_r_init=0.0,
    )

    # Etats (nt+1), alignés sur time_index (nt) en coupant le dernier point
    h_a = res["h_a"][:-1]
    h_s = res["h_s"][:-1]
    h_r = res["h_r"][:-1]
    r_rate = res["r_rate"]           # [m/s]

    # Lame de ruissellement instantanée en mm / 5 min
    runoff_5min_mm = r_rate * dt * 1000.0

    base_dir = Path(__file__).resolve().parent
    plots_dir = base_dir.parent / "03_Plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path_main = plots_dir / "runoff_scs_hsm_PQ_BV_Cloutasse_debug.png"
    out_path_Q = plots_dir / "runoff_vs_Qls_PQ_BV_Cloutasse_debug.png"

    # --------------------------------------------------------------
    # Figure principale : états + flux surfaciques
    # --------------------------------------------------------------
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
        "SCS-like runoff-infiltration model (Guinot/HSM)\n"
        "(PQ_BV_Cloutasse)"
    )
    fig.tight_layout()
    fig.savefig(out_path_main, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Figure principale enregistrée dans : {out_path_main.resolve()}")

    # --------------------------------------------------------------
    # Comparaison Q_mod vs Q_obs 
    # --------------------------------------------------------------
    if q_obs is not None:
        q_obs = np.asarray(q_obs, dtype=float)
        q_mod = r_rate * A_BV_M2  # [m³/s]

        fig2, ax_q = plt.subplots(figsize=(12, 4))
        ax_q.plot(time_index, q_mod, label="Q modèle", linewidth=1.0)
        ax_q.plot(time_index, q_obs, label="Q_ls observé", linewidth=1.0, alpha=0.7)
        ax_q.set_xlabel("Date")
        ax_q.set_ylabel("Débit (m³/s)")
        ax_q.grid(True)
        ax_q.legend()
        fig2.suptitle("Comparaison Q_mod vs Q_ls observé ")
        fig2.tight_layout()
        fig2.savefig(out_path_Q, dpi=300, bbox_inches="tight")
        plt.show()
        print(f"Figure Q_mod vs Q_ls enregistrée dans : {out_path_Q.resolve()}")

    else:
        print("Pas de colonne Q_ls : pas de comparaison débit.")

    # --------------------------------------------------------------
    # Bilan de masse
    # --------------------------------------------------------------
    print_mass_balance(res["mass_balance"])


if __name__ == "__main__":
    main()
