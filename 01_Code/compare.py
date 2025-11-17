# -*- coding: utf-8 -*-
"""
compare.py
----------
Simulation du modèle SCS -HSM à partir d'une série de pluie
issue du fichier ../02_Data/PQ_BV_Cloutasse.csv, en intégrant un terme
de déplétion s_a(t) sur le réservoir d'abstraction h_a, lié à l'ETP
SAFRAN journalière.

- L'ETP agit UNIQUEMENT sur h_a (réservoir d'abstraction).
  s_a(t) = ETP_effective(t) = min(ETP_pot(t), h_a(t)).

- On calcule :
    * états h_a, h_s, h_r (cumul de runoff)
    * flux instantanés : q(t), infiltration(t), r(t)
    * bilan de masse global
    * Q_mod(t) = r(t) * A_BV_M2

- On suppose que Q_ls dans le CSV est en L/s -> converti en m³/s.

- Calage automatique : minimisation d’une fonction objectif
  J = w_rmse * RMSE(Q_mod, Q_obs) + w_vol * [(R_mod - R_obs)/R_obs]^2
  avec scipy.optimize.minimize.
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
import math
import numpy as np
import math
import numpy as np


def run_scs_hsm(
    dt: float,
    p_rate: np.ndarray,
    etp_rate: np.ndarray | None = None,
    i_a: float = 2e-3,
    s: float = 0.02,
    k_infiltr: float = 1e-5,   # k dans v = k (1 - hs/S)^2
    k_seepage: float = 1e-6,   # déplétion profonde du sol
    h_a_init: float = 0.0,
    h_s_init: float = 0.0,
    h_r_init: float = 0.0,
) -> dict:
    """
    Modèle SCS-like continu inspiré du code de Guinot (Chap.4) :

      - ETP agit uniquement sur h_a : sa(ha,t)
      - Réservoir d'accumulation Ia (i_a) : q = pluie nette
      - Sol h_s avec infiltration v = k (1 - h_s/S)^2 (Eq. 13d)
      - Capacité d'infiltration limitée par l'eau dispo à la surface:
          v <= q + h_r / dt
      - Réservoir de surface h_r = lame de ruissellement cumulée (m)
      - seepage profond sur h_s via k_seepage

    Etats :
      h_a : stock dans Ia
      h_s : stock dans le sol
      h_r : lame de ruissellement cumulée (comme 'h_r_vector' chez Guinot)

    Flux de sortie :
      r_rate[n] = (h_r[n+1] - h_r[n]) / dt  (m/s) => Q_mod = r_rate * A_BV_M2

    Le dict retourné contient aussi un bilan de masse dans res["mass_balance"].
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

    # -- Etats
    h_a = np.zeros(nt + 1, dtype=float)
    h_s = np.zeros(nt + 1, dtype=float)
    h_r = np.zeros(nt + 1, dtype=float)

    h_a[0] = float(h_a_init)
    h_s[0] = float(h_s_init)
    h_r[0] = float(h_r_init)

    # -- Flux
    q = np.zeros(nt, dtype=float)          # pluie nette (m/s)
    v_infil = np.zeros(nt, dtype=float)    # infiltration effective (m/s)
    r_rate = np.zeros(nt, dtype=float)     # ruissellement (m/s), calculé APRES la boucle

    sa_loss = np.zeros(nt, dtype=float)    # ETP [m] par pas de temps sur h_a
    seep_loss = np.zeros(nt, dtype=float)  # seepage profond [m] par pas de temps

    # === Boucle temporelle ===
    for n in range(nt):
        p = float(p_rate[n])       # [m/s]
        etp = float(etp_rate[n])   # [m/s]

        # 1) ETP sur h_a (sa)
        h_a_0 = h_a[n]
        etp_pot = etp * dt                  # [m]
        etp_eff = min(etp_pot, h_a_0)       # on ne peut évaporer plus que le stock
        h_a_after_etp = h_a_0 - etp_eff
        sa_loss[n] = etp_eff

        # 2) Réservoir Ia -> pluie nette q
        h_a_temp = h_a_after_etp + p * dt   # accumulation brute
        if h_a_temp < i_a:
            q_n = 0.0
            h_a_next = h_a_temp
        else:
            q_n = (h_a_temp - i_a) / dt     # [m/s]
            h_a_next = i_a

        h_a[n + 1] = h_a_next
        q[n] = q_n

        # 3) Infiltration potentielle v_cap = k (1 - hs/S)^2
        h_s_begin = h_s[n]

        if h_s_begin >= s:
            v_cap = 0.0
        else:
            X = 1.0 - h_s_begin / s
            if X < 0.0:
                X = 0.0
            v_cap = k_infiltr * (X * X)     # [m/s]

        # 4) Limitation par l'eau dispo à la surface : q + h_r/dt
        water_avail_rate = q_n + h_r[n] / dt   # [m/s]
        if water_avail_rate < 0.0:
            water_avail_rate = 0.0

        v_n = max(0.0, min(v_cap, water_avail_rate))   # infiltration effective
        v_infil[n] = v_n

        # 5) Mise à jour du sol AVANT seepage
        h_s_temp = h_s_begin + v_n * dt

        # 6) Seepage profond sur le sol : décroissance exponentielle
        if k_seepage > 0.0:
            h_s_after_seep = h_s_temp * math.exp(-k_seepage * dt)
            seep = h_s_temp - h_s_after_seep
        else:
            h_s_after_seep = h_s_temp
            seep = 0.0

        seep_loss[n] = seep
        h_s[n + 1] = h_s_after_seep

        # 7) Réservoir de surface h_r : stock de ruissellement cumulé
        #    dh_r/dt = q - v
        h_r[n + 1] = h_r[n] + (q_n - v_n) * dt
        if h_r[n + 1] < 0.0:
            h_r[n + 1] = 0.0

    # 8) Ruissellement instantané => dérivée de h_r
    for n in range(nt):
        dh = h_r[n + 1] - h_r[n]
        if dh < 0.0:
            dh = 0.0
        r_rate[n] = dh / dt    # [m/s]

    # 9) Bilan de masse global
    P_tot = float(np.sum(p_rate) * dt)              # pluie totale [m]
    ETP_tot = float(np.sum(sa_loss))                # ETP totale sur h_a [m]
    Seep_tot = float(np.sum(seep_loss))             # seepage total [m]
    Runoff_tot = float(h_r[-1])                     # lame de ruissellement cumulée [m]
    dStorage = (h_a[-1] - h_a[0]) + (h_s[-1] - h_s[0])

    closing = P_tot - (ETP_tot + Seep_tot + Runoff_tot + dStorage)

    mass_balance = {
        "P_tot_m": P_tot,
        "ETP_tot_m": ETP_tot,
        "Seep_tot_m": Seep_tot,
        "Runoff_tot_m": Runoff_tot,
        "dStorage_m": dStorage,
        "closing_m": closing,
    }

    return {
        "t": t,
        "h_a": h_a,
        "h_s": h_s,
        "h_r": h_r,          # lame de ruissellement cumulée (comme chez Guinot)
        "q": q,
        "v_infil": v_infil,
        "r_rate": r_rate,    # débit de ruissellement [m/s]
        "sa_loss": sa_loss,
        "seep_loss": seep_loss,
        "mass_balance": mass_balance,
    }

# ======================================================================
# 2. Lecture pluie + Q_ls (en L/s) 
# ======================================================================

def read_rain_series_from_csv(csv_name: str, dt: float = 300.0):
    """
    Lit ../02_Data/csv_name (séparateur ';'), extrait :
      - dateP   : datetimes
      - P_mm    : pluie (mm / 5 min)  (NA -> 0)
      - Q_ls    : débit mesuré (L/s) -> converti en m³/s

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
# 5. Calage des paramètres sur Q_ls : fonction objectif
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
    w_rmse: float = 1.0,
    w_vol: float = 1.0,
):
    """
    Calage de (i_a, S, k_infiltr, k_seepage) par minimisation d'une
    fonction objectif :

        J(theta) = w_rmse * RMSE(Q_mod, Q_obs)
                 + w_vol  * ((R_mod - R_obs)/R_obs)^2

    avec :
      - Q_mod(t) = r_rate(t) * A_bv_m2
      - R_mod = profondeur de ruissellement modélisée (m)
      - R_obs = profondeur de ruissellement observée (m)

    On utilise scipy.optimize.minimize (L-BFGS-B).
    """
    if not HAS_SCIPY:
        raise RuntimeError("SciPy n'est pas disponible : calage impossible.")

    p_rate = np.asarray(p_rate, dtype=float)
    etp_rate = np.asarray(etp_rate, dtype=float)
    q_obs = np.asarray(q_obs, dtype=float)

    mask = ~np.isnan(q_obs)
    if mask.sum() == 0:
        raise ValueError("q_obs ne contient aucune valeur valide (toutes NaN ?)")

    # Pré-calcul du volume observé (profondeur R_obs)
    V_obs = np.nansum(q_obs[mask]) * dt          # volume [m3]
    R_obs = V_obs / A_bv_m2                      # profondeur [m]

    def objective(theta: np.ndarray) -> float:
        i_a, S, k_inf, k_seep = theta

        # Garde-fous simples
        if i_a <= 0 or S <= 0 or k_inf <= 0 or k_seep < 0:
            return 1e6

        res = run_scs_hsm(
            dt=dt,
            p_rate=p_rate,
            etp_rate=etp_rate,
            i_a=i_a,
            s=S,
            k_infiltr=k_inf,
            k_seepage=k_seep,
            h_a_init=0.0,
            h_s_init=0.0,
        )

        r_rate = res["r_rate"]          # [m/s]
        q_mod = r_rate * A_bv_m2        # [m³/s]

        # --- 1) RMSE sur les débits ---
        diff = q_mod[mask] - q_obs[mask]
        rmse = float(np.sqrt(np.mean(diff**2)))

        # --- 2) Ecart relatif sur le volume de ruissellement ---
        V_mod = np.nansum(q_mod[mask]) * dt    # volume [m3]
        R_mod = V_mod / A_bv_m2                # [m]

        if R_obs > 0:
            vol_pen = ((R_mod - R_obs) / R_obs) ** 2
        else:
            vol_pen = 0.0

        J = w_rmse * rmse + w_vol * vol_pen
        return J

    x0 = np.array([i_a_init, s_init, k_infiltr_init, k_seepage_init], dtype=float)

    # Bornes "physiques" (à adapter si besoin)
    bounds = [
        (5e-4, 5e-3),   # i_a : 0.5 à 5 mm
        (5e-3, 0.15),   # S   : 5 à 150 mm
        (1e-7, 5e-6),   # k_infiltr (m/s)
        (0.0,  5e-5),   # k_seepage (s^-1)
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

    # Paramètres initiaux (point de départ du calage)
    i_a = 2e-3
    s = 0.02
    k_infiltr = 1e-5
    k_seepage = 1e-6

    # --------------------------------------------------------------
    # Calage automatique si Q_obs et SciPy disponibles
    # --------------------------------------------------------------
    if q_obs is not None and HAS_SCIPY:
        print("Lancement du calage des paramètres sur Q_ls (fonction objectif)...")
        calib = calibrate_scs_hsm(
            dt, p_rate, etp_rate, q_obs, A_BV_M2,
            i_a, s, k_infiltr, k_seepage,
            w_rmse=1.0,   # poids RMSE débit
            w_vol=1.0,    # poids contrainte volume
        )
        print("\n=== Résultats du calage ===")
        print(f"Succès      : {calib['success']}")
        print(f"Message     : {calib['message']}")
        print(f"J_opt       = {calib['rmse_opt']:.3f}")
        for k, v in calib["opt_params"].items():
            print(f"  {k} = {v:.6g}")

        i_a = calib["opt_params"]["i_a"]
        s = calib["opt_params"]["s"]
        k_infiltr = calib["opt_params"]["k_infiltr"]
        k_seepage = calib["opt_params"]["k_seepage"]
    else:
        print("Pas de calage (Q_ls manquant ou SciPy indisponible).")

    # --------------------------------------------------------------
    # Simulation finale avec paramètres (éventuellement calibrés)
    # --------------------------------------------------------------
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

    # --------------------------------------------------------------
    # Figure principale
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
        "SCS-like runoff-infiltration model (HSM)\n"
        "(données PQ_BV_Cloutasse + ETP SAFRAN, ETP sur h_a uniquement)"
    )
    fig.tight_layout()
    fig.savefig(out_path_main, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Figure principale enregistrée dans : {out_path_main.resolve()}")

    # --------------------------------------------------------------
    # Comparaison Q_mod vs Q_obs + indicateurs
    # --------------------------------------------------------------
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

            # Profondeurs de ruissellement observée vs modélisée
            V_obs = np.nansum(q_obs[mask]) * dt
            V_mod = np.nansum(q_mod[mask]) * dt
            R_obs = V_obs / A_BV_M2
            R_mod = V_mod / A_BV_M2
            print(f"Runoff obs (profondeur) = {R_obs*1000:.1f} mm")
            print(f"Runoff mod (profondeur) = {R_mod*1000:.1f} mm")
    else:
        print("Pas de colonne Q_ls : pas de comparaison débit.")

    print_mass_balance(res["mass_balance"])


if __name__ == "__main__":
    main()
