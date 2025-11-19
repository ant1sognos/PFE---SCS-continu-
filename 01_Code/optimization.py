# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 13:03:17 2025

@author: asognos
"""

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
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib as mpl
import matplotlib.pyplot as plt

# Accélérer le rendu des grandes séries
mpl.rcParams["path.simplify"] = True
mpl.rcParams["path.simplify_threshold"] = 1.0
mpl.rcParams["agg.path.chunksize"] = 10000

# import matplotlib
# matplotlib.use("Agg")  # backend non interactif, plus rapide
# import matplotlib.pyplot as plt


# ======================================================================
# 1. Modèle SCS-like HSM/Guinot avec ETP sur h_a
# ======================================================================
def run_scs_hsm_guinot(
    dt: float,
    p_rate: np.ndarray,
    etp_rate: np.ndarray | None = None,
    i_a: float = 2e-3,
    s: float = 0.02,
    k_infiltr: float = 1e-6,
    k_seepage: float = 1e-3,
    h_a_init: float = 0.0,
    h_s_init: float = 0.0,
    h_r_init: float = 0.0,
) -> dict:
    """
    Modèle SCS-like HSM/Guinot avec ETP sur h_a.

    - Réservoir d'abstraction h_a (Ia) soumis à l'ETP effective s_a(t)
    - Pluie nette q(t) issue du remplissage de Ia
    - Réservoir sol h_s avec infiltration potentielle loi HSM (X_begin/X_end)
    - Infiltration limitée par l'eau dispo à la surface : v <= q + h_r/dt
    - Réservoir de surface h_r = lame de ruissellement cumulée
    - Seepage profond sur h_s
    - Bilan de masse global

    Sortie : dict contenant t, p, h_a, h_s, h_r, q, infil, r_rate, sa_loss,
             seep_loss, mass_balance.
    """
    # --- Prétraitement des forçages ---
    p_rate = np.nan_to_num(np.asarray(p_rate, dtype=float), nan=0.0)
    nt = len(p_rate)

    if etp_rate is None:
        etp_rate = np.zeros(nt, dtype=float)
    else:
        etp_rate = np.nan_to_num(np.asarray(etp_rate, dtype=float), nan=0.0)
        if len(etp_rate) != nt:
            raise ValueError("etp_rate doit avoir la même longueur que p_rate")

    # --- Temps ---
    t = np.array([i * dt for i in range(nt + 1)], dtype=float)

    # --- États (nt+1) ---
    h_a = np.zeros(nt + 1, dtype=float)
    h_s = np.zeros(nt + 1, dtype=float)
    h_r = np.zeros(nt + 1, dtype=float)

    h_a[0] = float(h_a_init)
    h_s[0] = float(h_s_init)
    h_r[0] = float(h_r_init)

    # --- Flux (nt) ---
    p_store   = np.zeros(nt, dtype=float)
    q         = np.zeros(nt, dtype=float)
    infil     = np.zeros(nt, dtype=float)
    r_rate    = np.zeros(nt, dtype=float)
    sa_loss   = np.zeros(nt, dtype=float)
    seep_loss = np.zeros(nt, dtype=float)

    # ========================
    #  BOUCLE TEMPORELLE
    # ========================
    for n in range(nt):
        p   = p_rate[n]     # [m/s]
        etp = etp_rate[n]   # [m/s]
        p_store[n] = p

        # 1) ETP sur h_a (sa)
        h_a_0   = h_a[n]
        etp_pot = etp * dt          # [m]
        etp_eff = min(etp_pot, h_a_0)
        h_a_after_etp = h_a_0 - etp_eff
        sa_loss[n] = etp_eff

        # 2) Réservoir Ia -> pluie nette q_n
        h_a_temp = h_a_after_etp + p * dt
        if h_a_temp < i_a:
            q_n = 0.0
            h_a_next = h_a_temp
        else:
            q_n = (h_a_temp - i_a) / dt   # [m/s]
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
        infil_pot = (h_s_end - h_s_begin) / dt   # [m/s]

        # 4) Limitation par l'eau dispo en surface
        h_r_begin = h_r[n]
        water_avail_rate = q_n + h_r_begin / dt   # [m/s]
        if water_avail_rate < 0.0:
            water_avail_rate = 0.0

        infil_n = max(0.0, min(infil_pot, water_avail_rate))
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
        h_r_next = h_r_begin + (q_n - infil_n) * dt
        if h_r_next < 0.0:
            h_r_next = 0.0
        h_r[n + 1] = h_r_next

    # 8) Ruissellement instantané r_rate = max(dh/dt, 0)
    for n in range(nt):
        dh = h_r[n + 1] - h_r[n]
        if dh < 0.0:
            dh = 0.0
        r_rate[n] = dh / dt

    # ========================
    #  BILAN DE MASSE
    # ========================
    P_tot   = float(np.nansum(p_rate) * dt)   # [m]
    R_tot   = float(h_r[-1])                  # [m]
    Seep_tot= float(np.nansum(seep_loss))     # [m]
    ET_tot  = float(np.nansum(sa_loss))       # [m]

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
    
    
    

# ======================================================================
# 5. Fonction objectif : RMSE entre Q_mod et Q_obs
# ======================================================================

def compute_rmse(q_obs: np.ndarray, q_mod: np.ndarray) -> float:
    """
    Calcule la RMSE entre Q_obs et Q_mod (m³/s).
    Renvoie une grosse valeur si série vide ou NaN.
    """
    q_obs = np.asarray(q_obs, dtype=float)
    q_mod = np.asarray(q_mod, dtype=float)

    mask = np.isfinite(q_obs) & np.isfinite(q_mod)
    if mask.sum() == 0:
        return 1e6

    diff = q_mod[mask] - q_obs[mask]
    rmse = np.sqrt(np.mean(diff**2))
    return float(rmse)


def objective(theta: np.ndarray, data: dict) -> float:
    """
    Fonction objectif à MINIMISER (multistart + optimisation locale).

    theta = [i_a, s, log10_k_infiltr, log10_k_seepage]

    On travaille en log10 pour les coefficients pour les garder > 0
    et mieux conditionner l'optimisation.
    """
    i_a, s, log10_k_infiltr, log10_k_seepage = theta

    # Quelques gardes-fous simples (éviter des valeurs absurdes)
    if i_a < 0.0 or i_a > 0.3 or s <= 0.0 or s > 1.5:
        return 1e6
    k_infiltr = 10.0 ** log10_k_infiltr
    k_seepage = 10.0 ** log10_k_seepage

    # Simulation du modèle avec ces paramètres
    res = run_scs_hsm_guinot(
        dt=data["dt"],
        p_rate=data["p_rate"],
        etp_rate=data["etp_rate"],
        i_a=i_a,
        s=s,
        k_infiltr=k_infiltr,
        k_seepage=k_seepage,
        h_a_init=0.0,
        h_s_init=0.0,
        h_r_init=0.0,
    )

    r_rate = res["r_rate"]              # [m/s]
    q_mod = r_rate * data["A_BV_M2"]    # -> m³/s

    rmse = compute_rmse(data["q_obs_m3s"], q_mod)
    if not np.isfinite(rmse):
        return 1e6

    return rmse


def sample_random_theta(bounds: list[tuple[float, float]]) -> np.ndarray:
    """
    Tire un vecteur de paramètres aléatoires dans les bornes.
    bounds = [(min_i_a, max_i_a), (min_s, max_s),
              (min_log10_k_infiltr, max_log10_k_infiltr),
              (min_log10_k_seepage, max_log10_k_seepage)]
    """
    return np.array(
        [np.random.uniform(lo, hi) for (lo, hi) in bounds],
        dtype=float
    )


def calibrate_multistart(
    data: dict,
    bounds: list[tuple[float, float]],
    n_starts: int = 15,
) -> tuple[np.ndarray, float]:
    """
    Stratégie multistart + optimisation locale (Powell) :
      - n_starts points initiaux aléatoires
      - pour chacun, optimisation locale
      - on garde le meilleur RMSE

    Retourne (theta_opt, J_opt).
    """
    best_theta = None
    best_J = np.inf

    for k in range(n_starts):
        theta0 = sample_random_theta(bounds)

        res = minimize(
    objective,
    theta0,
    args=(data,),
    method="Powell",
    bounds=bounds,          # ➜ très important
    options={"maxiter": 150, "disp": False},
)


        print(f"Essai {k+1}/{n_starts} : J = {res.fun:.4f}")

        if res.fun < best_J:
            best_J = float(res.fun)
            best_theta = np.array(res.x, dtype=float)

    return best_theta, best_J
    
def main():
    dt = 300.0  # pas de temps = 5 min
    csv_rain = "PQ_BV_Cloutasse.csv"
    csv_etp = "ETP_SAFRAN_J.csv"

    # Aire du bassin pour construire Q_mod
    A_BV_M2 = 800000 

    # --------------------------------------------------------------
    # 1. Lecture des données
    # --------------------------------------------------------------
    time_index, rain_5min_mm, p_rate_input, q_obs = read_rain_series_from_csv(csv_rain, dt)
    etp_rate = read_etp_series_for_time_index(csv_etp, time_index)

    # Q_obs en m³/s si dispo
    if q_obs is not None:
        q_obs_m3s = np.asarray(q_obs, dtype=float) / 1000.0
    else:
        q_obs_m3s = None

    # --------------------------------------------------------------
    # 2. CALAGE (optionnel)
    # --------------------------------------------------------------
    DO_CALIBRATION = True
    
    if DO_CALIBRATION and q_obs_m3s is not None:
        bounds = [
            (0.0, 0.1),      # i_a : 0 à 10 cm 
            (0.05, 0.8),     # s   : 5 cm à 80 cm
            (-10.0, -4.0),   # log10(k_infiltr) ~ 1e-10 à 1e-4
            (-10.0, -4.0),   # log10(k_seepage) ~ 1e-10 à 1e-4
        ]

        data = {
            "dt": dt,
            "p_rate": p_rate_input,
            "etp_rate": etp_rate,
            "q_obs_m3s": q_obs_m3s,
            "A_BV_M2": A_BV_M2,
        }

        print("Lancement du calage (multistart + Powell) sur RMSE(Q_mod, Q_obs)...")
        theta_opt, J_opt = calibrate_multistart(data, bounds, n_starts=3)

        i_a_opt, s_opt, log10_k_infiltr_opt, log10_k_seepage_opt = theta_opt
        k_infiltr_opt = 10.0 ** log10_k_infiltr_opt
        k_seepage_opt = 10.0 ** log10_k_seepage_opt

        print("\n=== Résultats du calage ===")
        print(f"RMSE_opt = {J_opt:.4f} m³/s")
        print(f"  i_a        = {i_a_opt:.6f} m")
        print(f"  s          = {s_opt:.6f} m")
        print(f"  k_infiltr  = {k_infiltr_opt:.3e} m/s")
        print(f"  k_seepage  = {k_seepage_opt:.3e} s^-1")

        i_a = i_a_opt
        s = s_opt
        k_infiltr = k_infiltr_opt
        k_seepage = k_seepage_opt

    else:
        i_a = 0.04  
        s = 0.5    
        k_infiltr = 5e-7
        k_seepage = 5e-7

    # --------------------------------------------------------------
    # 3. Simulation
    # --------------------------------------------------------------
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

    # --------------------------------------------------------------
    # 4. Séries complètes (AUCUNE décimation)
    # --------------------------------------------------------------
    p_rate    = res["p"]         # [m/s]
    q_rate    = res["q"]         # [m/s]
    infil     = res["infil"]     # [m/s]
    r_rate    = res["r_rate"]    # [m/s]
    sa_loss   = res["sa_loss"]   # [m] par pas
    seep_loss = res["seep_loss"] # [m] par pas

    # Flux instantanés en mm / 5 min
    factor_mm_5min = dt * 1000.0   # m/s * dt -> m, *1000 -> mm

    P_mm_5      = p_rate   * factor_mm_5min
    Qeff_mm_5   = q_rate   * factor_mm_5min
    Infil_mm_5  = infil    * factor_mm_5min
    Runoff_mm_5 = r_rate   * factor_mm_5min

    # Cumuls en mm
    P_cum_mm      = np.cumsum(p_rate   * dt * 1000.0)
    R_cum_mm      = np.cumsum(r_rate   * dt * 1000.0)
    Infil_cum_mm  = np.cumsum(infil    * dt * 1000.0)
    Seep_cum_mm   = np.cumsum(seep_loss * 1000.0)
    ET_cum_mm     = np.cumsum(sa_loss   * 1000.0)

    # Pour les traces : on garde tout
    t_plot           = time_index
    P_mm_5_plot      = P_mm_5
    Qeff_mm_5_plot   = Qeff_mm_5
    Infil_mm_5_plot  = Infil_mm_5
    Runoff_mm_5_plot = Runoff_mm_5

    P_cum_plot       = P_cum_mm
    R_cum_plot       = R_cum_mm
    Infil_cum_plot   = Infil_cum_mm
    Seep_cum_plot    = Seep_cum_mm
    ET_cum_plot      = ET_cum_mm

    h_a_plot = h_a
    h_s_plot = h_s
    h_r_plot = h_r

    if q_obs is not None:
        q_obs_m3s = np.asarray(q_obs, dtype=float) / 1000.0  # si Q_ls en L/s
        q_obs_plot = q_obs_m3s
    else:
        q_obs_plot = None

    q_mod_full = r_rate * A_BV_M2
    q_mod_plot = q_mod_full

    # --------------------------------------------------------------
    # 5. FIGURES (optimisées mais FULL RESOLUTION)
    # --------------------------------------------------------------
    base_dir = Path(__file__).resolve().parent
    plots_dir = base_dir.parent / "03_Plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # FIGURE 1 : flux instantanés
    fig1, ax = plt.subplots(figsize=(10, 4))

    ax.plot(t_plot, Qeff_mm_5_plot,   label="Pluie nette q (mm / 5 min)",
            linewidth=0.8, antialiased=False)
    ax.plot(t_plot, Infil_mm_5_plot,  label="Infiltration (mm / 5 min)",
            linewidth=0.8, antialiased=False)
    ax.plot(t_plot, Runoff_mm_5_plot, label="Ruissellement r (mm / 5 min)",
            linewidth=0.8, antialiased=False)

    # Pluie en aplat plutôt qu'en barres (plus rapide que ax.bar)
    ax.fill_between(t_plot, 0, P_mm_5_plot,
                    step="post", alpha=0.3, label="P (mm / 5 min)")

    ax.set_xlabel("Date")
    ax.set_ylabel("Flux (mm / 5 min)")
    ax.grid(True, linewidth=0.3)
    ax.legend(loc="upper right")
    fig1.suptitle("Flux instantanés (P, q, infiltration, ruissellement)")
    fig1.savefig(plots_dir / "flux_instantanes_P_q_infil_r.png", dpi=80)
    plt.close(fig1)

    # FIGURE 2 : cumuls (mm)
    fig2, ax2 = plt.subplots(figsize=(10, 4))

    ax2.plot(t_plot, P_cum_plot,      label="P cumulée", linewidth=1.0, antialiased=False)
    ax2.plot(t_plot, R_cum_plot,      label="Ruissellement cumulé", linewidth=1.0, antialiased=False)
    ax2.plot(t_plot, Infil_cum_plot,  label="Infiltration cumulée", linewidth=0.8, antialiased=False)
    ax2.plot(t_plot, Seep_cum_plot,   label="Seepage cumulé", linestyle="--", linewidth=0.8)
    ax2.plot(t_plot, ET_cum_plot,     label="ET sur Ia cumulée", linestyle=":", linewidth=0.8)

    ax2.set_xlabel("Date")
    ax2.set_ylabel("Lame cumulée (mm)")
    ax2.grid(True, linewidth=0.3)
    ax2.legend(loc="upper left")
    fig2.suptitle("Cumuls P / ruissellement / infiltration / ET / seepage")
    fig2.savefig(plots_dir / "cumuls_P_R_infil_ET_seep.png", dpi=80)
    plt.close(fig2)

    # FIGURE 3 : États des réservoirs
    fig3, ax3 = plt.subplots(figsize=(10, 4))

    ax3.plot(t_plot, h_a_plot, label="h_a (Ia)",  color="grey",  linewidth=0.8)
    ax3.plot(t_plot, h_s_plot, label="h_s (sol)", color="green", linewidth=0.8)
    ax3.plot(t_plot, h_r_plot, label="h_r",       color="red",   linewidth=0.8)

    ax3.set_xlabel("Date")
    ax3.set_ylabel("Hauteurs (m)")
    ax3.grid(True, linewidth=0.3)
    ax3.legend(loc="upper left")
    fig3.suptitle("États des réservoirs (Ia, sol, runoff)")
    fig3.savefig(plots_dir / "etats_reservoirs_Ia_sol_runoff.png", dpi=80)
    plt.close(fig3)

    # FIGURE 4 : Hydrogramme Q_mod vs Q_obs
    if q_obs_plot is not None:
        fig4, ax4 = plt.subplots(figsize=(10, 4))
        ax4.plot(t_plot, q_mod_plot, label="Q_mod (r_rate * A)",
                 linewidth=0.8, antialiased=False)
        ax4.plot(t_plot, q_obs_plot, label="Q_obs (Q_ls)",
                 linewidth=0.8, alpha=0.7, antialiased=False)

        ax4.set_xlabel("Date")
        ax4.set_ylabel("Débit (m³/s)")
        ax4.grid(True, linewidth=0.3)
        ax4.legend(loc="upper right")
        fig4.suptitle("Comparaison Q_mod vs Q_obs (m³/s)")
        fig4.savefig(plots_dir / "Q_mod_vs_Q_obs.png", dpi=80)
        plt.close(fig4)
    else:
        print("Pas de Q_obs : pas de figure Q_mod vs Q_obs.")

    # --------------------------------------------------------------
    # 6. Bilan de masse
    # --------------------------------------------------------------
    print_mass_balance(res["mass_balance"])





if __name__ == "__main__":
    main()
