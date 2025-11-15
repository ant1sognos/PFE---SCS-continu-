# -*- coding: utf-8 -*-
"""
compare.py
----------
Simulation du modèle SCS-like HSM/Guinot à partir d'une série de pluie
issue du fichier ../02_Data/PQ_BV_Cloutasse.csv.

On trace :
  - h_a (Ia), h_s (sol), h_r (ruissellement) [m] sur l'axe de gauche
  - la pluie non cumulée P_mm [mm/5 min] sur l'axe de droite
  - l'axe des x est en dates (colonne dateP)
"""

from __future__ import annotations
from pathlib import Path

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ======================================================================
# 1. Modèle SCS-like HSM/Guinot
# ======================================================================

def run_scs_hsm(dt: float, p_rate: np.ndarray,
                i_a: float = 2e-3,
                s: float = 0.02,
                k_infiltr: float = 1e-5,
                k_seepage: float = 1e-6,
                h_a_init: float = 0.0,
                h_s_init: float = 0.0) -> dict:
    """
    Applique le modèle SCS-like HSM/Guinot à une série de pluie externe p_rate.

    dt        : pas de temps (s)
    p_rate    : intensité de pluie incidente (m/s), longueur nt
    i_a, s    : paramètres SCS (m)
    k_infiltr : coeff. d'infiltration (m/s)
    k_seepage : coeff. de fuite du sol (s^-1)
    h_a_init  : niveau initial Ia (m)
    h_s_init  : niveau initial sol (m)
    """
    nt = len(p_rate)
    t_vector = [i * dt for i in range(nt + 1)]

    precip_cumulated_vector = [0.0 for _ in range(nt + 1)]
    precip_rate_vector = [0.0 for _ in range(nt + 1)]
    h_a_vector = [0.0 for _ in range(nt + 1)]
    h_a_vector[0] = h_a_init
    h_s_vector = [0.0 for _ in range(nt + 1)]
    h_s_vector[0] = h_s_init
    h_r_vector = [0.0 for _ in range(nt + 1)]
    q_vector = [0.0 for _ in range(nt + 1)]
    infiltration_rate_vector = [0.0 for _ in range(nt + 1)]

    for n in range(nt):
        # Pluie incidente (m/s)
        p = float(p_rate[n])
        precip_rate_vector[n] = p
        precip_cumulated_vector[n + 1] = precip_cumulated_vector[n] + p * dt

        # Abstraction initiale Ia
        h_a_0 = h_a_vector[n]
        h_a = h_a_0 + dt * p
        if h_a < i_a:
            q = 0.0
            h_a_vector[n + 1] = h_a
        else:
            q = (h_a - i_a) / dt
            h_a_vector[n + 1] = i_a
        q_vector[n] = q

        # Répartition sol / ruissellement (loi HSM)
        h_s_begin = h_s_vector[n]
        X_begin = 1.0 - h_s_begin / s
        X_end = 1.0 / (1.0 / X_begin + k_infiltr * dt / s)
        h_s_end = (1.0 - X_end) * s
        infiltration_rate = (h_s_end - h_s_begin) / dt

        # Limitation par l'eau disponible à la surface
        h_r_begin = h_r_vector[n]
        infiltration_rate = min(infiltration_rate, h_r_begin / dt + q)

        # Mise à jour des stocks avant seepage
        h_r_vector[n + 1] = h_r_vector[n] + (q - infiltration_rate) * dt
        h_s_temp = h_s_vector[n] + infiltration_rate * dt
        infiltration_rate_vector[n] = infiltration_rate

        # Fuite du réservoir sol
        h_s_vector[n + 1] = h_s_temp * math.exp(-k_seepage * dt)

    results = {
        "t": np.array(t_vector),
        "P_cum": np.array(precip_cumulated_vector),
        "p": np.array(precip_rate_vector),
        "h_a": np.array(h_a_vector),
        "h_s": np.array(h_s_vector),
        "h_r": np.array(h_r_vector),
        "q": np.array(q_vector),
        "infil": np.array(infiltration_rate_vector),
    }
    return results


# ======================================================================
# 2. Lecture de la pluie : dates + P_mm + intensité (m/s)
# ======================================================================

def read_rain_series_from_csv(csv_name: str, dt: float = 300.0):
    """
    Lit ../02_Data/csv_name (séparateur ';'), extrait :
      - dateP   : datetimes
      - P_mm    : pluie (mm / 5 min)
    et renvoie (time_index, P_mm, p_rate_m_per_s)
    """
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir.parent / "02_Data"
    csv_path = data_dir / csv_name

    df = pd.read_csv(csv_path, sep=";")

    time_index = pd.to_datetime(df["dateP"])
    rain_5min_mm = df["P_mm"].to_numpy(dtype=float)

    # conversion mm/5min -> m/s
    p_rate = rain_5min_mm * 1e-3 / dt

    return time_index, rain_5min_mm, p_rate


# ======================================================================
# 3. Script principal : appel du modèle et tracé avec deux axes Y
# ======================================================================

def main():
    dt = 300.0  # 5 min en secondes
    csv_name = "PQ_BV_Cloutasse.csv"

    # Lecture dates + pluie
    time_index, rain_5min_mm, p_rate = read_rain_series_from_csv(csv_name, dt=dt)

    # Paramètres du modèle
    i_a = 2e-3
    s = 0.02
    k_infiltr = 1e-5
    k_seepage = 1e-6
    h_a_init = 0.0
    h_s_init = 0.0

    # Simulation
    res = run_scs_hsm(
        dt=dt,
        p_rate=p_rate,
        i_a=i_a,
        s=s,
        k_infiltr=k_infiltr,
        k_seepage=k_seepage,
        h_a_init=h_a_init,
        h_s_init=h_s_init,
    )

    # ⚠️ Alignement des tailles :
    # time_index et rain_5min_mm ont longueur nt
    # h_a, h_s, h_r ont longueur nt+1 -> on enlève le dernier point
    h_a = res["h_a"][:-1]
    h_s = res["h_s"][:-1]
    h_r = res["h_r"][:-1]

    # Préparation dossier de sortie
    base_dir = Path(__file__).resolve().parent
    plots_dir = base_dir.parent / "03_Plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / "runoff_scs_hsm_PQ_BV_Cloutasse.png"

    # --- Figure avec deux axes Y ---
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Axe gauche : hauteurs (m)
    line_ha, = ax1.plot(time_index, h_a, color="grey", label="h in i_a")
    line_hs, = ax1.plot(time_index, h_s, color="green", label="h in soil")
    line_hr, = ax1.plot(time_index, h_r, color="red", label="h runoff")

    ax1.set_xlabel("Date")
    ax1.set_ylabel("h (m)")
    ax1.grid(True)

    # Axe droit : pluie non cumulée (mm / 5 min)
    ax2 = ax1.twinx()
    line_P, = ax2.plot(time_index, rain_5min_mm, color="blue", alpha=0.5,
                       label="P (mm / 5 min)")
    ax2.set_ylabel("P (mm / 5 min)")

    # Légende combinée
    lines = [line_ha, line_hs, line_hr, line_P]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left")

    fig.suptitle("SCS-like runoff-infiltration model\n(données PQ_BV_Cloutasse)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Figure enregistrée dans : {out_path.resolve()}")


if __name__ == "__main__":
    main()
