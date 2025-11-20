# -*- coding: utf-8 -*-
"""
traitementQobs.py
-----------------
Remplissage hydrologique de Q_ls :
 - petites lacunes par interpolation
 - périodes sèches hors événements pluie → Q = 0
 - interpolation à l’intérieur des événements
Écrit un nouveau CSV avec une colonne Q_ls_filled.

Ajout : calcul de la moyenne et de la variance des débits (original et final)
"""

import pandas as pd
import numpy as np
from pathlib import Path

def main():
    # ------------------------------------------------------------------
    # 1. Localisation des fichiers
    # ------------------------------------------------------------------
    base_dir = Path(__file__).resolve().parent           # .../01_Code
    data_dir = base_dir.parent / "02_Data"               # .../02_Data

    path_in = data_dir / "PQ_BV_Cloutasse.csv"
    path_out = data_dir / "PQ_BV_Cloutasse_filled_Q0.csv"

    print(f"Lecture du fichier : {path_in}")

    # ------------------------------------------------------------------
    # 2. Lecture du CSV
    # ------------------------------------------------------------------
    df = pd.read_csv(
        path_in,
        sep=";",
        na_values=["NA", "NaN", "", "-9999", "-9999.0"]
    )

    # Parse dates
    df["dateP"] = pd.to_datetime(df["dateP"])
    df["dateQ"] = pd.to_datetime(df["dateQ"])

    # Tri par temps
    df = df.sort_values("dateQ").reset_index(drop=True)

    # Série de base
    Q = df["Q_ls"].astype(float)
    P = df["P_mm"].astype(float)

    # ------------------------------------------------------------------
    # 3. Petites lacunes (<= 3 pas) : interpolation linéaire locale
    # ------------------------------------------------------------------
    Q_step1 = Q.interpolate(method="linear", limit=3, limit_direction="both")

    # ------------------------------------------------------------------
    # 4. Identification des périodes évènementielles
    # ------------------------------------------------------------------
    is_rain = P > 0.1
    window = 7  # environ 35 min
    event_mask = is_rain.rolling(
        window=window, center=True, min_periods=1
    ).max().astype(bool)

    # ------------------------------------------------------------------
    # 5. Remplir les NaN hors évènements (périodes sèches) par 0
    # ------------------------------------------------------------------
    Q_step2 = Q_step1.copy()
    dry_mask = (~event_mask) & Q_step2.isna()
    Q_step2.loc[dry_mask] = 0.0

    # ------------------------------------------------------------------
    # 6. Interpolation dans les évènements
    # ------------------------------------------------------------------
    Q_final = Q_step2.copy()
    in_event = event_mask.fillna(False)
    segment_id = (in_event != in_event.shift(1)).cumsum()

    for seg_val in segment_id.unique():
        seg_mask = segment_id == seg_val

        if not in_event[seg_mask].any():
            continue

        Q_seg = Q_final[seg_mask]

        if Q_seg.notna().sum() >= 2:
            Q_final.loc[seg_mask] = Q_seg.interpolate(
                method="linear",
                limit_direction="both"
            )

    # ------------------------------------------------------------------
    # 7. Remplacer les NaN résiduels par 0
    # ------------------------------------------------------------------
    Q_final = Q_final.fillna(0.0)

    # ------------------------------------------------------------------
    # 8. Statistiques supplémentaires (MOYENNE & VARIANCE)
    # ------------------------------------------------------------------
    Q_original_mean = float(Q.mean())
    Q_original_var  = float(Q.var())

    Q_final_mean = float(Q_final.mean())
    Q_final_var  = float(Q_final.var())

    # ------------------------------------------------------------------
    # 9. Résumé du traitement
    # ------------------------------------------------------------------
    before_nan_pct = Q.isna().mean() * 100.0
    after_nan_pct = Q_final.isna().mean() * 100.0
    nonzero_before = (Q > 0).mean() * 100.0
    nonzero_after = (Q_final > 0).mean() * 100.0

    summary = {
        "NaN_before_%": before_nan_pct,
        "NaN_after_%": after_nan_pct,
        "Nonzero_before_%": nonzero_before,
        "Nonzero_after_%": nonzero_after,
        "min_Q_original": float(Q.min()),
        "max_Q_original": float(Q.max()),
        "min_Q_final": float(Q_final.min()),
        "max_Q_final": float(Q_final.max()),
        "mean_Q_original": Q_original_mean,
        "var_Q_original": Q_original_var,
        "mean_Q_filled": Q_final_mean,
        "var_Q_filled": Q_final_var,
    }

    # Impression propre
    print("\n=== DIAGNOSTIC Q_ls ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    # ------------------------------------------------------------------
    # 10. Sauvegarde du CSV enrichi
    # ------------------------------------------------------------------
    df["Q_ls_filled"] = Q_final

    df.to_csv(path_out, sep=";", index=False)
    print(f"\nNouveau fichier écrit : {path_out}")

if __name__ == "__main__":
    main()
