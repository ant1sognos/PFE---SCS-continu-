# -*- coding: utf-8 -*-
"""
observeMod_Th.py
-----------------
Script AUTONOME pour :
  - charger une série de débits et de pluie depuis un CSV,
  - détecter les évènements de ruissellement,
  - filtrer les pseudo-évènements (sans pluie / faible amplitude),
  - calculer des métriques par évènement (Qmax, volume, durées),
  - stocker les résultats dans des CSV,
  - tracer un graphique par évènement avec Q et P superposés.

Hypothèses sur le fichier ../02_Data/PQ_BV_Cloutasse.csv :
  - séparateur ;
  - colonnes : dateP, P_mm, + une colonne de débit (observé ou modélisé).

Pour changer la source du débit analysé, modifier la constante Q_SOURCE :
    Q_SOURCE = "obs"  -> débit observé
    Q_SOURCE = "mod"  -> débit modélisé
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# ===================== PARAMÈTRE CLÉ ===================== #
# "obs" = débit observé ; "mod" = débit modélisé
Q_SOURCE = "obs"   # changer en "mod" pour découper les évènements simulés
# ========================================================= #


def detect_runoff_events(
    Q: pd.Series,
    P: pd.Series | None = None,
    threshold: float | None = None,
    threshold_quantile: float = 60.0,
    qmin_event: float | None = None,
    q_amp_min: float | None = None,
    pmin_event_mm: float | None = None,
    min_gap_steps: int = 24,
    pre_pad_steps: int = 12,
    post_pad_steps: int = 24,
    dt_seconds: float | None = None,
    out_dir: str | Path | None = None,
    prefix: str = "events",
):
    """
    Détection et découpage des évènements de ruissellement dans une série de débits.

    Q : pd.Series
        Débit (L/s ou m3/s) avec index temporel.
    P : pd.Series ou None
        Pluie (mm / pas de temps) avec le même index que Q (optionnel mais recommandé).
    threshold :
        - Si None, calculé comme le 'threshold_quantile' de Q (>0).
        - Sinon, valeur absolue en unités de Q.
    qmin_event :
        Seuil minimal sur Qmax pour garder un événement (sinon rejeté).
        Si None -> même valeur que 'threshold'.
    q_amp_min :
        Amplitude minimale (Qmax - Qmin_full) pour garder un événement.
        Si None -> pas de filtrage sur l'amplitude.
    pmin_event_mm :
        Pluie cumulée minimale (mm) sur la fenêtre de l'évènement pour le garder.
        Si None -> pas de filtrage sur la pluie.
    """

    if not isinstance(Q, pd.Series):
        raise TypeError("Q doit être une pd.Series (débit avec index temporel).")

    n = len(Q)
    if n < 2:
        raise ValueError("La série Q est trop courte pour détecter des évènements.")

    if P is not None and not isinstance(P, pd.Series):
        raise TypeError("P doit être une pd.Series si fournie.")
    if P is not None and not Q.index.equals(P.index):
        raise ValueError("Q et P doivent avoir exactement le même index.")

    Q_values = Q.values.astype(float)

    # 1) Détermination du pas de temps si possible
    if dt_seconds is None:
        if isinstance(Q.index, pd.DatetimeIndex):
            dt_seconds = (Q.index[1] - Q.index[0]).total_seconds()
        else:
            raise ValueError(
                "dt_seconds doit être fourni si l'index de Q n'est pas un DatetimeIndex."
            )

    # 2) Calcul / choix du seuil d'activité
    if threshold is None:
        q_pos = Q_values[np.isfinite(Q_values) & (Q_values > 0)]
        if len(q_pos) == 0:
            raise ValueError(
                "Impossible d'estimer un seuil : Q ne contient que des NaN ou des zéros."
            )
        threshold = float(np.nanpercentile(q_pos, threshold_quantile))
        print(
            f"[INFO] Seuil automatique d'évènement (quantile {threshold_quantile}%) : "
            f"{threshold:.3f} (unités de Q)"
        )
    else:
        print(f"[INFO] Seuil d'évènement fixé manuellement : {threshold:.3f} (unités de Q)")

    if qmin_event is None:
        qmin_event = threshold
        print(f"[INFO] Seuil de Qmax pour garder un évènement : qmin_event = {qmin_event:.3f}")
    else:
        print(f"[INFO] Seuil de Qmax fixé : qmin_event = {qmin_event:.3f}")

    if q_amp_min is not None:
        print(f"[INFO] Amplitude minimale de crue : q_amp_min = {q_amp_min:.3f} (unités de Q)")
    if pmin_event_mm is not None:
        print(f"[INFO] Pluie minimale par évènement : pmin_event_mm = {pmin_event_mm:.2f} mm")

    # 3) Détection brute des instants actifs (Q > threshold)
    is_valid = np.isfinite(Q_values)
    is_active = (Q_values > threshold) & is_valid
    is_active_clean = is_active.copy()

    # 4) On comble les petits "trous" (segments inactifs trop courts)
    inact = ~is_active_clean
    changes = np.diff(inact.astype(int))

    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0]

    if inact[0]:
        starts = np.r_[0, starts]
    if inact[-1]:
        ends = np.r_[ends, n - 1]

    for s, e in zip(starts, ends):
        length = e - s + 1
        if length < min_gap_steps:
            is_active_clean[s : e + 1] = True

    # 5) Après comblement, on redétecte les segments actifs
    act = is_active_clean
    changes = np.diff(act.astype(int))

    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0]

    if act[0]:
        starts = np.r_[0, starts]
    if act[-1]:
        ends = np.r_[ends, n - 1]

    events = []
    ts_rows = []

    for s_core, e_core in zip(starts, ends):
        # Indices avec padding avant/après
        s_full = max(0, s_core - pre_pad_steps)
        e_full = min(n - 1, e_core + post_pad_steps)

        Q_evt = Q.iloc[s_full : e_full + 1]
        Q_vals_evt = Q_evt.values.astype(float)

        if not np.any(np.isfinite(Q_vals_evt)):
            continue  # rien de valide

        Qmax = float(np.nanmax(Q_vals_evt))
        Qmin = float(np.nanmin(Q_vals_evt))
        Q_amp = Qmax - Qmin

        # -- filtres --
        if Qmax < qmin_event:
            continue
        if (q_amp_min is not None) and (Q_amp < q_amp_min):
            continue

        # Pluie sur l'évènement si dispo
        if P is not None:
            P_evt = P.iloc[s_full : e_full + 1].values.astype(float)
            P_tot = float(np.nansum(np.nan_to_num(P_evt, nan=0.0)))
            if (pmin_event_mm is not None) and (P_tot < pmin_event_mm):
                continue
        else:
            P_tot = np.nan

        t_Qmax = Q_evt.idxmax()
        volume = float(np.nansum(np.nan_to_num(Q_vals_evt, nan=0.0)) * dt_seconds)
        duration_full_s = (e_full - s_full + 1) * dt_seconds
        duration_core_s = (e_core - s_core + 1) * dt_seconds

        event_id = len(events) + 1

        events.append(
            {
                "event_id": event_id,
                "t_start_full": Q.index[s_full],
                "t_end_full": Q.index[e_full],
                "t_start_core": Q.index[s_core],
                "t_end_core": Q.index[e_core],
                "duration_full_s": duration_full_s,
                "duration_core_s": duration_core_s,
                "Qmax": Qmax,
                "Qmin_full": Qmin,
                "Q_amp": Q_amp,
                "t_Qmax": t_Qmax,
                "volume_Q_dt": volume,
                "P_tot_mm": P_tot,
            }
        )

        if P is not None:
            P_evt_series = P.iloc[s_full : e_full + 1]
        else:
            P_evt_series = pd.Series(data=np.nan, index=Q_evt.index, name="P_mm")

        tmp = pd.DataFrame(
            {
                "event_id": event_id,
                "time": Q_evt.index,
                "Q": Q_evt.values,
                "P_mm": P_evt_series.values,
            }
        )
        ts_rows.append(tmp)

    if events:
        events_df = pd.DataFrame(events).set_index("event_id")
        ts_df = pd.concat(ts_rows, ignore_index=True)
    else:
        events_df = pd.DataFrame(
            columns=[
                "t_start_full",
                "t_end_full",
                "t_start_core",
                "t_end_core",
                "duration_full_s",
                "duration_core_s",
                "Qmax",
                "Qmin_full",
                "Q_amp",
                "t_Qmax",
                "volume_Q_dt",
                "P_tot_mm",
            ]
        )
        ts_df = pd.DataFrame(columns=["event_id", "time", "Q", "P_mm"])

    # 6) Sauvegarde éventuelle sur disque
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        events_path = out_dir / f"{prefix}_summary.csv"
        ts_path = out_dir / f"{prefix}_timeseries.csv"

        events_df.to_csv(events_path, index=True)
        ts_df.to_csv(ts_path, index=False)

        print(f"[OK] Résumé évènements écrit dans : {events_path}")
        print(f"[OK] Séries découpées écrites dans : {ts_path}")

    return events_df, ts_df


def main():
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir.parent / "02_Data" / "PQ_BV_Cloutasse.csv"

    # ---- On adapte les colonnes et les sorties selon la source Q_SOURCE ---- #
    if Q_SOURCE == "obs":
        candidate_cols = ["Q_ls", "Q_m3s", "Q_ls_m3s", "Q_obs", "Q"]
        out_dir = base_dir.parent / "03_Plots" / "events_obs"
        prefix = "obs"
        label_Q = "Q (obs)"
    elif Q_SOURCE == "mod":
        # à adapter aux noms de colonnes de ton CSV de simulation
        candidate_cols = ["Q_mod", "Q_sim", "Q_model", "Q_cal", "Q"]
        out_dir = base_dir.parent / "03_Plots" / "events_mod"
        prefix = "mod"
        label_Q = "Q (mod)"
    else:
        raise ValueError("Q_SOURCE doit être 'obs' ou 'mod'.")

    default_dt_seconds = 300.0  # 5 minutes

    # --- paramètres de détection / filtrage ---
    threshold = None           # None -> seuil automatique (quantile)
    threshold_quantile = 60.0  # ex : 60e percentile de Q>0
    qmin_event = None          # None -> même valeur que threshold
    q_amp_min = 1.0            # amplitude min en L/s (ou unités de Q)
    pmin_event_mm = 1.0        # pluie min cumulée sur l'évènement (mm)

    min_gap_steps = 24         # 24*5 min = 2 h de trou pour clore un évènement
    pre_pad_steps = 12         # 1 h avant
    post_pad_steps = 36        # 3 h après

    # ===== 2. Lecture des données =====
    if not data_path.exists():
        raise FileNotFoundError(
            f"Fichier de données introuvable : {data_path.resolve()}\n"
            f"→ Adapte 'data_path' dans main()."
        )

    print(f"Lecture du fichier : {data_path}")
    df = pd.read_csv(
        data_path,
        sep=";",
        na_values=["NA", "NaN", "", -9999, -9999.0],
    )

    print("Colonnes trouvées dans le CSV :", list(df.columns))

    if "dateP" not in df.columns:
        raise ValueError("La colonne 'dateP' est absente du CSV.")

    df["dateP"] = pd.to_datetime(df["dateP"])
    df = df.set_index("dateP")

    if "P_mm" not in df.columns:
        raise ValueError("La colonne 'P_mm' (pluie) est absente du CSV.")
    P_mm = df["P_mm"].astype(float)

    Q_col = None
    for col in candidate_cols:
        if col in df.columns:
            Q_col = col
            break
    if Q_col is None:
        raise ValueError(
            "Impossible de trouver une colonne de débit dans le CSV pour la source "
            f"{Q_SOURCE!r}.\n"
            f"Colonnes candidates testées : {candidate_cols}\n"
            f"Colonnes présentes : {list(df.columns)}"
        )
    print(f"[INFO] Colonne de débit utilisée ({Q_SOURCE}) : {Q_col}")
    Q = df[Q_col].astype(float)

    if isinstance(Q.index, pd.DatetimeIndex) and len(Q) > 1:
        dt_seconds = (Q.index[1] - Q.index[0]).total_seconds()
        print(f"Pas de temps inféré : {dt_seconds:.1f} s")
    else:
        dt_seconds = default_dt_seconds
        print(f"Pas de temps par défaut utilisé : {dt_seconds:.1f} s")

    # ===== 4. Détection des évènements =====
    print(f"Détection des évènements de ruissellement sur Q_{Q_SOURCE}...")
    events_df, ts_df = detect_runoff_events(
        Q=Q,
        P=P_mm,
        threshold=threshold,
        threshold_quantile=threshold_quantile,
        qmin_event=qmin_event,
        q_amp_min=q_amp_min,
        pmin_event_mm=pmin_event_mm,
        min_gap_steps=min_gap_steps,
        pre_pad_steps=pre_pad_steps,
        post_pad_steps=post_pad_steps,
        dt_seconds=dt_seconds,
        out_dir=out_dir,
        prefix=prefix,
    )

    print(f"\nNombre d'évènements détectés après filtrage : {len(events_df)}")
    if len(events_df):
        print(events_df[["t_start_full", "t_end_full", "Qmax", "Q_amp", "P_tot_mm"]].head())
    else:
        print("Aucun évènement détecté avec les paramètres actuels.")
        return

    # ===== 5. Tracés par évènement =====
    out_dir.mkdir(parents=True, exist_ok=True)

    for event_id, row in events_df.iterrows():
        t0 = row["t_start_full"]
        t1 = row["t_end_full"]

        Q_evt = Q.loc[t0:t1]
        P_evt = P_mm.loc[t0:t1]

        # Interpolation pour l'affichage (pour lisser les trous)
        Q_plot = Q_evt.copy()
        P_plot = P_evt.copy()
        if isinstance(Q_plot.index, pd.DatetimeIndex):
            Q_plot = Q_plot.interpolate(method="time", limit_direction="both")
            P_plot = P_plot.interpolate(method="time", limit_direction="both")
        else:
            Q_plot = Q_plot.interpolate(limit_direction="both")
            P_plot = P_plot.interpolate(limit_direction="both")

        fig, ax1 = plt.subplots(figsize=(10, 4))

        ax1.plot(Q_plot.index, Q_plot.values, label=label_Q, linewidth=1.2)
        ax1.set_ylabel("Débit (L/s ou m³/s)")
        ax1.set_xlabel("Temps")
        ax1.grid(True, which="both", linestyle="--", alpha=0.4)

        ax2 = ax1.twinx()
        dt_days = dt_seconds / 86400.0
        ax2.bar(
            P_plot.index,
            P_plot.values,
            width=dt_days * 0.8,
            alpha=0.3,
            label="P (mm/pas)",
        )
        ax2.set_ylabel("Pluie (mm/pas)")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        fig.suptitle(
            f"Évènement {event_id} ({Q_SOURCE}) : {t0:%Y-%m-%d %H:%M} → {t1:%Y-%m-%d %H:%M}",
            fontsize=10,
        )
        fig.tight_layout()

        fig_path = out_dir / f"event_{prefix}_{event_id:03d}.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)

        print(f"[OK] Figure évènement {event_id} enregistrée : {fig_path}")


if __name__ == "__main__":
    main()
