# -*- coding: utf-8 -*-
"""
gaussiennes_Qmax.py — version avec sauvegarde automatique

Les figures sont sauvegardées dans :
    ../03_Plots/Gaussienne/
par rapport au script.

Le dossier est créé automatiquement s'il n'existe pas.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def fit_gaussian(sample):
    sample = np.asarray(sample)
    mu = np.mean(sample)
    sigma = np.std(sample, ddof=1)
    return mu, sigma


def gaussian_pdf(x, mu, sigma):
    return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
        - (x - mu) ** 2 / (2 * sigma ** 2)
    )


def plot_gaussians_Qmax(Qmax_obs, Qmax_mod, name="gaussiennes_Qmax"):
    Qmax_obs = np.asarray(Qmax_obs)
    Qmax_mod = np.asarray(Qmax_mod)

    # --- 1. Ajustement des paramètres ---
    mu_obs, sigma_obs = fit_gaussian(Qmax_obs)
    mu_mod, sigma_mod = fit_gaussian(Qmax_mod)

    print(f"Observé : mu = {mu_obs:.2f}, sigma = {sigma_obs:.2f}")
    print(f"Modélisé : mu = {mu_mod:.2f}, sigma = {sigma_mod:.2f}")

    # --- 2. Domaine ---
    q_min = min(Qmax_obs.min(), Qmax_mod.min())
    q_max = max(Qmax_obs.max(), Qmax_mod.max())
    marge = 0.1 * (q_max - q_min) if q_max > q_min else 1.0

    x = np.linspace(q_min - marge, q_max + marge, 500)

    # --- 3. PDF ---
    pdf_obs = gaussian_pdf(x, mu_obs, sigma_obs)
    pdf_mod = gaussian_pdf(x, mu_mod, sigma_mod)

    # --- 4. Figure ---
    plt.figure(figsize=(8, 5))

    plt.plot(x, pdf_obs, label=fr"Qmax obs ($\mu={mu_obs:.2f}$, $\sigma={sigma_obs:.2f}$)")
    plt.plot(x, pdf_mod, "--", label=fr"Qmax mod ($\mu={mu_mod:.2f}$, $\sigma={sigma_mod:.2f}$)")

    plt.hist(Qmax_obs, bins=10, density=True, alpha=0.25, label="Histogramme obs")
    plt.hist(Qmax_mod, bins=10, density=True, alpha=0.25, label="Histogramme mod")

    plt.xlabel(r"Débit maximal $Q_{\max}$ (m$^3$/s)")
    plt.ylabel("Densité de probabilité")
    plt.title("Comparaison des distributions de $Q_{\\max}$")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # ----------------------------------------------------------------------
    # 5. Sauvegarde DANS 03_Plots/Gaussienne/
    # ----------------------------------------------------------------------
    output_dir = Path(__file__).resolve().parent.parent / "03_Plots" / "Gaussienne"
    output_dir.mkdir(parents=True, exist_ok=True)

    save_path = output_dir / f"{name}.png"
    plt.savefig(save_path, dpi=300)
    print(f"\n✔ Figure sauvegardée dans : {save_path}\n")

    plt.show()


if __name__ == "__main__":
    # Exemple fictif -> à remplacer par tes Qmax réels
    rng = np.random.default_rng(42)
    Qmax_obs = rng.normal(5.0, 1.0, 60)
    Qmax_mod = rng.normal(4.5, 1.2, 60)

    plot_gaussians_Qmax(Qmax_obs, Qmax_mod)
