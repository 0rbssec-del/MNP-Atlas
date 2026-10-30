# experiments/akarlar_tableau4.py
# Test sur les données publiées d'Akarlar (2026, arXiv 2604.15400, tableau 4 : balayage des 28 couches au pas 1,
# 24 essais par cellule) : homogénéité entre couches (khi²) et probabilité que le maximum sur 28 couches atteigne
# la valeur mise en avant si toutes les couches ont le même taux (simulation binomiale).
# Usage : python experiments/akarlar_tableau4.py   (CPU, quelques secondes ; numpy et scipy)

import numpy as np
from scipy.stats import binom, chi2_contingency

N = 24
# Taux publiés (%) par couche, L0 à L27, tableau 4 de l'article
H2C = [25.0, 8.3, 16.7, 12.5, 16.7, 16.7, 25.0, 20.8, 16.7, 20.8, 16.7, 25.0, 12.5, 4.2, 8.3, 16.7, 4.2, 16.7, 8.3,
       12.5, 20.8, 12.5, 20.8, 20.8, 33.3, 8.3, 12.5, 12.5]
C2H = [66.7, 83.3, 66.7, 66.7, 70.8, 62.5, 58.3, 66.7, 75.0, 66.7, 66.7, 70.8, 54.2, 62.5, 62.5, 79.2, 79.2, 58.3,
       58.3, 75.0, 87.5, 62.5, 79.2, 62.5, 58.3, 58.3, 66.7, 79.2]


def main():
    rng = np.random.default_rng(0)
    for nom, pct in (("H→C (correction)", H2C), ("C→H (corruption)", C2H)):
        pct = np.array(pct)
        k = np.round(pct * N / 100).astype(int)
        assert np.allclose(k / N * 100, pct, atol=0.06), nom            # les pourcentages sont bien des k/24
        p = k.sum() / (N * len(k))
        _, pval, _, _ = chi2_contingency(np.stack([k, N - k]))
        maxi = rng.binomial(N, p, (200_000, len(k))).max(1)
        print(f"{nom} : taux global {k.sum()}/{N * len(k)} = {p:.3f} ; khi² d'homogénéité p = {pval:.2f} ; "
              f"maximum publié {k.max()}/{N} (couche {k.argmax()}) ; P(maximum ≥ publié | aucun effet de couche) = "
              f"{(maxi >= k.max()).mean():.2f} ; maximum médian attendu {int(np.median(maxi))}/{N} ; "
              f"P(une couche fixée d'avance ≥ {k.max()}/{N}) = {1 - binom.cdf(k.max() - 1, N, p):.3f}")


if __name__ == "__main__":
    main()
