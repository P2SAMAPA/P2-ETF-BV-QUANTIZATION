import numpy as np
from scipy.linalg import eigh

def bv_ghost_number_scores(returns, epsilon=1e-10):
    """
    Compute per‑ETF Batalin–Vilkovisky ghost number anomaly.
    Uses spectral decomposition of the correlation matrix.
    Ghost number per eigenmode: +1 (positive eigenvalue), 0 (zero), -1 (negative).
    Per‑ETF score = sum_i ghost_i * (v_i^2) where v_i is the ETF's component in eigenvector i.
    """
    returns_clean = returns.dropna()
    n = returns_clean.shape[1]
    if n < 2:
        return {t: 0.0 for t in returns_clean.columns}
    # Use correlation matrix (standardised covariance)
    corr = returns_clean.corr().values
    # Eigen decomposition (symmetric)
    eigvals, eigvecs = eigh(corr)
    # Assign ghost number to each eigenmode
    ghost_numbers = np.zeros_like(eigvals)
    ghost_numbers[eigvals > epsilon] = 1.0
    ghost_numbers[np.abs(eigvals) <= epsilon] = 0.0
    ghost_numbers[eigvals < -epsilon] = -1.0
    # For each ETF, sum over modes: ghost_number * (eigenvector component)^2
    scores = np.zeros(n)
    for i in range(n):
        # eigvecs[:, i] is i-th eigenvector (column)
        # component for each ETF = eigvecs[:, i]^2
        scores += ghost_numbers[i] * (eigvecs[:, i] ** 2)
    # Ghost number can be between -1 and 1; shift to 0-1 for readability? We'll keep raw.
    tickers = returns_clean.columns
    return {ticker: scores[j] for j, ticker in enumerate(tickers)}
