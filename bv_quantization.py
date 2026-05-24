import numpy as np
from scipy.linalg import eigh

def bv_ghost_number_scores(returns, epsilon=1e-10):
    """
    Compute per‑ETF ghost number anomaly using median eigenvalue threshold.
    Eigenvalues > median → ghost number +1, else 0.
    Score = sum over modes (ghost_number * (eigenvector component)^2)
    """
    returns_clean = returns.dropna()
    n = returns_clean.shape[1]
    if n < 2:
        return {t: 0.0 for t in returns_clean.columns}
    corr = returns_clean.corr().values
    eigvals, eigvecs = eigh(corr)
    # Use median as threshold (or any percentile, e.g., 75%)
    threshold = np.median(eigvals)
    ghost = np.zeros_like(eigvals)
    ghost[eigvals > threshold] = 1.0
    # ghost[eigvals <= threshold] stays 0.0
    scores = np.zeros(n)
    for i in range(n):
        scores += ghost[i] * (eigvecs[:, i] ** 2)
    tickers = returns_clean.columns
    return {ticker: scores[j] for j, ticker in enumerate(tickers)}
