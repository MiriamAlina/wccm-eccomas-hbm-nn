import numpy as np


def compute_aft_solution(N, H, Q_ce, kt, fc):
    """
    Compute Fourier coefficients of nonlinear forcing from displacement
    coefficients.
    Inputs:
        N: Number of time points per period (int)
        H: Number of harmonics (int)
        Q_ce: Fourier coefficients of displacement (complex array)
        kt: Tangential spring stiffness (float)
        fc: Friction limit force (float)
    Returns:
        Fnl_ce: Fourier coefficients of nonlinear force (complex array)
    """

    N_orig = N
    N = 2 * N                              # Evaluate force over two periods
    n = np.arange(N).reshape(-1, 1)        # Column vector for time indices
    h = np.arange(-H, H+1).reshape(1, -1)  # Row vector for harmonics

    # Inverse DFT matrix with factor 2 in exponent for 2 periods
    E_NH = np.exp(1j * 2 * 2 * np.pi / N * (n @ h))
    q = np.real(E_NH @ Q_ce).flatten()     # Time domain reconstruction

    # Initialize force and slider position vectors
    fnl = np.zeros_like(q)
    qsl = np.zeros_like(q)

    # Iterative computation of friction force (Jenkins element)
    for ij in range(1, len(fnl)):

        # Predictor step
        fnl[ij] = kt * (q[ij] - qsl[ij-1])

        # Corrector step
        if abs(fnl[ij]) >= fc:
            fnl[ij] = fc * np.sign(fnl[ij])
            qsl[ij] = q[ij] - fnl[ij] / kt
        else:
            qsl[ij] = qsl[ij-1]

    N = int(N_orig)  # Reset N to original (one period)

    # Forward DFT using only second period samples
    Fnl_ce = (E_NH[:N, :].conj().T @ fnl[-N:]) / N

    return Fnl_ce
