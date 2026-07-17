import numpy as np


def compute_nonlinear_force_coefficients(N, H, Q_ce, kt, fc):
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


def compute_analytical_jacobian(N, H, x, k, fc):
    """
    Jenkins (elastic dry friction) element is piecewise differentiable.
    This function computes the Jacobian of the nonlinear force in AFT
    with cosine/sine coefficients.
    Inputs:
        N: Number of time samples per period (int)
        H: Highest retained harmonic (int)
        x: Cosine/sine displacement coefficients:
            [q0, a1, b1, a2, b2, ..., aH, bH] (ndarray of shape (2*H+1,))
        k: Tangential spring stiffness (float)
        fc: Coulomb limit force (float)
    Returns:
        F: Cosine/sine force coefficients (ndarray of shape (2*H+1,))
            [f0, A1, B1, A2, B2, ..., AH, BH]
        dF: Jacobian of F with respect to x (ndarray of shape (2*H+1, 2*H+1))
    """

    x = np.asarray(x, dtype=float).reshape(-1)
    n_coeff = 2 * H + 1

    if x.size != n_coeff:
        raise ValueError(f"x must have length {n_coeff}, got {x.size}.")

    # Time samples over one period
    tau_1 = np.arange(N) * 2 * np.pi / N

    # March over two periods for hysteresis stabilization
    tau = np.concatenate([tau_1, tau_1 + 2 * np.pi])
    Nt = tau.size

    # Build q(tau) and dq/dx
    q = np.full(Nt, x[0], dtype=float)
    Dq = np.zeros((Nt, n_coeff), dtype=float)
    Dq[:, 0] = 1.0

    for h in range(1, H + 1):
        cos_h = np.cos(h * tau)
        sin_h = np.sin(h * tau)

        ia = 2 * h - 1
        ib = 2 * h

        q += x[ia] * cos_h + x[ib] * sin_h
        Dq[:, ia] = cos_h
        Dq[:, ib] = sin_h

    # States
    f = np.zeros(Nt, dtype=float)
    qsl = np.zeros(Nt, dtype=float)

    Df = np.zeros((Nt, n_coeff), dtype=float)
    Dqsl = np.zeros((Nt, n_coeff), dtype=float)

    # Time marching
    for i in range(1, Nt):
        # Predictor
        f_pred = k * (q[i] - qsl[i - 1])
        Df_pred = k * (Dq[i, :] - Dqsl[i - 1, :])

        if abs(f_pred) >= fc:
            # Slip
            f[i] = fc * np.sign(f_pred)

            # Piecewise-linear Jacobian: zero away from switching points
            Df[i, :] = 0.0

            # Slider update
            qsl[i] = q[i] - f[i] / k
            Dqsl[i, :] = Dq[i, :] - Df[i, :] / k  # here simply Dq[i, :]
        else:
            # Stick
            f[i] = f_pred
            Df[i, :] = Df_pred

            qsl[i] = qsl[i - 1]
            Dqsl[i, :] = Dqsl[i - 1, :]

    # Keep last period only
    f_last = f[-N:]
    Df_last = Df[-N:, :]

    # Fourier coefficients in cosine/sine form
    F = np.zeros(n_coeff, dtype=float)
    dF = np.zeros((n_coeff, n_coeff), dtype=float)

    # Mean value
    F[0] = np.sum(f_last) / N
    dF[0, :] = np.sum(Df_last, axis=0) / N

    # Harmonics
    for h in range(1, H + 1):
        cos_h = np.cos(h * tau_1)
        sin_h = np.sin(h * tau_1)

        ia = 2 * h - 1
        ib = 2 * h

        # Cosine and sine coefficients
        F[ia] = 2.0 / N * np.sum(f_last * cos_h)
        F[ib] = 2.0 / N * np.sum(f_last * sin_h)

        dF[ia, :] = 2.0 / N * np.sum(Df_last * cos_h[:, None], axis=0)
        dF[ib, :] = 2.0 / N * np.sum(Df_last * sin_h[:, None], axis=0)

    return F, dF
