# for now, only base harmonics
import numpy as np
from scipy.stats import truncnorm
from sklearn.model_selection import train_test_split
from time import strftime
from hbm_nn.aft_nonlinearity import compute_nonlinear_force_coefficients
from hbm_nn.fourier_conversion import (convert_cossin_to_comexp,
                                       convert_comexp_to_cossin)


training_points = 10000

H = 3       # Number of harmonics
N = 2**6    # Number of time samples
Omega = 2 * np.pi   # frequency (full cycle at 1)
T = 2 * np.pi / Omega
t = np.linspace(0, 2 * T, 2000)  # two cycles
# set k_t, f_c = 1 for nondimensional data
f_c = 1     # Coulomb force normalized to 1
k_t = 1     # stiffness normalized to 1

q_samples = []
fnl_samples = []

for i in range(training_points):
    # uniform distribution for base harmonic coefficient
    a1 = np.random.uniform(0.001, 5)
    # normal distribution between -0.2 and 0.2 for third harmonic coefficients
    a3 = truncnorm.rvs((-0.5)/0.1, (0.5)/0.1, loc=0.0, scale=0.5)
    b3 = truncnorm.rvs((-0.5)/0.1, (0.5)/0.1, loc=0.0, scale=0.5)
    # q_h = a1 * np.cos(t) + a3 * np.cos(3*t) + b3 * np.sin(3*t)
    q_cs = np.zeros(2 * H + 1)  # fourier coefficients a0, a1, b1, ...
    q_cs[1] = a1
    q_cs[5] = a3
    q_cs[6] = b3
    q_ce = convert_cossin_to_comexp(q_cs)

    fnl_ce = compute_nonlinear_force_coefficients(N, H, q_ce, k_t, f_c)
    fnl_cs = convert_comexp_to_cossin(fnl_ce, H)

    q_samples.append([a1, a3, b3])
    fnl_samples.append([fnl_cs[1], fnl_cs[2], fnl_cs[5], fnl_cs[6]])

# split data into 60% train, 20% valdation and 20% test
X_tmp, X_test, y_tmp, y_test = train_test_split(
    q_samples, fnl_samples, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_tmp, y_tmp, test_size=0.25, random_state=42
)

# save the splits to separate files
current_time = strftime("%Y-%m-%d_%H-%M-%S")
filename = f'data_H{H}_{current_time}'
np.savez(f'data/train_{filename}.npz',
         q_coeffs=X_train, fnl_coeffs=y_train)
np.savez(f'data/val_{filename}.npz',
         q_coeffs=X_val, fnl_coeffs=y_val)
np.savez(f'data/test_{filename}.npz',
         q_coeffs=X_test, fnl_coeffs=y_test)

print('Generated data:')
print(f'{np.shape(q_samples)[0]} samples for ' +
      f'{np.shape(q_samples)[1]} input features')
print(f'{np.shape(fnl_samples)[0]} samples for corresponding ' +
      f'{np.shape(fnl_samples)[1]} output features')
print(f'Data was split into {len(X_train)} training, {len(X_val)} ' +
      f'validation, and {len(X_test)} test samples')
