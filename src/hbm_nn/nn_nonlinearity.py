import numpy as np
import torch
import joblib
from functools import lru_cache
from pathlib import Path


def _artifact_path(filename: str) -> Path:
    """Return an artifact path that works from repo root and scripts/."""
    candidate_dirs = [
        Path('models'),
        Path('../models'),
        Path(__file__).resolve().parents[2] / 'models',
    ]
    for directory in candidate_dirs:
        path = directory / filename
        if path.is_file():
            return path
    return candidate_dirs[0] / filename


@lru_cache(maxsize=None)
def _load_artifacts(NN_id: str):
    """Load and cache model/scaler tensors for one trained model id."""
    NN_id = str(NN_id)
    model = torch.load(_artifact_path(f'mlp_jenkins_h3_{NN_id}.pt'),
                       weights_only=False)
    model.eval()

    scaler = joblib.load(
        _artifact_path(f'scaling_params_jenkins_h3_{NN_id}.joblib')
    )
    scaler_tensors = {
        'X_mean': torch.tensor(scaler['X_mean'], dtype=torch.float32),
        'X_std': torch.tensor(scaler['X_std'], dtype=torch.float32),
        'y_mean': torch.tensor(scaler['y_mean'], dtype=torch.float32),
        'y_std': torch.tensor(scaler['y_std'], dtype=torch.float32),
    }
    return model, scaler_tensors


def infer_nonlinear_force_coefficients(NN_id, X):
    """
    Evaluate the trained neural network model for the given input X.
    Inputs:
        NN_id: Identifier for the trained neural network model (str)
        X : Input data for the neural network (numpy array)

    Returns:
        Rescaled neural network's prediction for the given input X
            (numpy array)
    """
    NN_model, scaler = _load_artifacts(str(NN_id))

    nn_input = torch.tensor(np.array(X, dtype=float), dtype=torch.float32)
    nn_input_scaled = (
        (nn_input - scaler['X_mean']) / scaler['X_std']
    )

    with torch.no_grad():
        output_scaled = NN_model(nn_input_scaled)

    output = output_scaled * scaler['y_std'] + scaler['y_mean']

    return output.detach().numpy()


def compute_autodiff_jacobian(NN_id, input):
    """
    Compute the Jacobian of the neural network model with respect to the input
    using automatic differentiation.
    Inputs:
        NN_id: Identifier for the trained neural network model (str)
        input : Input data for the neural network (numpy array)
    Returns:
        Jacobian of the neural network model with respect to the input
            (numpy array)
    """
    NN_model, scaler = _load_artifacts(str(NN_id))

    X = np.array(input, dtype=float)
    x = torch.tensor(X, dtype=torch.float32, requires_grad=True)
    x_scaled = (x - scaler['X_mean']) / scaler['X_std']

    jac_scaled = torch.autograd.functional.jacobian(NN_model, x_scaled)
    jac_scaled = jac_scaled.squeeze()

    jac = (
        jac_scaled
        * scaler['y_std'].unsqueeze(1)
        / scaler['X_std'].unsqueeze(0)
    )

    return jac.detach().numpy()


def infer_nonlinear_force_and_jacobian(NN_id, X):
    """Evaluate the trained model and its input Jacobian in one Python call."""
    NN_model, scaler = _load_artifacts(str(NN_id))

    X = np.array(X, dtype=float)
    x = torch.tensor(X, dtype=torch.float32, requires_grad=True)
    x_scaled = (x - scaler['X_mean']) / scaler['X_std']

    with torch.no_grad():
        output_scaled = NN_model(x_scaled.detach())
    output = output_scaled * scaler['y_std'] + scaler['y_mean']

    jac_scaled = torch.autograd.functional.jacobian(NN_model, x_scaled)
    jac_scaled = jac_scaled.squeeze()
    jac = (
        jac_scaled
        * scaler['y_std'].unsqueeze(1)
        / scaler['X_std'].unsqueeze(0)
    )

    return output.detach().numpy(), jac.detach().numpy()
