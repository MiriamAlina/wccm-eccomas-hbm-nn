import numpy as np
import torch
import joblib
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
    model_path = _artifact_path(f'mlp_jenkins_h3_{NN_id}.pt')
    NN_model = torch.load(model_path, weights_only=False)
    NN_model.eval()

    nn_input = torch.tensor(X, dtype=torch.float32)

    scaler = joblib.load(
        _artifact_path(f'scaling_params_jenkins_h3_{NN_id}.joblib')
    )
    X_mean = torch.tensor(scaler['X_mean'], dtype=torch.float32)
    X_std = torch.tensor(scaler['X_std'], dtype=torch.float32)
    y_mean = torch.tensor(scaler['y_mean'], dtype=torch.float32)
    y_std = torch.tensor(scaler['y_std'], dtype=torch.float32)
    nn_input_scaled = (nn_input - X_mean) / X_std

    with torch.no_grad():
        output_scaled = NN_model(nn_input_scaled)

    output = output_scaled * y_std + y_mean

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
    NN_model = torch.load(_artifact_path(f'mlp_jenkins_h3_{NN_id}.pt'),
                          weights_only=False)
    NN_model.eval()

    scaler = joblib.load(
        _artifact_path(f'scaling_params_jenkins_h3_{NN_id}.joblib')
    )
    X_mean = torch.tensor(scaler['X_mean'], dtype=torch.float32)
    X_std = torch.tensor(scaler['X_std'], dtype=torch.float32)
    y_std = torch.tensor(scaler['y_std'], dtype=torch.float32)

    X = np.array(input, dtype=float)
    x = torch.tensor(X, dtype=torch.float32, requires_grad=True)
    x_scaled = (x - X_mean) / X_std

    jac_scaled = torch.autograd.functional.jacobian(NN_model, x_scaled)
    jac_scaled = jac_scaled.squeeze()

    jac = jac_scaled * y_std.unsqueeze(1) / X_std.unsqueeze(0)

    return jac.detach().numpy()
