import torch
import joblib


def evaluate_model(NN_id, X):
    """
    Evaluate the trained neural network model for the given input X.
    Inputs:
        NN_id: Identifier for the trained neural network model (str)
        X : Input data for the neural network (numpy array)

    Returns:
        Rescaled neural network's prediction for the given input X
            (numpy array)
    """
    model_path = f'models/mlp_jenkins_h3_{NN_id}.pt'
    NN_model = torch.load(model_path, weights_only=False)
    NN_model.eval()

    nn_input = torch.tensor(X, dtype=torch.float32)

    scaler = joblib.load(f'models/scaling_params_jenkins_h3_{NN_id}.joblib')
    X_mean = torch.tensor(scaler['X_mean'], dtype=torch.float32)
    X_std = torch.tensor(scaler['X_std'], dtype=torch.float32)
    y_mean = torch.tensor(scaler['y_mean'], dtype=torch.float32)
    y_std = torch.tensor(scaler['y_std'], dtype=torch.float32)
    nn_input_scaled = (nn_input - X_mean) / X_std

    with torch.no_grad():
        output_scaled = NN_model(nn_input_scaled)

    output = output_scaled * y_std + y_mean

    return output.detach().numpy()
