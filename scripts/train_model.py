import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from hbm_nn.artifact_selection import select_data_id
from hbm_nn.config import ModelConfig
from hbm_nn.model import build_mlp
from hbm_nn.plotting import plot_loss
from hbm_nn.training import load_and_scale_data, fit, save_artifacts

os.environ["CUDA_VISIBLE_DEVICES"]=""

# suppress duplicate OpenMP warning
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

###############################################################################
# IDs and flags
###############################################################################
SAVE = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not SAVE:
    RED = "\033[91m"
    RESET = "\033[0m"
    input(f'{RED}WARNING: SAVE is set to {SAVE}. To save the model, '
          f'set SAVE = True. Press Enter to continue...{RESET}')

data_id = select_data_id()
data_file = 'data_H3_'+data_id

###############################################################################
# Configuration — edit here to change hyperparameters
###############################################################################
config = ModelConfig(
    hidden_sizes=[128, 128, 128, 128, 128],
    activation='gelu',
    dropout=0.0,
    n_epochs=1000,
    batch_size=64,
    early_stopping_patience=50,
    learning_rate=0.002,
)

###############################################################################
# Data
###############################################################################
X_train, y_train, X_val, y_val, *_, scaler = load_and_scale_data(data_file)

###############################################################################
# Model
###############################################################################
model = build_mlp(
    input_size=X_train.shape[1],
    output_size=y_train.shape[1],
    hidden_sizes=config.hidden_sizes,
    activation=config.activation,
    dropout=config.dropout,
)
print('\nModel architecture:')
print(model)

###############################################################################
# Training
###############################################################################
history = fit(model, X_train, y_train, X_val, y_val, config, device=device)

print(f'\nFinal train loss : {history["train_losses"][-1]:.3e}')
print(f'Final val   loss : {history["val_losses"][-1]:.3e}')
print(f'Best  val   loss : {history["best_val_loss"]:.3e}')

###############################################################################
# Save
###############################################################################
save_date = (np.datetime64('now').astype('str')
             .replace(':', '-').replace('T', '_'))
model_id = 'jenkins_h3_'+save_date
if SAVE:
    save_artifacts(model, scaler, history, model_id)

###############################################################################
# Plot loss curves
###############################################################################
plot_loss(history['train_losses'], history['val_losses'],
          figure_name=f'loss_{model_id}', save_figure=SAVE)
plt.show()
