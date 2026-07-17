import pandas as pd
import numpy as np


jac_info = pd.read_csv('data/EB1to3_cond_Om_force80_kt10000000_muN106.csv')

print('fnorm_error average:', np.average(jac_info['fnorm_error']))
