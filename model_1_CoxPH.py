import pandas as pd
import numpy as np
import torch
from utils import create_unified_time_grid
from sklearn.model_selection import train_test_split
from sksurv.metrics import integrated_brier_score

df_all = pd.read_csv('data/1_gbsg_all.csv')
feature_cols = ['meno', 'grade', 'hormon', 'age', 'size', 'nodes', 'pgr', 'er']

df_train, df_test = train_test_split(df_all, test_size=0.3)

# organize the data, let the event and time be the last two columns
df_train = df_train[feature_cols + ['event', 'time']]
df_test = df_test[feature_cols + ['event', 'time']]

# cox model
from lifelines import CoxPHFitter
cph = CoxPHFitter()
cph.fit(df_train[feature_cols + ['time', 'event']], duration_col='time', event_col='event')

# compute the concordance index on the test data
concordance_index = cph.score(df_test[feature_cols + ['time', 'event']], scoring_method="concordance_index")
print(f"Traditional Cox Model - Concordance Index: {concordance_index:.4f}")

# compute the ibs score based on scikit-survival
# Convert DataFrames to structured arrays for scikit-survival
y_train = np.array([(bool(e), float(t)) for e, t in zip(df_train['event'], df_train['time'])],
                dtype=[('event', bool), ('time', float)])
y_test = np.array([(bool(e), float(t)) for e, t in zip(df_test['event'], df_test['time'])],
                dtype=[('event', bool), ('time', float)])
# Create unified time grid for IBS computation
time_grid = create_unified_time_grid(df_test['time'].values, n_points=200)
# Get survival function predictions (returns DataFrame with times as index and patients as columns)
# lifelines returns DataFrame, NOT scikit-survival StepFunctions
cph_surv_funcs = cph.predict_survival_function(df_test[feature_cols])

# Convert lifelines DataFrame to scikit-survival StepFunctions
from sksurv.functions import StepFunction
cph_step_funcs = []
for col in cph_surv_funcs.columns:
    # Each column is one patient's survival function
    times = cph_surv_funcs.index.values
    probs = cph_surv_funcs[col].values
    cph_step_funcs.append(StepFunction(times, probs))

# Adjust time_grid to be within the survival function's domain
# Get the domain from the first function (all should have similar domains)
func_min, func_max = cph_step_funcs[0].domain
# Clip time_grid to be within the function's domain
time_grid_clipped = time_grid[(time_grid >= func_min) & (time_grid <= func_max)]

# Now we can call them
cph_surv_prob = np.vstack([fn(time_grid_clipped) for fn in cph_step_funcs])
ibs = integrated_brier_score(y_train, y_test, cph_surv_prob, time_grid_clipped)
print(f"Traditional Cox Model - IBS: {ibs:.4f}")

