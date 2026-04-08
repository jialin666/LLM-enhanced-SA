import pandas as pd
import numpy as np
import torch
from utils import create_unified_time_grid
from sksurv.ensemble import RandomSurvivalForest
from sklearn.model_selection import train_test_split

df_all = pd.read_csv('data/1_gbsg_all.csv')
feature_cols = ['meno', 'grade', 'hormon', 'age', 'size', 'nodes', 'pgr', 'er']

df_train, df_test = train_test_split(df_all, test_size=0.3)

# organize the data, let the event and time be the last two columns
df_train = df_train[feature_cols + ['event', 'time']]
df_test = df_test[feature_cols + ['event', 'time']]


# compute the ibs score based on scikit-survival
from sksurv.metrics import integrated_brier_score

# Convert DataFrames to structured arrays for scikit-survival
y_train = np.array([(bool(e), float(t)) for e, t in zip(df_train['event'], df_train['time'])],
                   dtype=[('event', bool), ('time', float)])
y_test = np.array([(bool(e), float(t)) for e, t in zip(df_test['event'], df_test['time'])],
                  dtype=[('event', bool), ('time', float)])
# Note: Time grid is now created using unified function in the RSF section below


# Random Survival Forest (scikit-survival)
# Create and fit Random Survival Forest
rsf = RandomSurvivalForest(n_estimators=100, random_state=42)
rsf.fit(df_train[feature_cols].values, y_train)
# Make predictions
rsf_pred = rsf.predict(df_test[feature_cols].values)
# Compute concordance index
from sksurv.metrics import concordance_index_censored
c_index, _, _, _, _ = concordance_index_censored(
    y_test['event'], y_test['time'], rsf_pred, tied_tol=1e-08
)
print("Random Survival Forest Results:")
print(f"Concordance Index: {c_index:.4f}")

# Compute BS and IBS for Random Survival Forest
# Get survival function from Random Survival Forest
rsf_surv = rsf.predict_survival_function(df_test[feature_cols], return_array=False)
# Convert to DataFrame format
rsf_surv_times = []
rsf_surv_probs = []
for i, surv_func in enumerate(rsf_surv):
    times = surv_func.x
    probs = surv_func.y
    rsf_surv_times.append(times)
    rsf_surv_probs.append(probs)

# Create unified time grid for IBS computation
time_grid_rsf = create_unified_time_grid(df_test['time'].values, n_points=200)

# Interpolate survival functions to common time grid
rsf_surv_interp = np.zeros((len(rsf_surv), len(time_grid_rsf)))
for i, (times, probs) in enumerate(zip(rsf_surv_times, rsf_surv_probs)):
    rsf_surv_interp[i] = np.interp(time_grid_rsf, times, probs)

rsf_surv_df = pd.DataFrame(rsf_surv_interp, columns=time_grid_rsf)
# y_test is a structured array, use field names not indices
# compute the ibs score based on scikit-survival
ibs = integrated_brier_score(y_train, y_test, rsf_surv_interp, time_grid_rsf)
print(f"Random Survival Forest - IBS: {ibs:.4f}")
    
