import torch
from sklearn.metrics import roc_auc_score
import h5py
from collections import defaultdict
import pandas as pd
import numpy as np

def add_time_step(x, MAX_SEQ_LEN):
    
    # Repeat x for max_seq times
    y = x.repeat(MAX_SEQ_LEN, 1)  # PyTorch's repeat instead of tf.tile
    
    # Create time steps
    t = torch.arange(MAX_SEQ_LEN, device=x.device).float()
    t = t.view(MAX_SEQ_LEN, 1)  # Reshape to [max_seq, 1]
    
    # Concatenate features and time
    z = torch.cat([y, t], dim=1)  # Concatenate along dim 1
    return z

def getStatStr(file_name, category, global_step, mean_ce_loss, mean_anlp_loss, mean_c_index):
        statistics_log = str(file_name) + "\t" + category + "\t" + "global_step:" + str(global_step) + "\t" \
                         "mean ce loss:" + "{:.6f}".format(mean_ce_loss) + "\t" + \
                         "mean anlp loss:" + "{:.6f}".format(mean_anlp_loss) + "\t" + \
                         "mean c-index:" + "{:.4f}".format(mean_c_index) + "\n"
        return statistics_log

def getStatStr_c_index(file_name, category, global_step, mean_c_index):
        statistics_log = str(file_name) + "\t" + category + "\t" + "global_step:" + str(global_step) + "\t" \
                         "mean c-index:" + "{:.4f}".format(mean_c_index) + "\n"
        return statistics_log

def get_best_device():
    """
    Simple function to get the best available device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # For Apple Silicon, sometimes MPS has issues with embeddings
        # You can force CPU usage by uncommenting the next line
        # return torch.device("cpu")
        return torch.device("mps")
    else:
        return torch.device("cpu")

# Complete fix:
def calculate_auc(train_auc_label, train_auc_prob):
    try:
        with torch.no_grad():  # Alternative way to handle requires_grad
            if isinstance(train_auc_label, torch.Tensor):
                labels = train_auc_label.cpu().numpy()
            else:
                labels = torch.cat(train_auc_label).cpu().numpy()
                
            if isinstance(train_auc_prob, torch.Tensor):
                probs = train_auc_prob.cpu().numpy()
            else:
                probs = torch.cat(train_auc_prob).cpu().numpy()
            
            labels = labels.flatten()
            probs = probs.flatten()
            
            return roc_auc_score(labels, probs)
            
    except Exception as e:
        print(f"AUC calculation error: {e}")
        return 0.5


def read_h5_file(file_path):
    datasets = defaultdict(list)
    with h5py.File(file_path, 'r') as f:
        for key in f.keys():
            # Handle both datasets and groups
            if isinstance(f[key], h5py.Dataset):
                datasets[key].append(f[key][()])
            elif isinstance(f[key], h5py.Group):
                # Recursively read all datasets in the group
                for subkey in f[key].keys():
                    if isinstance(f[key][subkey], h5py.Dataset):
                        datasets[f"{key}/{subkey}"].append(f[key][subkey][()])
    return datasets

def format_dataset_to_df(dataset, duration_col, event_col, trt_idx = None):
    xdf = pd.DataFrame(dataset['train/x'])
    if trt_idx is not None:
        xdf = xdf.rename(columns={trt_idx : 'treat'})

    dt = pd.DataFrame(dataset['train/t'], columns=[duration_col])
    censor = pd.DataFrame(dataset['train/e'], columns=[event_col])
    cdf = pd.concat([xdf, dt, censor], axis=1)
    return cdf


def create_unified_time_grid(time_test: np.ndarray, n_points: int = 200, 
                             min_time: float = None, max_time: float = None) -> np.ndarray:
    """
    Create a unified time grid for IBS computation across all models.
    
    Args:
        time_test: Test set time values (used to determine valid range)
        n_points: Number of points in the grid (default: 200)
        min_time: Optional minimum time (defaults to ceil(min(time_test)))
        max_time: Optional maximum time (defaults to floor(max(time_test)))
    
    Returns:
        time_grid: Array of time points for IBS evaluation
                   Grid is [t_min, t_max) (exclusive upper bound to stay strictly below max)
    
    Note:
        This ensures all models use the same time grid for fair IBS comparison.
        The grid excludes the endpoint to satisfy scikit-survival's requirement
        that time_grid < max_test_time.
    """
    time_test = np.asarray(time_test, dtype=float)
    
    if len(time_test) == 0:
        raise ValueError("time_test cannot be empty")
    
    # Determine bounds
    if min_time is None:
        t_min = float(np.ceil(np.min(time_test)))
    else:
        t_min = float(np.ceil(min_time))
    
    if max_time is None:
        t_max = float(np.floor(np.max(time_test)))
    else:
        t_max = float(np.floor(max_time))
    
    # Ensure valid range
    if t_min >= t_max:
        # Fallback: use a small range if min >= max
        t_min = float(np.min(time_test))
        t_max = float(np.max(time_test))
        if t_min >= t_max:
            # If still invalid, create a minimal grid
            return np.array([t_min], dtype=float)
    
    # Create grid using linspace with endpoint=False to stay strictly below t_max
    time_grid = np.linspace(t_min, t_max, n_points, endpoint=False)
    
    return time_grid
