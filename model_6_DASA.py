import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score
import random
import time
from pathlib import Path
import json
from util_dasa import getStatStr, calculate_auc  
from utils import create_unified_time_grid
from concordance import concordance_index
from sksurv.metrics import integrated_brier_score

class DRSA(nn.Module):
    def __init__(self, config):
        super(DRSA, self).__init__()
        
        self.emb_dim = config['emb_dim']    
        self.feature_size = config['feature_size']
        self.batch_size = config['batch_size']
        self.max_den = config.get('max_den', None)  # Not needed for float features
        self.max_seq_len = config['max_seq_len']
        self.state_size = config['state_size']
        self.middle_feature_size = config['middle_feature_size']
        self.device = config['device']

        # Layers - changed from Embedding to Linear projection for float features
        # Project each feature to emb_dim dimensions
        self.feature_projection = nn.Linear(self.feature_size, self.feature_size * self.emb_dim)
        self.middle_layer = nn.Linear(self.feature_size * self.emb_dim, self.middle_feature_size)
        self.lstm = nn.LSTM(self.middle_feature_size,
                         self.state_size,
                         batch_first=True)
        self.output_layer = nn.Linear(self.state_size, 1)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Feature projection (for float features)
        # x shape: [batch_size, feature_size] (float features)
        x_emb = self.feature_projection(x)  # [batch_size, feature_size * emb_dim]
        
        # Middle layer
        middle = torch.relu(self.middle_layer(x_emb))  # [batch_size, middle_feature_size]
        
        middle = middle.unsqueeze(1)
        middle = middle.repeat(1, self.max_seq_len, 1)
            
        # LSTM
        outputs, _ = self.lstm(middle)  # [batch_size, max_seq_len, state_size]
        
        # Output layer
        logits = self.output_layer(outputs)  # [batch_size, max_seq_len, 1]
        preds = torch.sigmoid(logits).squeeze(-1)  # [batch_size, max_seq_len]
        
        return preds

    def compute_loss(self, preds, label, event_time, observation_time, win):
        batch_size = preds.size(0)
        
        # Compute survival rate
        survival_rate = preds  # [batch_size, max_seq_len] # shape: [batch_size, max_seq_len]
        
        # Compute final rates
        survival_rates = []
        anlp_rates = []
        anlp_rates_prev = []

        for i in range(batch_size):
            event_time_i = event_time[i].long()
            observation_time_i = observation_time[i].long()
            
            surv = torch.prod(survival_rate[i, :observation_time_i])
            anlp = torch.prod(survival_rate[i, :event_time_i+1])
            anlp_prev = torch.prod(survival_rate[i, :event_time_i])
        
            survival_rates.append(surv)
            anlp_rates.append(anlp)
            anlp_rates_prev.append(anlp_prev)
            
        survival_rates = torch.stack(survival_rates)
        anlp_rates = torch.stack(anlp_rates)
        anlp_rates_prev = torch.stack(anlp_rates_prev)
        
        # Compute predictions
        dead_rates = 1 - survival_rates
        predict = torch.stack([survival_rates, dead_rates], dim=1)
        
        # Compute losses
        ce_loss = -torch.sum(label * torch.log(predict + 1e-10))/batch_size

        win_index = win == 1
        win_index = win_index.to(preds.device)
        lost_index = win == 0
        lost_index = lost_index.to(preds.device)
        # anlp_loss = -torch.sum(torch.log(anlp_rates_prev[win_index] - anlp_rates[win_index] + 1e-20)) / torch.sum(win_index)
        # t1 = torch.prod(t)
        if torch.sum(win_index) == 0:
            anlp_loss = torch.tensor(0.0, device=preds.device, requires_grad=True)
        else:
            anlp_loss = -torch.sum(torch.log(anlp_rates_prev[win_index] - anlp_rates[win_index] + 1e-20)) / torch.sum(win_index)
  
        return ce_loss, anlp_loss, predict


def evaluate(model, test_loader, train_loader, config):
    """
    Evaluate model on test set and compute c-index and IBS.
    
    Args:
        model: The DRSA model
        test_loader: Test data loader
        train_loader: Train data loader (for censoring distribution estimation)
        config: Configuration dictionary
        
    Returns:
        test_cindex: Concordance index
        mae_obs: Mean absolute error for observed events
        test_ibs: Integrated Brier Score
        total_surv_probs: Survival probabilities for all test samples
    """
    # Collect training data for censoring distribution estimation
    survival_train = []
    for batch in train_loader:
        # New format: {'x': tensor, 'event': tensor, 'time': tensor}
        event = batch['event']  # 1 if event occurred, 0 if censored
        time = batch['time']
        for i in range(len(event)):
            is_observed = int(event[i].item())  # 1 if event, 0 if censored
            time_val = float(time[i].item())
            survival_train.append((bool(is_observed), time_val))
    
    # Convert to structured array for scikit-survival
    survival_train = np.array(survival_train, dtype=[('event', bool), ('time', float)])
    
    model.eval()
    with torch.no_grad():
        pred_times, true_times, is_observed = [], [], []
        pred_obs_times, true_obs_times = [], []
        total_surv_probs = []
        survival_test = []
        
        for batch in test_loader:
            # New format: {'x': tensor, 'event': tensor, 'time': tensor}
            x = batch['x'].to(model.device)
            event = batch['event']  # 1 if event occurred, 0 if censored
            time = batch['time']
            
            # For evaluation, use time as the true time
            # event=1 means event occurred, so time is event_time
            # event=0 means censored, so time is observation_time
            
            # Get predictions
            preds = model(x)  # [batch_size, max_seq_len] - conditional survival rates
            # Compute cumulative survival probabilities
            surv_probs = torch.cumprod(preds, dim=1)  # [batch_size, max_seq_len]
            
            batch_size = x.size(0)
            for i in range(batch_size):
                is_obs = int(event[i].item())  # 1 if event occurred
                true_time = float(time[i].item())
                
                is_observed.append(is_obs)
                survival_test.append((bool(is_obs), true_time))
                
                # Store survival probabilities
                surv_probs_i = surv_probs[i].cpu().numpy()
                total_surv_probs.append(surv_probs_i)
                
                # Compute predicted time (mean of survival probabilities)
                if config.get('pred_method', 'mean') == 'mean':
                    pred_time = float(torch.sum(surv_probs[i]).item())
                elif config.get('pred_method', 'mean') == 'median':
                    pred_time = 0
                    while pred_time < len(surv_probs_i):
                        if surv_probs_i[pred_time] < 0.5:
                            break
                        pred_time += 1
                    if pred_time >= len(surv_probs_i):
                        pred_time = len(surv_probs_i) - 1
                else:
                    pred_time = float(torch.sum(surv_probs[i]).item())
                
                pred_times.append(pred_time)
                true_times.append(true_time)
                
                if is_obs:
                    pred_obs_times.append(pred_time)
                    true_obs_times.append(true_time)
        
        # Convert to numpy arrays
        total_surv_probs = np.array(total_surv_probs)
        survival_test = np.array(survival_test, dtype=[('event', bool), ('time', float)])
        
        pred_obs_times = np.asarray(pred_obs_times)
        true_obs_times = np.asarray(true_obs_times)
        mae_obs = np.mean(np.abs(pred_obs_times - true_obs_times)) if len(pred_obs_times) > 0 else 0.0
        
        pred_times = np.asarray(pred_times)
        true_times = np.asarray(true_times)
        is_observed = np.asarray(is_observed, dtype=bool)
        
        # Compute c-index
        test_cindex = concordance_index(true_times, pred_times, is_observed)
        
        # Compute IBS score using unified time grid
        test_times = survival_test['time'].astype(float)
        time_grid = create_unified_time_grid(test_times, n_points=200, 
                                            max_time=min(config['max_seq_len'], float(np.max(test_times))))
        
        # Interpolate survival probabilities to unified time grid
        # total_surv_probs is (N_test, max_seq_len) where indices correspond to times 0, 1, 2, ...
        # We need to interpolate to the unified grid
        time_indices = np.arange(total_surv_probs.shape[1], dtype=float)
        surv_probs_interp = np.zeros((total_surv_probs.shape[0], len(time_grid)))
        for i in range(total_surv_probs.shape[0]):
            surv_probs_interp[i] = np.interp(time_grid, time_indices, total_surv_probs[i])
        
        if len(time_grid) > 0:
            test_ibs = integrated_brier_score(survival_train, survival_test, 
                                             surv_probs_interp, time_grid)
        else:
            test_ibs = float('inf')
        
        print(f'c index: {test_cindex:.4f}, IBS: {test_ibs:.4f}, MAE (observed): {mae_obs:.4f}')
    
    return test_cindex, mae_obs, test_ibs, total_surv_probs


def checkpoint(model, total_surv_probs, best_test_ibs, best_test_cindex, config, optimizer=None, epoch=None):
    """
    Save model checkpoint + metrics + config, and survival probs array.
    """
    out_dir = Path("model_inventory")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    model_name = config.get("model_name", "dasa_model")
    ckpt_path = out_dir / f"{model_name}.pt"
    probs_path = out_dir / f"{model_name}_surv_probs_test.npy"
    meta_path = out_dir / f"{model_name}_meta.json"
    
    # Build checkpoint
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "metrics": {
            "best_test_ibs": float(best_test_ibs),
            "best_test_cindex": float(best_test_cindex),
        },
        "config": config,
        "framework": {
            "torch_version": torch.__version__,
        },
    }
    torch.save(ckpt, ckpt_path)
    print(f"[checkpoint] saved: {ckpt_path}")
    
    # Save survival probabilities
    if isinstance(total_surv_probs, torch.Tensor):
        total_surv_probs = total_surv_probs.detach().cpu().numpy()
    np.save(probs_path, total_surv_probs)
    print(f"[checkpoint] saved: {probs_path}")
    
    # Save metadata
    meta = {
        "model_path": str(ckpt_path),
        "probs_path": str(probs_path),
        "best_test_ibs": float(best_test_ibs),
        "best_test_cindex": float(best_test_cindex),
        "epoch": epoch,
        "model_name": model_name,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[checkpoint] saved: {meta_path}")


def train_model(model, train_dataloader, test_dataloader, config):
    optimizer_ce = optim.Adam(model.parameters(), lr=config['lr'])
    
    # Track best performance
    best_test_cindex, best_test_mae, best_test_ibs, best_epoch = -1, float('inf'), float('inf'), 0
    best_test_total_surv_probs = None
    
    for epoch in range(config['epochs']):
        print(f'epoch {epoch} {"="*50}')
        model.train()
        tot_loss = 0.0
        num_batches = 0
        
        for batch in train_dataloader:
            # New format: {'x': tensor, 'event': tensor, 'time': tensor, 'embedding': tensor (optional)}
            x = batch['x']
            event = batch['event']  # 1 if event occurred, 0 if censored
            time = batch['time']
            
            batch_size = x.size(0)
            num_batches += 1
            
            # Convert to old format expected by compute_loss
            win = event.clone()  # event=1 means win=1, event=0 means win=0

            event_time = torch.zeros(batch_size, device=x.device)
            observation_time = torch.zeros(batch_size, device=x.device)

            time = time.to(x.device)

            event_mask = (event == 1)
            censor_mask = (event == 0)

            if torch.any(event_mask):
                t_e = time[event_mask]
                event_time[event_mask] = t_e
                observation_time[event_mask] = t_e + torch.rand(t_e.shape, device=x.device) * t_e * 0.1

            if torch.any(censor_mask):
                t_o = time[censor_mask]
                observation_time[censor_mask] = t_o
                event_time[censor_mask] = t_o + torch.rand(t_o.shape, device=x.device) * t_o * 0.1

            
            # Create label: [survival_prob, death_prob]
            # If event occurred (win=1): label = [0, 1] (dead)
            # If censored (win=0): label = [1, 0] (still alive)
            label = torch.zeros(batch_size, 2, device=x.device)
            label[win == 1, 1] = 1.0  # Event occurred -> dead
            label[win == 0, 0] = 1.0  # Censored -> still alive

            # Move to device (they might already be on device from dataset)
            x = x.to(model.device)
            label = label.to(model.device)
            event_time = event_time.to(model.device)
            observation_time = observation_time.to(model.device)
            win = win.to(model.device)
            
            outputs = model(x)
            ce_loss, anlp_loss, predict = model.compute_loss(outputs, label, 
                                             event_time, observation_time, win)
            alpha = config.get('alpha', 1.0)
            beta = config.get('beta', 0.5)
            loss = alpha*ce_loss + beta*anlp_loss
            
            optimizer_ce.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            optimizer_ce.step()
            
            tot_loss += loss.item()
        
        print(f'train total loss: {tot_loss:.4f}')
        
        # Evaluate at every epoch to track best model
        if epoch >= 0:  # Test from first epoch
            print('Testing...')
            test_cindex, test_mae, test_ibs, test_total_surv_probs = evaluate(model, test_dataloader, train_dataloader, config)
            
            # Save if this is the best model so far (using c-index as primary criterion)
            if test_cindex > best_test_cindex:
                best_test_cindex = test_cindex
                best_test_mae = test_mae
                best_test_ibs = test_ibs
                best_epoch = epoch
                best_test_total_surv_probs = test_total_surv_probs
                # Save model immediately when we find a better one
                checkpoint(model, test_total_surv_probs, best_test_ibs, best_test_cindex, config, optimizer=optimizer_ce, epoch=epoch)
                print(f'NEW BEST MODEL saved at epoch {epoch}: test_cindex={test_cindex:.4f}, test_ibs={test_ibs:.4f}, test_mae={test_mae:.4f}')
            else:
                print(f'Epoch {epoch}: current cindex={test_cindex:.4f}, IBS={test_ibs:.4f}, MAE={test_mae:.4f}')
                print(f'BEST: cindex={best_test_cindex:.4f}, IBS={best_test_ibs:.4f}, MAE={best_test_mae:.4f} at epoch {best_epoch}')
    
    # Final summary
    print(f'\n=== TRAINING COMPLETED ===')
    print(f'Best test model found at epoch {best_epoch} with cindex={best_test_cindex:.4f}, IBS={best_test_ibs:.4f}, MAE={best_test_mae:.4f}')
    
    return best_test_cindex, best_test_mae, best_test_ibs, best_epoch

def run_test(model, global_step, test_dataloader, config):
    """
    Test function adapted for SurvivalDataSet from dataset_sa_standard.py
    """
    model.eval()  # Set model to evaluation mode
    ce_loss_arr = []
    anlp_arr = []
    auc_prob = []
    auc_label = []
    total_time = 0

    # Test on data
    with torch.no_grad():
        for batch in test_dataloader:
            # New format: {'x': tensor, 'event': tensor, 'time': tensor, 'embedding': tensor (optional)}
            test_batch_x = batch['x']
            test_batch_event = batch['event']  # 1 if event occurred, 0 if censored
            test_batch_time = batch['time']
            
            # Convert to old format expected by compute_loss
            # event=1 means event occurred (win=1), event=0 means censored (win=0)
            test_batch_win = test_batch_event.clone()
            
            # For event occurred: time is event_time, observation_time = event_time
            # For censored: time is observation_time, event_time is unknown (use max_seq_len as approximation)
            batch_size = test_batch_x.size(0)
            test_batch_event_time = test_batch_time.clone()
            test_batch_observation_time = test_batch_time.clone()
            
            # For censored cases, set event_time to a large value (max_seq_len)
            # since we don't know the true event time
            censored_mask = (test_batch_event == 0)
            if torch.any(censored_mask):
                test_batch_event_time[censored_mask] = config['max_seq_len']
            
            # Create label: [survival_prob, death_prob]
            # If event occurred (win=1): label = [0, 1] (dead)
            # If censored (win=0): label = [1, 0] (still alive)
            test_batch_label = torch.zeros(batch_size, 2, device=test_batch_x.device)
            test_batch_label[test_batch_win == 1, 1] = 1.0  # Event occurred -> dead
            test_batch_label[test_batch_win == 0, 0] = 1.0  # Censored -> still alive

            # Move to device (they might already be on device from dataset)
            test_batch_x = test_batch_x.to(model.device)
            test_batch_label = test_batch_label.to(model.device)
            test_batch_event_time = test_batch_event_time.to(model.device)
            test_batch_observation_time = test_batch_observation_time.to(model.device)
            test_batch_win = test_batch_win.to(model.device)
            
            start_time = time.time()
            # Forward pass
            preds = model(test_batch_x)
            ce_loss, anlp_loss, predict = model.compute_loss(preds, test_batch_label, 
                                                               test_batch_event_time, 
                                                               test_batch_observation_time, test_batch_win)

            total_time += time.time() - start_time

            # Store results
            auc_prob.append(predict.t()[0].cpu().detach())
            auc_label.append(test_batch_label.t()[0].cpu().detach())

            anlp_arr.append(anlp_loss.item())
            ce_loss_arr.append(ce_loss.item())

        mean_ce_loss = np.mean(ce_loss_arr)
        mean_anlp = np.mean(anlp_arr)
        # print(auc_label[0], auc_prob[0])
        auc_label = torch.cat(auc_label)
        auc_label = auc_label.reshape(1, -1)
        auc_prob = torch.cat(auc_prob)
        auc_prob = auc_prob.reshape(1, -1)
        mean_auc = calculate_auc(auc_label, auc_prob)

        log = getStatStr("TEST_DATA", "Test", global_step, mean_anlp, mean_ce_loss, mean_auc)
        print(log)
        
        return mean_ce_loss, mean_auc, mean_anlp

def test_model(model, test_dataloader, config):
    """
    Test function adapted for SurvivalDataSet from dataset_sa_standard.py
    """
    model.eval()  # Set model to evaluation mode
    test_ce_losses = []
    test_anlp_losses = []
    test_auc_label = []
    test_auc_prob = []
    
    with torch.no_grad():  # No gradient computation needed for testing
        for batch in test_dataloader:
            # New format: {'x': tensor, 'event': tensor, 'time': tensor, 'embedding': tensor (optional)}
            x = batch['x']
            event = batch['event']  # 1 if event occurred, 0 if censored
            time = batch['time']
            
            # Convert to old format expected by compute_loss
            batch_size = x.size(0)
            win = event.clone()  # event=1 means win=1, event=0 means win=0
            
            # For event occurred: time is event_time, observation_time = event_time
            # For censored: time is observation_time, event_time is unknown (use max_seq_len)
            event_time = time.clone()
            observation_time = time.clone()
            
            censored_mask = (event == 0)
            if torch.any(censored_mask):
                event_time[censored_mask] = config['max_seq_len']
            
            # Create label: [survival_prob, death_prob]
            label = torch.zeros(batch_size, 2, device=x.device)
            label[win == 1, 1] = 1.0  # Event occurred -> dead
            label[win == 0, 0] = 1.0  # Censored -> still alive
            
            # Move to device
            x = x.to(model.device)
            label = label.to(model.device)
            event_time = event_time.to(model.device)
            observation_time = observation_time.to(model.device)
            win = win.to(model.device)
            
            # Forward pass
            preds = model(x)
            ce_loss, anlp_loss, predict = model.compute_loss(preds, label, 
                                                           event_time, 
                                                           observation_time, win)
            
            # Track metrics
            test_ce_losses.append(ce_loss.item())
            test_anlp_losses.append(anlp_loss.item())
            test_auc_label.append(label)
            test_auc_prob.append(predict)
    
    # Calculate average loss and AUC
    avg_test_ce_loss = sum(test_ce_losses) / len(test_ce_losses)
    avg_test_anlp_loss = sum(test_anlp_losses) / len(test_anlp_losses)
    test_auc_label = torch.cat(test_auc_label)
    test_auc_label = test_auc_label.reshape(1, -1)
    test_auc_prob = torch.cat(test_auc_prob)
    test_auc_prob = test_auc_prob.reshape(1, -1)
    test_auc = calculate_auc(test_auc_label, test_auc_prob)
    
    print(f"Test Results:")
    print(f"Average CE Loss: {avg_test_ce_loss:.4f}")
    print(f"Average ANLP Loss: {avg_test_anlp_loss:.4f}")
    print(f"AUC Score: {test_auc:.4f}")
    
    return avg_test_ce_loss, avg_test_anlp_loss, test_auc


    