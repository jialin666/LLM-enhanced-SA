import torch
from sklearn.metrics import roc_auc_score


def add_time_step(x, MAX_SEQ_LEN):
    
    # Repeat x for max_seq times
    y = x.repeat(MAX_SEQ_LEN, 1)  # PyTorch's repeat instead of tf.tile
    
    # Create time steps
    t = torch.arange(MAX_SEQ_LEN, device=x.device).float()
    t = t.view(MAX_SEQ_LEN, 1)  # Reshape to [max_seq, 1]
    
    # Concatenate features and time
    z = torch.cat([y, t], dim=1)  # Concatenate along dim 1
    return z

def getStatStr(file_name, category, global_step, mean_anlp_loss, mean_ce_loss, mean_auc):
        statistics_log = str(file_name) + "\t" + category + "\t" + "global_step:" + str(global_step) + "\t" \
                         "mean anlp loss:" + "{:.6f}".format(mean_anlp_loss) + "\t" + \
                         "mean ce loss:" + "{:.4f}".format(mean_ce_loss) + "\t" + \
                         "mean auc:" + "{:.4f}".format(mean_auc) + "\n"
        return statistics_log

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
    
if __name__ == "__main__":
      x = torch.randn(1, 3)
      print(add_time_step(x, 10))