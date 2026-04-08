import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from pathlib import Path
import json
from torch.utils.data import DataLoader
from utils import get_best_device, create_unified_time_grid
from concordance import concordance_index
from sksurv.metrics import integrated_brier_score

device = get_best_device()

class InputEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Total input dimension: categorical features + numerical features
        input_dim = config['input_dim']
        self.w = nn.Linear(input_dim, config['d_model'])
        self.norm = nn.LayerNorm(config['d_model'])
        
    def forward(self, x, x_embedding=None):
        # x shape: (batch_size, num_features) or (batch_size, max_time, num_features)
        # Convert to float if needed
        if x.dtype != torch.float32:
            x = x.float()
        
        # Apply linear transformation and layer norm
        output = self.norm(self.w(x))
        
        # If input is 2D (batch_size, num_features), expand to (batch_size, max_time, d_model)
        if output.dim() == 2:
            output = output.unsqueeze(1)
            output = output.repeat(1, self.config['max_seq_len'], 1)
        
        return output  # shape: (B, max_time, d_model)

# ============================================
# Positional Encoding (same for both)
# ============================================
class PositionalEncoding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config['d_model']
        self.max_len = config['max_seq_len']

        # Create a matrix of shape (max_len, d_model)
        pe = torch.zeros(self.max_len, self.d_model)

        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))

        # sin for even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # cos for odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        # Unsqueeze to match input shape: (1, max_len, d_model)
        pe = pe.unsqueeze(0)

        # Register as buffer (non-trainable, moves with .to(device))
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x


# ============================================
# Causal Multi-head Attention (DECODER)
# ============================================
class CausalMultiheadAttention(nn.Module):
    """Decoder-style causal self-attention with masking"""
    
    def __init__(self, config, dropout=0.1):
        super(CausalMultiheadAttention, self).__init__()
        self.d_model = config['d_model']
        self.n_heads = config['n_heads']
        self.head_dim = config['d_model'] // config['n_heads']
        assert self.head_dim * config['n_heads'] == config['d_model'], "d_model must be divisible by n_heads"

        self.w_q = nn.Linear(config['d_model'], config['d_model'])
        self.w_k = nn.Linear(config['d_model'], config['d_model'])
        self.w_v = nn.Linear(config['d_model'], config['d_model'])
        self.w_o = nn.Linear(config['d_model'], config['d_model'])

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

        # ⭐ KEY CHANGE: Register causal mask buffer
        # Lower triangular matrix: allows attending to current and past positions only
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config['max_seq_len'], config['max_seq_len'])).view(1, 1, config['max_seq_len'], config['max_seq_len'])
        )
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: Optional additional mask (batch_size, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.size()

        # Compute Q, K, V
        query = self.w_q(x)
        key = self.w_k(x)
        value = self.w_v(x)

        # Reshape for multi-head attention
        # (batch_size, seq_len, d_model) -> (batch_size, n_heads, seq_len, head_dim)
        query = query.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        # scores shape: (batch_size, n_heads, seq_len, seq_len)

        # Mask out future positions by setting them to -inf
        causal_mask = self.causal_mask[:, :, :seq_len, :seq_len]
        scores = scores.masked_fill(causal_mask == 0, float('-inf'))

        # Apply additional mask if provided (e.g., padding mask)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Compute attention output
        output = torch.matmul(attn_weights, value)
        # output shape: (batch_size, n_heads, seq_len, head_dim)

        # Concatenate heads
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.d_model)

        # Final linear projection
        output = self.w_o(output)

        return output


# ============================================
# Residual Connection (same for both)
# ============================================
class ResidualConnection(nn.Module):
    def __init__(self, config, dropout=0.1):
        super(ResidualConnection, self).__init__()
        self.norm = nn.LayerNorm(config['d_model'])   
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # Pre-norm: norm -> sublayer -> dropout -> residual
        return x + self.dropout(sublayer(self.norm(x)))


# ============================================
# Feed Forward Block (same for both)
# ============================================
class FeedForwardBlock(nn.Module):
    def __init__(self, config, dropout=0.1):
        super(FeedForwardBlock, self).__init__()
        self.linear_1 = nn.Linear(config['d_model'], config['d_ff'])
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(config['d_ff'], config['d_model'])

    def forward(self, x):
        return self.linear_2(self.dropout(F.relu(self.linear_1(x))))


# ============================================
# Decoder Block (uses CausalMultiheadAttention)
# ============================================
class DecoderBlock(nn.Module):
    """Single decoder layer with causal self-attention"""
    
    def __init__(self, causal_attention_block, feed_forward_block, config, dropout=0.1):
        super(DecoderBlock, self).__init__()
        self.causal_attention_block = causal_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([
            ResidualConnection(config, dropout) for _ in range(2)
        ])

    def forward(self, x, mask=None):
        # Causal self-attention with residual connection
        x = self.residual_connections[0](x, lambda x: self.causal_attention_block(x, mask))
        # Feed-forward with residual connection
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


# ============================================
# Final Layer (same concept, but for decoder)
# ============================================
class DecoderFinalLayer(nn.Module):
    """Final projection layer for decoder output"""
    
    def __init__(self, config):
        super(DecoderFinalLayer, self).__init__()
        self.w_1 = nn.Linear(config['d_model'], config['d_model'] // 2)
        self.norm = nn.LayerNorm(config['d_model'] // 2)
        self.w_2 = nn.Linear(config['d_model'] // 2, 1)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            output: (batch_size, seq_len) - predictions for each position
        """
        x = F.relu(self.w_1(x))
        x = self.norm(x)
        x = self.w_2(x)
        return torch.sigmoid(x.squeeze(-1))


# ============================================
# Decoder (Main Model)
# ============================================
class Decoder(nn.Module):
    """Decoder-only Transformer (GPT-style)"""
    
    def __init__(self, layers, positional_encoding, embed_layer, config):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.positional_encoding = positional_encoding
        self.embed_layer = embed_layer
        self.config = config
        self.final_layer = DecoderFinalLayer(config)

    def forward(self, x, x_embedding=None, mask=None):
        """
        Args:
            x: Input tensor
            x_embedding: Optional pre-computed embeddings (not used)
            mask: Optional padding mask
        Returns:
            output: (batch_size, seq_len) - predictions
        """
        # Embed input
        x = self.embed_layer(x)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Pass through decoder layers
        for layer in self.layers:
            x = layer(x, mask)
        
        # Final projection
        x = self.final_layer(x)
        return x


# ============================================
# Model Builder Function
# ============================================
def build_decoder_transformer(config):
    """
    Build a decoder-only transformer model
    
    Args:
        config: dict with keys:
            - d_model: model dimension
            - n_heads: number of attention heads
            - d_ff: feedforward dimension
            - n_layers: number of decoder layers
            - dropout: dropout rate
            - max_seq_len: maximum sequence length
        embed_layer: embedding layer module
    
    Returns:
        Decoder model
    """

    embed_layer = InputEmbeddings(config)
    
    # Positional encoding
    pos_encoding = PositionalEncoding(config)
    
    # Build decoder layers
    decoder_layers = []
    for _ in range(config['n_layers']):
        # Causal self-attention
        causal_attn = CausalMultiheadAttention(
            config, dropout=config['dropout']
        )
        # Feed-forward network
        ffn = FeedForwardBlock(config, dropout=config['dropout'])
        # Decoder block
        decoder_block = DecoderBlock(causal_attn, ffn, config, dropout=config['dropout'])
        decoder_layers.append(decoder_block)
    
    # Build final decoder model
    decoder = Decoder(decoder_layers, pos_encoding, embed_layer, config)
    
    return decoder


def evaluate(decoder, test_loader, train_loader, config):
    # The reason to use train_loader is to get the censoring distribution, which is used to compute the IBS score.
    # iterate through the train_loader and get the censoring distribution.
    survival_train = []
    for features, times, mask, label, is_observed_single, embedding in train_loader:
        for i in range(len(is_observed_single[0])):
            survival_train.append((is_observed_single[0][i].item(), times[0][i].item()))
    # convert the survival_train to a structured array
    survival_train = np.array(survival_train, dtype=[('event', bool), ('time', float)])

    decoder.eval()  
    with torch.no_grad():
        pred_times, true_times, is_observed = [], [], []
        pred_obs_times, true_obs_times = [], []
        total_surv_probs = []
        survival_test = []

        # NOTE batch size is 1
        for features, times, mask, label, is_observed_single, embedding in test_loader:
            is_observed.append(is_observed_single.item())
            # Handle embedding based on config
            embedding = embedding if config['use_embedding_input'] else None
            sigmoid_preds = decoder.forward(features, embedding)
            surv_probs = torch.cumprod(sigmoid_preds, dim=1).squeeze()
            total_surv_probs.append(surv_probs)
            
            survival_test.append((is_observed_single.item(), times.squeeze().item()))

            if config['pred_method'] == 'mean':
                pred_time = torch.sum(surv_probs).item()
            elif config['pred_method'] == 'median':
                pred_time = 0
                while True:
                    if surv_probs[pred_time] < 0.5:
                        break
                    else:
                        pred_time += 1
                        if pred_time == len(surv_probs):
                            break

            true_time = times.squeeze().item()
            pred_times.append(pred_time)
            true_times.append(true_time)

            if is_observed_single:
                pred_obs_times.append(pred_time)
                true_obs_times.append(true_time)

        total_surv_probs = torch.stack(total_surv_probs).cpu().numpy()
        survival_test = np.array(survival_test, dtype=[('event', bool), ('time', float)])

        
        pred_obs_times = np.asarray(pred_obs_times)
        true_obs_times = np.asarray(true_obs_times)
        mae_obs = np.mean(np.abs(pred_obs_times - true_obs_times))

        pred_times = np.asarray(pred_times)
        true_times = np.asarray(true_times)
        is_observed = np.asarray(is_observed, dtype=bool)

        test_cindex = concordance_index(true_times, pred_times, is_observed)
        
        # Use unified time grid for consistent IBS computation
        test_times = survival_test['time'].astype(float)
        max_time_allowed = min(
            config['max_seq_len'] - 1,  # last index model predicts
            float(np.max(survival_train['time'].astype(float))),
            float(np.max(test_times)),
        )
        time_grid = create_unified_time_grid(test_times, n_points=200, 
                                            max_time=max_time_allowed)

        # Interpolate survival probabilities to unified time grid
        # total_surv_probs is (N_test, max_seq_len) where indices correspond to times 0, 1, 2, ...
        # We need to interpolate to the unified grid
        time_indices = np.arange(total_surv_probs.shape[1], dtype=float)
        surv_probs_for_ibs = np.zeros((total_surv_probs.shape[0], len(time_grid)))
        for i in range(total_surv_probs.shape[0]):
            surv_probs_for_ibs[i] = np.interp(time_grid, time_indices, total_surv_probs[i])

        # Now all times are within [min_test_time, max_test_time) and [min_train, max_train)
        test_ibs = integrated_brier_score(
            survival_train,
            survival_test,
            surv_probs_for_ibs,
            time_grid,
        )
                    
        print('c index', test_cindex, 'IBS', test_ibs)
                    

    return test_cindex, mae_obs, test_ibs, total_surv_probs

def checkpoint(model,
               total_surv_probs,          # torch.Tensor (n, T) or (B, T, ...)
               best_test_ibs: float,
               best_test_cindex: float,
               config: dict,
               optimizer=None,
               epoch: int = None,
               step: int = None):
    """
    Save model checkpoint + metrics + config, and survival probs array.
    """
    out_dir = Path("model_inventory")
    out_dir.mkdir(parents=True, exist_ok=True)

    model_name = config.get("model_name", "model")
    ckpt_path = out_dir / f"{model_name}.pt"
    probs_path = out_dir / f"{model_name}_surv_probs_test.npy"
    meta_path = out_dir / f"{model_name}_meta.json"

    # Build a compact checkpoint (avoid pickling full model)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "step": step,
        "metrics": {
            "best_test_ibs": float(best_test_ibs),
            "best_test_cindex": float(best_test_cindex),
        },
        "config": config,  # ensure it's JSON-serializable if you plan to dump it separately
        # optionally record library versions
        "framework": {
            "torch_version": torch.__version__,
        },
    }
    torch.save(ckpt, ckpt_path)
    print(f"[checkpoint] saved: {ckpt_path}")

    # Save survival probabilities separately for easy loading/plotting
    if isinstance(total_surv_probs, torch.Tensor):
        total_surv_probs = total_surv_probs.detach().cpu().numpy()
    np.save(probs_path, total_surv_probs)
    print(f"[checkpoint] saved: {probs_path}")

    # (Optional) small JSON meta file for quick inspection/versioning
    meta = {
        "model_path": str(ckpt_path),
        "probs_path": str(probs_path),
        "best_test_ibs": float(best_test_ibs),
        "best_test_cindex": float(best_test_cindex),
        "epoch": epoch,
        "step": step,
        "model_name": model_name,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[checkpoint] saved: {meta_path}")

def load_checkpoint(model, optimizer=None, path="model_inventory/my_model.pt", map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt  # contains metrics/config/epoch for reference




def train(train_loader, test_loader, decoder, config):
    
    # NOTE using C index as early stopping criterion
    best_test_cindex, best_test_mae, best_test_ibs, best_epoch = -1, 9999999, 9999999, 0
    best_test_total_surv_probs = None  # Store test predictions of best model

    optimizer = torch.optim.Adam(decoder.parameters(), lr=config['lr'])

    for t in range(config['num_epochs']):
        print('epoch', t, '************************************************')
        decoder.train()

        tot_loss = 0.
        batch_idx = 0
        for features, true_durations, mask, label, is_observed, embedding in train_loader:

            batch_idx += 1
            if batch_idx % 10 == 0:
                print('batch', batch_idx)
            # in the training stage, all these are list of length 2
            optimizer.zero_grad()

            is_observed_a = is_observed[0]
            mask_a = mask[0]
            mask_b = mask[1]
            label_a = label[0]
            label_b = label[1]
            true_durations_a = true_durations[0]
            true_durations_b = true_durations[1]

            # Handle embedding based on config
            embedding_a = embedding[0] if config['use_embedding_input'] else None
            embedding_b = embedding[1] if config['use_embedding_input'] else None

            sigmoid_a = decoder.forward(features[0], embedding_a) # input shape: (batch_size, max_time, num_features), output shape: (batch_size, max_time)

            surv_probs_a = torch.cumprod(sigmoid_a, dim=1) # input shape: (batch_size, max_time), output shape: (batch_size, max_time)
            loss = nn.BCELoss()(surv_probs_a * mask_a, label_a * mask_a) # formular 14, 15

            sigmoid_b = decoder.forward(features[1], embedding_b)
            surv_probs_b = torch.cumprod(sigmoid_b, dim=1)
            cond_a = is_observed_a & (true_durations_a < true_durations_b) # a is observed and b survival longer than a
            if torch.sum(cond_a) > 0:
                mean_lifetimes_a = torch.sum(surv_probs_a, dim=1)
                mean_lifetimes_b = torch.sum(surv_probs_b, dim=1)
                diff = mean_lifetimes_b[cond_a] - mean_lifetimes_a[cond_a]
                true_diff = true_durations_b[cond_a] - true_durations_a[cond_a]
                loss += config['coeff'] * torch.mean(nn.ReLU()(true_diff - diff)) # formula 16-1

            cond_a2 = is_observed_a
            if torch.sum(cond_a2) > 0:
                mean_lifetimes_a = torch.sum(surv_probs_a, dim=1)
                loss += config['coeff2'] * F.l1_loss(mean_lifetimes_a[cond_a2], true_durations_a[cond_a2]) # formula 16-2

            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
        print('train total loss', tot_loss)

        # Evaluate at every epoch to track best model
        if t > 0:
            print('Test')
            test_cindex, test_mae, test_ibs, test_total_surv_probs = evaluate(decoder, test_loader, train_loader, config)
            
            # Save if this is the best model so far
            if test_cindex > best_test_cindex:
                best_test_cindex = test_cindex
                best_test_mae = test_mae
                best_test_ibs = test_ibs
                best_epoch = t
                # Save model immediately when we find a better one
                checkpoint(decoder, test_total_surv_probs, best_test_ibs, best_test_cindex, config, epoch=t)
                print(f'NEW BEST MODEL saved at epoch {t}: test_cindex={test_cindex:.4f}, test_ibs={test_ibs:.4f}')
            
           
    # Final evaluation of best model on test set
    print(f'\n=== TRAINING COMPLETED ===')
    print(f'Best test model found at epoch {best_epoch} with cindex={best_test_cindex:.4f}, IBS={best_test_ibs:.4f}')
    
    # If we never got test results for the best model, evaluate it now
    if best_test_total_surv_probs is None:
        print('Evaluating best model on test set...')
        test_cindex, test_mae, test_ibs, test_total_surv_probs = evaluate(decoder, test_loader, train_loader, config)
        checkpoint(decoder, test_total_surv_probs, best_test_ibs, best_test_cindex, config)
        print(f'Final best model test performance: cindex={test_cindex:.4f}, mae={test_mae:.4f}, IBS={test_ibs:.4f}')
    else:
        print(f'Best model test performance: cindex={best_test_cindex:.4f}, mae={best_test_mae:.4f}, IBS={best_test_ibs:.4f}')
    
    return best_test_cindex, best_test_mae, best_test_ibs, best_epoch


def main(train_loader, test_loader, config):
     
    decoder = build_decoder_transformer(config)
    decoder.to(device)
    best_test_cindex, best_test_mae, best_test_ibs, best_epoch = train(train_loader, test_loader, decoder, config)
    return best_test_cindex, best_test_mae, best_test_ibs, best_epoch

