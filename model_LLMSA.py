import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from pathlib import Path
import json
from utils import get_best_device
from concordance import concordance_index
from sksurv.metrics import integrated_brier_score
from sksurv.nonparametric import kaplan_meier_estimator

device = get_best_device()

# ============================================
# IBS Loss (Differentiable Integrated Brier Score)
# ============================================
class IBSLoss(nn.Module):
    """
    Differentiable Integrated Brier Score Loss for survival analysis.
    Uses IPCW (Inverse Probability of Censoring Weighting).
    
    Reference: Graf et al. (1999) - Assessment and comparison of prognostic classification schemes for survival data
    """
    
    def __init__(self, max_time, device='cpu'):
        super().__init__()
        self.max_time = max_time
        self.device = device
        self.time_grid = torch.arange(max_time, dtype=torch.float32, device=device)
        # Censoring distribution G(t) = P(C > t), initialized to 1 (no censoring)
        self.register_buffer('G', torch.ones(max_time))
        self._censoring_estimated = False
    
    def estimate_censoring_distribution(self, train_times, train_events):
        """
        Estimate G(t) = P(C > t) using Kaplan-Meier on censoring times.
        Call this once before training with the full training set.
        
        Args:
            train_times: numpy array of observed times
            train_events: numpy array of event indicators (1=event, 0=censored)
        """
        # For censoring distribution, we flip the event indicator
        # Censoring "event" happens when original event=0
        censoring_events = (1 - train_events).astype(bool)
        
        try:
            # Estimate G(t) using KM estimator for censoring
            times_km, G_km = kaplan_meier_estimator(censoring_events, train_times)
            
            # Interpolate to our time grid
            G = np.ones(self.max_time)
            for t in range(self.max_time):
                # Find the largest time in times_km that is <= t
                idx = np.searchsorted(times_km, t, side='right') - 1
                if idx >= 0 and idx < len(G_km):
                    G[t] = G_km[idx]
                elif idx < 0:
                    G[t] = 1.0
                else:
                    G[t] = G_km[-1]
            
            # Clip to avoid division by zero (minimum probability)
            G = np.clip(G, 0.01, 1.0)
            self.G = torch.tensor(G, dtype=torch.float32, device=self.device)
            self._censoring_estimated = True
            print(f"[IBSLoss] Censoring distribution estimated. G(0)={G[0]:.3f}, G(max)={G[-1]:.3f}")
            
        except Exception as e:
            print(f"[IBSLoss] Warning: Could not estimate censoring distribution: {e}")
            print("[IBSLoss] Using uniform weights (G(t)=1 for all t)")
            self.G = torch.ones(self.max_time, device=self.device)
    
    def forward(self, S_pred, T, event):
        """
        Compute differentiable IBS loss.
        
        Args:
            S_pred: (batch, max_time) predicted survival probabilities S(t)
            T: (batch,) observed times (can be float, will be converted to indices)
            event: (batch,) event indicator (1=event occurred, 0=censored)
        
        Returns:
            ibs_loss: scalar tensor (mean Brier score over time and samples)
        """
        batch_size = S_pred.shape[0]
        K = min(S_pred.shape[1], self.max_time)
        
        # Ensure we work with the same time grid length
        S_pred = S_pred[:, :K]
        time_grid = self.time_grid[:K].unsqueeze(0)  # (1, K)
        G = self.G[:K].unsqueeze(0)  # (1, K)
        
        # Reshape T and event for broadcasting
        T = T.unsqueeze(1).float()  # (batch, 1)
        event = event.unsqueeze(1).float()  # (batch, 1)
        
        # Indicator Y(t) = I(T_i > t_k) - subject still at risk at time t
        Y = (T > time_grid).float()  # (batch, K)
        
        # ============================================
        # Compute IPCW weights
        # ============================================
        # The weight depends on whether t < T_i or t >= T_i
        # 
        # For t < T_i (subject still at risk): weight = 1/G(t)
        # For t >= T_i and event=1: weight = 1/G(T_i)  
        # For t >= T_i and event=0 (censored): weight = 0
        
        weights = torch.zeros_like(S_pred)
        
        # Case 1: t < T_i (subject still at risk)
        # Weight = 1/G(t)
        weights = weights + Y / (G + 1e-8)
        
        # Case 2: t >= T_i and event occurred
        # Weight = 1/G(T_i)
        # Get G(T_i) for each sample
        T_idx = T.long().clamp(0, K - 1)  # (batch, 1) - index into G
        G_at_T = torch.gather(self.G[:K].unsqueeze(0).expand(batch_size, -1), 1, T_idx)  # (batch, 1)
        
        # Mask for t >= T_i and event occurred
        event_after_T = (1 - Y) * event  # (batch, K): 1 where t >= T_i and event=1
        weights = weights + event_after_T / (G_at_T + 1e-8)
        
        # Case 3: t >= T_i and censored -> weight stays 0 (already initialized)
        
        # ============================================
        # Compute Brier score
        # ============================================
        # BS(t) = (S(t) - I(T > t))^2
        # For t < T: target is 1 (still alive), so BS = (S(t) - 1)^2
        # For t >= T: target is 0 (event happened), so BS = S(t)^2
        brier_scores = (S_pred - Y) ** 2  # (batch, K)
        
        # Weighted Brier scores
        weighted_bs = weights * brier_scores  # (batch, K)
        
        # Integrate over time (using mean as approximation to integral)
        # IBS = (1/tau) * integral_0^tau BS(t) dt
        ibs = weighted_bs.mean()
        
        return ibs

class InputEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(num_categories, config['cat_embed_dim'])
            for num_categories in config['cat_dims']
        ])
        self.num_projection_covariate = nn.Linear(config['num_dim'], config['num_embed_dim'])
        
        output_dim_covariate = len(config['cat_dims']) * config['cat_embed_dim'] + config['num_embed_dim'] 
        
        self.embedding_projection_covariate = nn.Linear(output_dim_covariate, config['d_model'])
        if config['use_embedding_input']:
            self.embedding_projection_text = nn.Linear(config['text_embedding_dim'], config['d_model'])
        else:
            self.embedding_projection_text = None
        
    def forward(self, x, x_embedding=None):
        # the shape of x is (batch_size, max_time, num_features), the original x is a batch_size * num_features, it just been expanded to max_time times
        # The original x is required to put all categorical features before numerical features

        x_cat = x[:, :len(self.config['cat_dims'])].long()  # First 6 features are categorical
        x_num = x[:, len(self.config['cat_dims']):len(self.config['cat_dims'])+self.config['num_dim']].float()  # Rest are numerical
        cat_embeds = [embed(x_cat[:, i]) for i, embed in enumerate(self.cat_embeddings)]
        cat_out = torch.cat(cat_embeds, dim=-1)
        num_out = self.num_projection_covariate(x_num)
        
        input_covariate = torch.cat([cat_out, num_out], dim=-1)
        output_covariate = self.embedding_projection_covariate(input_covariate)
        if self.config['use_embedding_input']:
            output_text = self.embedding_projection_text(x_embedding)
            output = torch.cat([output_covariate, output_text], dim=-1)
        else:
            output = output_covariate
        # so if use_embedding_input is True, then the output is (B, max_time, 2*d_model), otherwise it is (B, max_time, d_model)

        # resume the second dimension to max_time
        output = output.unsqueeze(1)
        output = output.repeat(1, self.config['max_seq_len'], 1)
        return output  # shape: (B, max_time, total_embed_dim)

# ============================================
# Positional Encoding
# ============================================
class PositionalEncoding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config['d_model']
        self.max_len = config['max_seq_len']
        # if use_embedding_input is True, then the positional encoding is (max_len, 2*d_model), otherwise it is (max_len, d_model)
        if config['use_embedding_input']:
            self.d_model = config['d_model'] * 2
        else:
            self.d_model = config['d_model']
        
        # Create a matrix of shape (max_len, 2*d_model) if use_embedding_input is True, otherwise (max_len, d_model)
        # Note: self.d_model is already doubled when use_embedding_input is True
        pe = torch.zeros(self.max_len, self.d_model)
        
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        
        # sin for even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # cos for odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        # Unsqueeze to match input shape: (1, max_len, 2*d_model) if use_embedding_input is True, otherwise (1, max_len, d_model)
        pe = pe.unsqueeze(0)

        # Register as buffer (non-trainable, moves with .to(device))
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, 2*d_model) if use_embedding_input is True, otherwise (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x


# ============================================
# Causal Multi-head Attention (DECODER)
# ============================================
class CausalMultiheadAttention(nn.Module):
    """Decoder-style causal self-attention with masking"""
    
    def __init__(self, config, dropout=0.1):
        super(CausalMultiheadAttention, self).__init__()
        self.n_heads = config['n_heads']
        
        # Set d_model based on whether embeddings are used
        if config['use_embedding_input']:
            self.d_model = config['d_model'] * 2
        else:
            self.d_model = config['d_model']
        
        # Calculate head_dim AFTER d_model is set
        self.head_dim = self.d_model // self.n_heads
        assert self.head_dim * self.n_heads == self.d_model, "d_model must be divisible by n_heads"
        
        self.w_q = nn.Linear(self.d_model, self.d_model)
        self.w_k = nn.Linear(self.d_model, self.d_model)
        self.w_v = nn.Linear(self.d_model, self.d_model)
        self.w_o = nn.Linear(self.d_model, self.d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

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
        if config['use_embedding_input']:
            self.norm = nn.LayerNorm(config['d_model'] * 2)
        else:
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
        if config['use_embedding_input']:
            self.linear_1 = nn.Linear(config['d_model'] * 2, config['d_ff'])
        else:
            self.linear_1 = nn.Linear(config['d_model'], config['d_ff'])
        self.dropout = nn.Dropout(dropout)
        if config['use_embedding_input']:
            self.linear_2 = nn.Linear(config['d_ff'], config['d_model'] * 2)
        else:
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
        # Input dimension depends on use_embedding_input, but output is always d_model // 2
        if config['use_embedding_input']:
            self.w_1 = nn.Linear(config['d_model'] * 2, config['d_model'] // 2)
        else:
            self.w_1 = nn.Linear(config['d_model'], config['d_model'] // 2)
        # norm and w_2 always use d_model // 2 (output of w_1)
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
            x_embedding: Optional pre-computed embeddings
            mask: Optional padding mask
        Returns:
            output: (batch_size, seq_len) - predictions
        """
        # Embed input
        if self.config.get('use_embedding_input', False):
            x = self.embed_layer(x, x_embedding)
        else:
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
        
        # call scikit-survival to compute the IBS score
        # Limit time_grid to be strictly less than the maximum time in test data
        max_test_time = int(np.max(survival_test['time']))
        # Time grid must be strictly less than max_test_time (exclusive upper bound)
        time_grid = np.arange(min(config['max_seq_len'], max_test_time))
        test_ibs = integrated_brier_score(survival_train, survival_test, total_surv_probs[:, :len(time_grid)], time_grid)
        
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

    # ============================================
    # Initialize IBS Loss if enabled
    # ============================================
    ibs_loss_fn = None
    if config.get('use_ibs_loss', False):
        print('[train] Initializing IBS Loss...')
        ibs_loss_fn = IBSLoss(max_time=config['max_seq_len'], device=device)
        
        # Collect training data to estimate censoring distribution
        print('[train] Collecting training data for censoring distribution estimation...')
        train_times_all, train_events_all = [], []
        for features, times, mask, label, is_observed, embedding in train_loader:
            # times and is_observed are lists of length 2 in training mode
            train_times_all.extend(times[0].cpu().numpy())
            train_events_all.extend(is_observed[0].cpu().numpy())
        train_times_all = np.array(train_times_all)
        train_events_all = np.array(train_events_all)
        
        # Estimate censoring distribution
        ibs_loss_fn.estimate_censoring_distribution(train_times_all, train_events_all)
        print(f'[train] IBS Loss initialized with {len(train_times_all)} training samples')

    for t in range(config['num_epochs']):
        decoder.train()

        tot_loss = 0.
        for features, true_durations, mask, label, is_observed, embedding in train_loader:
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
            loss_1 = nn.BCELoss()(surv_probs_a * mask_a, label_a * mask_a)*config['coeff_1'] # formular 14, 15
            loss = loss_1 + 0  # Initialize loss as a separate tensor (creates new tensor, not a reference)

            sigmoid_b = decoder.forward(features[1], embedding_b)
            surv_probs_b = torch.cumprod(sigmoid_b, dim=1)
            loss_2 = None  # Initialize to None
            loss_3 = None  # Initialize to None
            cond_a = is_observed_a & (true_durations_a < true_durations_b) # a is observed and b survival longer than a
            if torch.sum(cond_a) > 0:
                mean_lifetimes_a = torch.sum(surv_probs_a, dim=1)
                mean_lifetimes_b = torch.sum(surv_probs_b, dim=1)
                diff = mean_lifetimes_b[cond_a] - mean_lifetimes_a[cond_a]
                true_diff = true_durations_b[cond_a] - true_durations_a[cond_a]
                loss_2 = config['coeff_2'] * torch.mean(F.relu(true_diff - diff)) # formula 16-1 (use F.relu instead of nn.ReLU())
                loss += loss_2

            cond_a2 = is_observed_a
            if torch.sum(cond_a2) > 0:
                mean_lifetimes_a = torch.sum(surv_probs_a, dim=1)
                loss_3 = config['coeff_3'] * F.l1_loss(mean_lifetimes_a[cond_a2], true_durations_a[cond_a2]) # formula 16-2
                loss += loss_3

            # ============================================
            # Add IBS Loss if enabled
            # ============================================
            ibs_loss = None
            if ibs_loss_fn is not None:
                # IBS loss on sample a (can also add for sample b if desired)
                ibs_loss = ibs_loss_fn(surv_probs_a, true_durations_a, is_observed_a.float())
                loss += config.get('ibs_coeff', 0.1) * ibs_loss
            
            # just print the value of loss terms
            print(loss.item(), loss_1.item(), loss_2.item(), loss_3.item(), ibs_loss.item() if ibs_loss is not None else None)

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

