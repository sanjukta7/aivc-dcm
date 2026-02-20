import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

Tensor = torch.Tensor


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal positional/time embedding."""

    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: Tensor) -> Tensor:
        """
        Args:
            t: Tensor of shape [batch_size] with values in [0, 1]

        Returns:
            Embeddings of shape [batch_size, dim]
        """
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half_dim, device=t.device, dtype=t.dtype)
            / half_dim
        )
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, seq_len: int, device: torch.device) -> Tensor:
        """Get rotary embeddings for sequence length."""
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb


def rotate_half(x: Tensor) -> Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(x: Tensor, freqs: Tensor) -> Tensor:
    cos = freqs.cos().unsqueeze(0).unsqueeze(0)  # [1, 1, seq, dim]
    sin = freqs.sin().unsqueeze(0).unsqueeze(0)
    return (x * cos) + (rotate_half(x) * sin)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with optional rotary embeddings."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        use_rotary: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.use_rotary = use_rotary
        self.dropout_p = dropout  # Store dropout rate for Flash Attention

        assert hidden_dim % num_heads == 0

        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        if use_rotary:
            self.rotary = RotaryEmbedding(self.head_dim)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x: Input tensor [batch, seq_len, hidden_dim]
            mask: Optional attention mask [batch, seq_len]

        Returns:
            Output tensor [batch, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply rotary embeddings
        if self.use_rotary:
            freqs = self.rotary(seq_len, x.device)
            q = apply_rotary_emb(q, freqs)
            k = apply_rotary_emb(k, freqs)

        # ========== FLASH ATTENTION - NO OOM! ==========
        # Convert mask format if needed
        # Flash Attention expects: [batch, num_heads, seq_len, seq_len] or None
        attn_mask = None
        if mask is not None:
            # mask: [batch, seq_len] -> [batch, 1, 1, seq_len]
            attn_mask = mask.unsqueeze(1).unsqueeze(2)
            # Expand to [batch, 1, seq_len, seq_len] for compatibility
            attn_mask = attn_mask.expand(batch_size, 1, seq_len, seq_len)
            # Convert boolean mask to additive mask (0 for attend, -inf for mask)
            attn_mask = torch.where(attn_mask == 0, float('-inf'), 0.0)

        # Use PyTorch's memory-efficient Flash Attention
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False
        )
        # ========== END FLASH ATTENTION ==========

        # Reshape output
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        out = self.out_proj(out)

        return out


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, hidden_dim: int, ff_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class AdaptiveLayerNorm(nn.Module):
    """Adaptive Layer Normalization conditioned on time embedding.

    Applies: y = gamma(t) * LayerNorm(x) + beta(t)
    """

    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.gamma_beta = nn.Linear(cond_dim, 2 * hidden_dim)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """
        Args:
            x: Input [batch, seq_len, hidden_dim]
            cond: Conditioning [batch, cond_dim]

        Returns:
            Normalized output [batch, seq_len, hidden_dim]
        """
        gamma_beta = self.gamma_beta(cond).unsqueeze(1)  # [batch, 1, 2*hidden_dim]
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return gamma * self.norm(x) + beta


class TransformerBlock(nn.Module):
    """Transformer block with adaptive layer normalization."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ff_dim: int,
        cond_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attn_norm = AdaptiveLayerNorm(hidden_dim, cond_dim)
        self.attn = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.ff_norm = AdaptiveLayerNorm(hidden_dim, cond_dim)
        self.ff = FeedForward(hidden_dim, ff_dim, dropout)

    def forward(self, x: Tensor, cond: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x: Input [batch, seq_len, hidden_dim]
            cond: Time conditioning [batch, cond_dim]
            mask: Optional attention mask

        Returns:
            Output [batch, seq_len, hidden_dim]
        """
        # Self-attention with residual
        x = x + self.attn(self.attn_norm(x, cond), mask)
        # Feed-forward with residual
        x = x + self.ff(self.ff_norm(x, cond))
        return x


class SEDDTransformer(nn.Module):

    def __init__(
        self,
        num_genes: int,
        num_bins: int,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        ff_mult: float = 4.0,
        dropout: float = 0.1,
        max_seq_len: int = 4096,
    ):

        super().__init__()
        self.num_genes = num_genes
        self.num_bins = num_bins
        self.vocab_size = num_bins + 1  # +1 for mask token
        self.hidden_dim = hidden_dim
        self.mask_index = num_bins

        ff_dim = int(hidden_dim * ff_mult)

        self.token_embed = nn.Embedding(self.vocab_size, hidden_dim)

        self.gene_embed = nn.Embedding(max_seq_len, hidden_dim)

        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, ff_dim, hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        self.out_norm = AdaptiveLayerNorm(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, self.vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        x: Tensor,
        sigma: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:

        batch_size, seq_len = x.shape
        device = x.device

        if sigma.dim() == 0:
            sigma = sigma.expand(batch_size)

        tok_emb = self.token_embed(x)
        pos_idx = torch.arange(seq_len, device=device)
        pos_emb = self.gene_embed(pos_idx).unsqueeze(0)
        h = tok_emb + pos_emb

        t_emb = self.time_embed(sigma)

        for block in self.blocks:
            h = block(h, t_emb, mask)

        h = self.out_norm(h, t_emb)
        logits = self.out_proj(h)

        return logits

    def score(
        self,
        x: Tensor,
        sigma: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:

        logits = self.forward(x, sigma, mask)

        score = F.log_softmax(logits, dim=-1)

        return score

    def get_loss(
        self,
        x_clean: Tensor,
        x_noised: Tensor,
        sigma: Tensor,
        graph,
        mask: Optional[Tensor] = None,
    ) -> Tensor:

        pred_score = self.score(x_noised, sigma, mask)

        is_masked = (x_noised == self.mask_index)

        if not is_masked.any():
            return torch.tensor(0.0, device=x_clean.device)

        pred_at_mask = pred_score[is_masked]  # [num_masked, vocab_size]
        target_at_mask = x_clean[is_masked]   # [num_masked]

        loss = F.cross_entropy(pred_at_mask, target_at_mask)

        return loss


class SEDDTransformerSmall(SEDDTransformer):

    def __init__(self, num_genes: int, num_bins: int, **kwargs):
        defaults = dict(
            hidden_dim=128,
            num_layers=4,
            num_heads=4,
            ff_mult=4.0,
            dropout=0.1,
        )
        defaults.update(kwargs)
        super().__init__(num_genes, num_bins, **defaults)


class SEDDTransformerMedium(SEDDTransformer):

    def __init__(self, num_genes: int, num_bins: int, **kwargs):
        defaults = dict(
            hidden_dim=256,
            num_layers=6,
            num_heads=8,
            ff_mult=4.0,
            dropout=0.1,
        )
        defaults.update(kwargs)
        super().__init__(num_genes, num_bins, **defaults)


class SEDDTransformerLarge(SEDDTransformer):

    def __init__(self, num_genes: int, num_bins: int, **kwargs):
        defaults = dict(
            hidden_dim=512,
            num_layers=8,
            num_heads=8,
            ff_mult=4.0,
            dropout=0.1,
        )
        defaults.update(kwargs)
        super().__init__(num_genes, num_bins, **defaults)


class SEDDPerturbationTransformer(SEDDTransformer):
    """SEDD Transformer with perturbation conditioning for control -> perturbed prediction.

    This model extends SEDDTransformer to condition on perturbation labels,
    enabling prediction of perturbed cell states from control cells.

    Architecture:
        - Inherits token, gene, and time embeddings from SEDDTransformer
        - Adds perturbation embedding for conditioning
        - Combines time and perturbation embeddings for adaptive layer norm

    Training:
        - Input: control cell expression + perturbation label
        - Target: perturbed cell expression
        - Uses discrete diffusion with masking
    """

    def __init__(
        self,
        num_genes: int,
        num_bins: int,
        num_perturbations: int,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        ff_mult: float = 4.0,
        dropout: float = 0.1,
        max_seq_len: int = 4096,
        precomputed_emb_dim: int = None,
        num_cell_types: int = None,
    ):
        """
        Args:
            num_genes: Number of genes (sequence length)
            num_bins: Number of expression bins (vocabulary size, excluding mask)
            num_perturbations: Number of unique perturbation conditions
            hidden_dim: Transformer hidden dimension
            num_layers: Number of transformer blocks
            num_heads: Number of attention heads
            ff_mult: Feed-forward expansion factor
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
            precomputed_emb_dim: Dimension of pre-computed embeddings (e.g., 320 for ESM2), if using
            num_cell_types: Number of unique cell types (optional, for dual conditioning)
        """
        super().__init__(
            num_genes=num_genes,
            num_bins=num_bins,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_mult=ff_mult,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )

        self.num_perturbations = num_perturbations
        self.precomputed_emb_dim = precomputed_emb_dim
        self.num_cell_types = num_cell_types

        # Perturbation embedding
        self.pert_embed = nn.Embedding(num_perturbations, hidden_dim)

        # Pre-computed embedding projection (e.g., for protein embeddings)
        # Projects from precomputed_emb_dim (e.g., 320) to hidden_dim
        if precomputed_emb_dim is not None:
            self.precomputed_proj = nn.Linear(precomputed_emb_dim, hidden_dim)
        else:
            self.precomputed_proj = None

        # Cell type embedding (optional for dual conditioning)
        if num_cell_types is not None and num_cell_types > 0:
            self.cell_type_embed = nn.Embedding(num_cell_types, hidden_dim)
            self.cell_type_proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Linear(hidden_dim * 4, hidden_dim),
            )
        else:
            self.cell_type_embed = None
            self.cell_type_proj = None

        # Update time embedding to also incorporate perturbation
        # New conditioning will be: time_emb + pert_emb (+ cell_type_emb if available)
        self.pert_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        self._init_weights()

    def forward(
        self,
        x: Tensor,
        sigma: Tensor,
        pert_labels: Tensor,
        mask: Optional[Tensor] = None,
        cell_type_labels: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x: Input tokens [batch, seq_len]
            sigma: Diffusion time [batch] or scalar
            pert_labels: Perturbation labels [batch]
            mask: Optional attention mask [batch, seq_len]
            cell_type_labels: Optional cell type labels [batch]

        Returns:
            Logits [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = x.shape
        device = x.device

        if sigma.dim() == 0:
            sigma = sigma.expand(batch_size)

        # Token and position embeddings
        tok_emb = self.token_embed(x)
        pos_idx = torch.arange(seq_len, device=device)
        pos_emb = self.gene_embed(pos_idx).unsqueeze(0)
        h = tok_emb + pos_emb

        # Time embedding
        t_emb = self.time_embed(sigma)

        # Perturbation embedding
        # Check if pert_labels are already embeddings (2D) or indices (1D)
        if pert_labels.dim() == 1:
            # pert_labels are indices, use embedding layer
            p_emb = self.pert_embed(pert_labels.long())
        else:
            # pert_labels are already embeddings (from cond_label_lookup)
            # Project from precomputed dimension to hidden_dim if needed
            if self.precomputed_proj is not None:
                p_emb = self.precomputed_proj(pert_labels)
            else:
                # Fallback: assume embeddings are already correct dimension
                p_emb = pert_labels
        p_emb = self.pert_proj(p_emb)

        # Combined conditioning: time + perturbation (+ cell_type if available)
        cond = t_emb + p_emb

        # Add cell type conditioning if available
        if cell_type_labels is not None and self.cell_type_embed is not None:
            ct_emb = self.cell_type_embed(cell_type_labels.long())
            ct_emb = self.cell_type_proj(ct_emb)
            cond = cond + ct_emb

        # Transformer blocks
        for block in self.blocks:
            h = block(h, cond, mask)

        # Output
        h = self.out_norm(h, cond)
        logits = self.out_proj(h)

        return logits

    def score(
        self,
        x: Tensor,
        sigma: Tensor,
        pert_labels: Tensor,
        mask: Optional[Tensor] = None,
        cell_type_labels: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute log probabilities (scores) for each token."""
        logits = self.forward(x, sigma, pert_labels, mask, cell_type_labels)
        score = F.log_softmax(logits, dim=-1)
        return score

    def get_loss(
        self,
        x_perturbed: Tensor,
        x_noised: Tensor,
        sigma: Tensor,
        pert_labels: Tensor,
        graph,
        mask: Optional[Tensor] = None,
        cell_type_labels: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute perturbation prediction loss.

        Args:
            x_control: Control cell expression [batch, seq_len]
            x_perturbed: True perturbed expression (target) [batch, seq_len]
            x_noised: Noised perturbed expression (input) [batch, seq_len]
            sigma: Diffusion time [batch]
            pert_labels: Perturbation labels [batch]
            graph: Diffusion graph (for mask index)
            mask: Optional attention mask
            cell_type_labels: Optional cell type labels [batch]

        Returns:
            Cross-entropy loss at masked positions
        """
        # Predict perturbed expression from noised perturbed + perturbation label + cell type
        pred_score = self.score(x_noised, sigma, pert_labels, mask, cell_type_labels)

        # Only compute loss at masked positions
        is_masked = (x_noised == self.mask_index)

        if not is_masked.any():
            return torch.tensor(0.0, device=x_perturbed.device, requires_grad = True)

        pred_at_mask = pred_score[is_masked]  # [num_masked, vocab_size]
        target_at_mask = x_perturbed[is_masked]  # [num_masked]

        loss = F.cross_entropy(pred_at_mask, target_at_mask)

        return loss


class SEDDPerturbationTransformerSmall(SEDDPerturbationTransformer):
    """Small version of perturbation transformer."""

    def __init__(self, num_genes: int, num_bins: int, num_perturbations: int, **kwargs):
        defaults = dict(
            hidden_dim=128,
            num_layers=4,
            num_heads=4,
            ff_mult=4.0,
            dropout=0.1,
        )
        defaults.update(kwargs)
        super().__init__(num_genes, num_bins, num_perturbations, **defaults)


class SEDDPerturbationTransformerMedium(SEDDPerturbationTransformer):
    """Medium version of perturbation transformer."""

    def __init__(self, num_genes: int, num_bins: int, num_perturbations: int, **kwargs):
        defaults = dict(
            hidden_dim=256,
            num_layers=6,
            num_heads=8,
            ff_mult=4.0,
            dropout=0.1,
        )
        defaults.update(kwargs)
        super().__init__(num_genes, num_bins, num_perturbations, **defaults)


class SEDDPerturbationTransformerLarge(SEDDPerturbationTransformer):
    """Large version of perturbation transformer."""

    def __init__(self, num_genes: int, num_bins: int, num_perturbations: int, **kwargs):
        defaults = dict(
            hidden_dim=512,
            num_layers=8,
            num_heads=8,
            ff_mult=4.0,
            dropout=0.1,
        )
        defaults.update(kwargs)
        super().__init__(num_genes, num_bins, num_perturbations, **defaults)


# =============================================================================
# Separate FiLM Conditioning Classes
# =============================================================================
# These classes implement sequential FiLM (Feature-wise Linear Modulation)
# where each conditioning signal (time, perturbation, cell_type) applies
# its own gamma/beta transformation sequentially, rather than summing
# embeddings before a single FiLM layer.


class SeparateFiLMAdaptiveLayerNorm(nn.Module):
    """Adaptive Layer Normalization with separate FiLM for each condition type.

    Instead of combining all conditions into one embedding and applying a single FiLM,
    this applies FiLM transformations sequentially:
        x_norm = LayerNorm(x)
        x1 = gamma_time * x_norm + beta_time
        x2 = gamma_pert * x1 + beta_pert
        x3 = gamma_celltype * x2 + beta_celltype  (if cell type is provided)

    This allows each conditioning signal to modulate the representation independently,
    enabling better compositional generalization for unseen combinations.
    """

    def __init__(self, hidden_dim: int, has_cell_type: bool = False):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.has_cell_type = has_cell_type

        # Separate FiLM projections for each condition type
        self.time_gamma_beta = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.pert_gamma_beta = nn.Linear(hidden_dim, 2 * hidden_dim)

        if has_cell_type:
            self.cell_type_gamma_beta = nn.Linear(hidden_dim, 2 * hidden_dim)

    def forward(
        self,
        x: Tensor,
        time_cond: Tensor,
        pert_cond: Tensor,
        cell_type_cond: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x: Input [batch, seq_len, hidden_dim]
            time_cond: Time conditioning [batch, hidden_dim]
            pert_cond: Perturbation conditioning [batch, hidden_dim]
            cell_type_cond: Optional cell type conditioning [batch, hidden_dim]

        Returns:
            Modulated output [batch, seq_len, hidden_dim]
        """
        # Layer norm first
        h = self.norm(x)

        # Apply time FiLM
        time_gb = self.time_gamma_beta(time_cond).unsqueeze(1)  # [batch, 1, 2*hidden_dim]
        time_gamma, time_beta = time_gb.chunk(2, dim=-1)
        h = time_gamma * h + time_beta

        # Apply perturbation FiLM
        pert_gb = self.pert_gamma_beta(pert_cond).unsqueeze(1)
        pert_gamma, pert_beta = pert_gb.chunk(2, dim=-1)
        h = pert_gamma * h + pert_beta

        # Apply cell type FiLM (if available)
        if cell_type_cond is not None and self.has_cell_type:
            ct_gb = self.cell_type_gamma_beta(cell_type_cond).unsqueeze(1)
            ct_gamma, ct_beta = ct_gb.chunk(2, dim=-1)
            h = ct_gamma * h + ct_beta

        return h


class SeparateFiLMTransformerBlock(nn.Module):
    """Transformer block with separate FiLM conditioning for each signal type."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.0,
        has_cell_type: bool = False,
    ):
        super().__init__()
        self.attn_norm = SeparateFiLMAdaptiveLayerNorm(hidden_dim, has_cell_type)
        self.attn = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.ff_norm = SeparateFiLMAdaptiveLayerNorm(hidden_dim, has_cell_type)
        self.ff = FeedForward(hidden_dim, ff_dim, dropout)

    def forward(
        self,
        x: Tensor,
        time_cond: Tensor,
        pert_cond: Tensor,
        cell_type_cond: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x: Input [batch, seq_len, hidden_dim]
            time_cond: Time conditioning [batch, hidden_dim]
            pert_cond: Perturbation conditioning [batch, hidden_dim]
            cell_type_cond: Optional cell type conditioning [batch, hidden_dim]
            mask: Optional attention mask

        Returns:
            Output [batch, seq_len, hidden_dim]
        """
        # Self-attention with residual
        x = x + self.attn(self.attn_norm(x, time_cond, pert_cond, cell_type_cond), mask)
        # Feed-forward with residual
        x = x + self.ff(self.ff_norm(x, time_cond, pert_cond, cell_type_cond))
        return x


class SEDDPerturbationTransformerSeparateFiLM(nn.Module):
    """SEDD Transformer with separate FiLM conditioning for each label type.

    Unlike SEDDPerturbationTransformer which sums time + perturbation + cell_type
    embeddings before applying a single FiLM layer, this model applies separate
    FiLM transformations for each conditioning signal sequentially.

    This enables:
        - Better compositional generalization (unseen label combinations)
        - Factorized representations (cell type and perturbation effects are separable)
        - More interpretable conditioning (each signal has its own modulation pathway)

    Architecture:
        - Each transformer block has separate gamma/beta projections for time, pert, cell_type
        - FiLM is applied sequentially: time -> perturbation -> cell_type
        - Each conditioning signal modulates the previous output
    """

    def __init__(
        self,
        num_genes: int,
        num_bins: int,
        num_perturbations: int,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        ff_mult: float = 4.0,
        dropout: float = 0.1,
        max_seq_len: int = 4096,
        precomputed_emb_dim: int = None,
        num_cell_types: int = None,
    ):
        """
        Args:
            num_genes: Number of genes (sequence length)
            num_bins: Number of expression bins (vocabulary size, excluding mask)
            num_perturbations: Number of unique perturbation conditions
            hidden_dim: Transformer hidden dimension
            num_layers: Number of transformer blocks
            num_heads: Number of attention heads
            ff_mult: Feed-forward expansion factor
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
            precomputed_emb_dim: Dimension of pre-computed embeddings (e.g., 320 for ESM2)
            num_cell_types: Number of unique cell types (optional, for dual conditioning)
        """
        super().__init__()
        self.num_genes = num_genes
        self.num_bins = num_bins
        self.vocab_size = num_bins + 1  # +1 for mask token
        self.hidden_dim = hidden_dim
        self.mask_index = num_bins
        self.num_perturbations = num_perturbations
        self.precomputed_emb_dim = precomputed_emb_dim
        self.num_cell_types = num_cell_types

        ff_dim = int(hidden_dim * ff_mult)
        has_cell_type = num_cell_types is not None and num_cell_types > 0

        # Token and position embeddings
        self.token_embed = nn.Embedding(self.vocab_size, hidden_dim)
        self.gene_embed = nn.Embedding(max_seq_len, hidden_dim)

        # Time embedding (projects to hidden_dim for FiLM)
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        # Perturbation embedding
        self.pert_embed = nn.Embedding(num_perturbations, hidden_dim)

        # Pre-computed embedding projection (e.g., for protein embeddings)
        if precomputed_emb_dim is not None:
            self.precomputed_proj = nn.Linear(precomputed_emb_dim, hidden_dim)
        else:
            self.precomputed_proj = None

        # Perturbation projection MLP (projects to hidden_dim for FiLM)
        self.pert_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        # Cell type embedding and projection (optional)
        if has_cell_type:
            self.cell_type_embed = nn.Embedding(num_cell_types, hidden_dim)
            self.cell_type_proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Linear(hidden_dim * 4, hidden_dim),
            )
        else:
            self.cell_type_embed = None
            self.cell_type_proj = None

        # Transformer blocks with separate FiLM
        self.blocks = nn.ModuleList([
            SeparateFiLMTransformerBlock(
                hidden_dim, num_heads, ff_dim, dropout, has_cell_type
            )
            for _ in range(num_layers)
        ])

        # Output normalization and projection (also uses separate FiLM)
        self.out_norm = SeparateFiLMAdaptiveLayerNorm(hidden_dim, has_cell_type)
        self.out_proj = nn.Linear(hidden_dim, self.vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        x: Tensor,
        sigma: Tensor,
        pert_labels: Tensor,
        mask: Optional[Tensor] = None,
        cell_type_labels: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x: Input tokens [batch, seq_len]
            sigma: Diffusion time [batch] or scalar
            pert_labels: Perturbation labels [batch] or embeddings [batch, emb_dim]
            mask: Optional attention mask [batch, seq_len]
            cell_type_labels: Optional cell type labels [batch]

        Returns:
            Logits [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = x.shape
        device = x.device

        if sigma.dim() == 0:
            sigma = sigma.expand(batch_size)

        # Token and position embeddings
        tok_emb = self.token_embed(x)
        pos_idx = torch.arange(seq_len, device=device)
        pos_emb = self.gene_embed(pos_idx).unsqueeze(0)
        h = tok_emb + pos_emb

        # Time embedding (separate conditioning signal)
        time_cond = self.time_embed(sigma)

        # Perturbation embedding (separate conditioning signal)
        if pert_labels.dim() == 1:
            # pert_labels are indices, use embedding layer
            p_emb = self.pert_embed(pert_labels.long())
        else:
            # pert_labels are already embeddings (from cond_label_lookup)
            if self.precomputed_proj is not None:
                p_emb = self.precomputed_proj(pert_labels)
            else:
                p_emb = pert_labels
        pert_cond = self.pert_proj(p_emb)

        # Cell type embedding (separate conditioning signal, optional)
        cell_type_cond = None
        if cell_type_labels is not None and self.cell_type_embed is not None:
            ct_emb = self.cell_type_embed(cell_type_labels.long())
            cell_type_cond = self.cell_type_proj(ct_emb)

        # Transformer blocks with separate FiLM conditioning
        for block in self.blocks:
            h = block(h, time_cond, pert_cond, cell_type_cond, mask)

        # Output with separate FiLM
        h = self.out_norm(h, time_cond, pert_cond, cell_type_cond)
        logits = self.out_proj(h)

        return logits

    def score(
        self,
        x: Tensor,
        sigma: Tensor,
        pert_labels: Tensor,
        mask: Optional[Tensor] = None,
        cell_type_labels: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute log probabilities (scores) for each token."""
        logits = self.forward(x, sigma, pert_labels, mask, cell_type_labels)
        score = F.log_softmax(logits, dim=-1)
        return score

    def get_loss(
        self,
        x_perturbed: Tensor,
        x_noised: Tensor,
        sigma: Tensor,
        pert_labels: Tensor,
        graph,
        mask: Optional[Tensor] = None,
        cell_type_labels: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute perturbation prediction loss.

        Args:
            x_perturbed: True perturbed expression (target) [batch, seq_len]
            x_noised: Noised perturbed expression (input) [batch, seq_len]
            sigma: Diffusion time [batch]
            pert_labels: Perturbation labels [batch]
            graph: Diffusion graph (for mask index)
            mask: Optional attention mask
            cell_type_labels: Optional cell type labels [batch]

        Returns:
            Cross-entropy loss at masked positions
        """
        pred_score = self.score(x_noised, sigma, pert_labels, mask, cell_type_labels)

        # Only compute loss at masked positions
        is_masked = (x_noised == self.mask_index)

        if not is_masked.any():
            return torch.tensor(0.0, device=x_perturbed.device, requires_grad=True)

        pred_at_mask = pred_score[is_masked]  # [num_masked, vocab_size]
        target_at_mask = x_perturbed[is_masked]  # [num_masked]

        loss = F.cross_entropy(pred_at_mask, target_at_mask)

        return loss


class SEDDPerturbationTransformerSeparateFiLMSmall(SEDDPerturbationTransformerSeparateFiLM):
    """Small version of separate FiLM perturbation transformer."""

    def __init__(self, num_genes: int, num_bins: int, num_perturbations: int, **kwargs):
        defaults = dict(
            hidden_dim=128,
            num_layers=4,
            num_heads=4,
            ff_mult=4.0,
            dropout=0.1,
        )
        defaults.update(kwargs)
        super().__init__(num_genes, num_bins, num_perturbations, **defaults)


class SEDDPerturbationTransformerSeparateFiLMMedium(SEDDPerturbationTransformerSeparateFiLM):
    """Medium version of separate FiLM perturbation transformer."""

    def __init__(self, num_genes: int, num_bins: int, num_perturbations: int, **kwargs):
        defaults = dict(
            hidden_dim=256,
            num_layers=6,
            num_heads=8,
            ff_mult=4.0,
            dropout=0.1,
        )
        defaults.update(kwargs)
        super().__init__(num_genes, num_bins, num_perturbations, **defaults)


class SEDDPerturbationTransformerSeparateFiLMLarge(SEDDPerturbationTransformerSeparateFiLM):
    """Large version of separate FiLM perturbation transformer."""

    def __init__(self, num_genes: int, num_bins: int, num_perturbations: int, **kwargs):
        defaults = dict(
            hidden_dim=512,
            num_layers=8,
            num_heads=8,
            ff_mult=4.0,
            dropout=0.1,
        )
        defaults.update(kwargs)
        super().__init__(num_genes, num_bins, num_perturbations, **defaults) 