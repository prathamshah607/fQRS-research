"""
model.py - Complete Multimodal Biomass Prediction Architecture
CSIRO Image2Biomass Competition

Integrated with BiomassDataset format:
- numeric: (8,) [ndvi_norm, height_norm, date_sin, date_cos, date_sin2, date_cos2, ndvi_height_interaction, seasonal_ndvi]
- categorical: (2,) [state_id, species_id]
"""

import torch
import torch.nn as nn
import timm


class TabularEmbedder(nn.Module):
    """
    Embeds tabular features to match DINO dimension (768)
    
    Features (10 total):
    - NDVI: Vegetation index (normalized to [0,1])
    - Height: Pasture height (log-transformed + standardized)
    - Date: Cyclical encoding (4 features: sin/cos × 2 harmonics)
    - NDVI×Height interaction: Captures growth dynamics
    - Seasonal NDVI: NDVI modulated by date
    - State: Australian region (categorical)
    - Species: Pasture species (categorical)
    """
    def __init__(self, num_states, num_species, output_dim=768):
        super().__init__()
        
        # Per-feature embedders (continuous features)
        self.ndvi_embed = nn.Sequential(
            nn.Linear(1, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 64)
        )
        
        self.height_embed = nn.Sequential(
            nn.Linear(1, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 64)
        )
        
        self.date_embed = nn.Sequential(
            nn.Linear(4, 32),  # 4 cyclical features
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 64)
        )
        
        # Interaction embedders
        self.interaction_embed = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 32)
        )
        
        self.seasonal_ndvi_embed = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 32)
        )
        
        # Learned embeddings (categorical features)
        self.state_embedding = nn.Embedding(num_states, 8)
        self.species_embedding = nn.Embedding(num_species, 16)
        
        # Fusion MLP: 64+64+64+32+32+8+16 = 280 → 768
        self.fusion = nn.Sequential(
            nn.Linear(280, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, numeric, categorical):
        """
        Args:
            numeric: (B, 8) [ndvi, height, date_sin, date_cos, date_sin2, date_cos2, interaction, seasonal_ndvi]
            categorical: (B, 2) [state_id, species_id]
        
        Returns:
            Embedded features: (B, 768)
        """
        # Extract individual features from numeric tensor
        ndvi = numeric[:, 0:1]  # (B, 1)
        height = numeric[:, 1:2]  # (B, 1)
        date_feats = numeric[:, 2:6]  # (B, 4) [sin, cos, sin2, cos2]
        interaction = numeric[:, 6:7]  # (B, 1)
        seasonal_ndvi = numeric[:, 7:8]  # (B, 1)
        
        # Embed continuous features
        ndvi_emb = self.ndvi_embed(ndvi)  # (B, 64)
        height_emb = self.height_embed(height)  # (B, 64)
        date_emb = self.date_embed(date_feats)  # (B, 64)
        interaction_emb = self.interaction_embed(interaction)  # (B, 32)
        seasonal_ndvi_emb = self.seasonal_ndvi_embed(seasonal_ndvi)  # (B, 32)
        
        # Extract categorical features
        state_id = categorical[:, 0]  # (B,)
        species_id = categorical[:, 1]  # (B,)
        
        # Embed categorical features
        state_emb = self.state_embedding(state_id)  # (B, 8)
        species_emb = self.species_embedding(species_id)  # (B, 16)
        
        # Concatenate all features
        combined = torch.cat([
            ndvi_emb,           # 64
            height_emb,         # 64
            date_emb,           # 64
            interaction_emb,    # 32
            seasonal_ndvi_emb,  # 32
            state_emb,          # 8
            species_emb,        # 16
        ], dim=-1)  # (B, 280)
        
        # Fusion MLP
        return self.fusion(combined)  # (B, 768)


class DINOBackbone(nn.Module):
    """DINO v2-B/14 backbone with selective fine-tuning"""
    def __init__(self, freeze_backbone_pct=0.7):
        super().__init__()
        
        # Load pretrained DINO v2
        self.dino = timm.create_model(
            'vit_base_patch14_dinov2.lvd142m',
            pretrained=True,
            num_classes=0
        )
        
        # Selective freezing
        total_blocks = len(list(self.dino.blocks))
        freeze_until = int(total_blocks * freeze_backbone_pct)
        
        for i, block in enumerate(self.dino.blocks):
            if i < freeze_until:
                for param in block.parameters():
                    param.requires_grad = False
        
        print(f"[DINO] Frozen {freeze_until}/{total_blocks} blocks ({freeze_backbone_pct*100:.0f}%)")
        
        # Freeze patch embedding and positional encoding
        for param in self.dino.patch_embed.parameters():
            param.requires_grad = False
        if hasattr(self.dino, 'pos_embed'):
            self.dino.pos_embed.requires_grad = False
    
    def forward(self, x):
        """
        Args: x (B, 3, 518, 518)
        Returns: patch_tokens (B, 196, 768)
        """
        features = self.dino.forward_features(x)  # (B, 197, 768)
        return features[:, 1:, :]  # Remove CLS token


class CrossModalFusion(nn.Module):
    """Fuses image patches with tabular embedding using cross-attention"""
    def __init__(self, dim=768, num_heads=12, num_layers=4):
        super().__init__()
        
        self.fusion_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=num_heads,
                batch_first=True,
                dropout=0.1
            )
            for _ in range(num_layers)
        ])
        
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
        
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, 4 * dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(4 * dim, dim)
            )
            for _ in range(num_layers)
        ])
    
    def forward(self, patch_tokens, tabular_embed):
        """
        Args:
            patch_tokens: (B, 196, 768) from DINO
            tabular_embed: (B, 768) from TabularEmbedder
        Returns:
            fused: (B, 768)
        """
        # Expand tabular to sequence format
        tab_seq = tabular_embed.unsqueeze(1)  # (B, 1, 768)
        
        # Apply fusion layers
        for attn, norm, ffn in zip(self.fusion_layers, self.norms, self.ffns):
            attn_out, _ = attn(query=patch_tokens, key=tab_seq, value=tab_seq)
            patch_tokens = norm(patch_tokens + attn_out)
            patch_tokens = norm(patch_tokens + ffn(patch_tokens))
        
        # Global average pooling + residual
        fused = patch_tokens.mean(dim=1) + tabular_embed
        
        return fused


class BiomassPredictionHeads(nn.Module):
    """Multi-task prediction heads with shared bottleneck"""
    def __init__(self, in_dim=768, hidden_dim=512):
        super().__init__()
        
        # Shared bottleneck
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Task-specific heads
        self.heads = nn.ModuleDict({
            'Dry_Green_g': self._make_head(hidden_dim),
            'Dry_Dead_g': self._make_head(hidden_dim),
            'Dry_Clover_g': self._make_head(hidden_dim),
            'GDM_g': self._make_head(hidden_dim),
            'Dry_Total_g': self._make_head(hidden_dim),
        })
    
    def _make_head(self, in_dim):
        return nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Softplus(beta=1.0)  # Ensures output ≥ 0
        )
    
    def forward(self, x):
        """
        Args: x (B, 768)
        Returns: dict of predictions {target_name: (B,)}
        """
        shared = self.shared(x)
        predictions = {
            target: head(shared).squeeze(-1)
            for target, head in self.heads.items()
        }
        return predictions


class MultimodalBiomassModel(nn.Module):
    """Complete multimodal architecture"""
    def __init__(self, num_states, num_species):
        super().__init__()
        
        self.dino_backbone = DINOBackbone(freeze_backbone_pct=0.7)
        self.tabular_embedder = TabularEmbedder(num_states, num_species, output_dim=768)
        self.fusion = CrossModalFusion(dim=768, num_heads=12, num_layers=4)
        self.heads = BiomassPredictionHeads(in_dim=768, hidden_dim=512)
        
        print(f"[Model] Initialized with {num_states} states, {num_species} species")
    
    def forward(self, images, numeric, categorical):
        """
        Args:
            images: (B, 3, 518, 518)
            numeric: (B, 8) tabular features
            categorical: (B, 2) category IDs
        Returns:
            dict of predictions {target_name: (B,)}
        """
        patch_tokens = self.dino_backbone(images)  # (B, 196, 768)
        tabular_embed = self.tabular_embedder(numeric, categorical)  # (B, 768)
        fused = self.fusion(patch_tokens, tabular_embed)  # (B, 768)
        predictions = self.heads(fused)
        
        return predictions


# Unit test
if __name__ == '__main__':
    print("="*60)
    print("Testing MultimodalBiomassModel")
    print("="*60)
    
    model = MultimodalBiomassModel(num_states=4, num_species=14)
    
    # Dummy batch
    batch_size = 4
    images = torch.randn(batch_size, 3, 518, 518)
    numeric = torch.randn(batch_size, 8)  # 8 numeric features
    categorical = torch.randint(0, 4, (batch_size, 2))  # state_id, species_id
    
    print("\nRunning forward pass...")
    predictions = model(images, numeric, categorical)
    
    print("\n✓ Forward pass successful!")
    print(f"  Predictions: {list(predictions.keys())}")
    print(f"  Shape: {predictions['Dry_Total_g'].shape}")
    print(f"  Sample values (Dry_Total_g): {predictions['Dry_Total_g']}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n✓ Model Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    print("\n" + "="*60)
