"""
Model architectures for SegFormer-based yield prediction.
"""

import torch
import torch.nn as nn
from mmseg.models.backbones import MixVisionTransformer


class SelfAttentionFusion(nn.Module):
    """
    Self-attention module to fuse visual and tabular features.
    
    Args:
        visual_dim: Dimension of visual features from encoder
        extra_dim: Dimension of tabular/extra features
        hidden_dim: Hidden dimension for fusion
    """
    
    def __init__(self, visual_dim, extra_dim, hidden_dim):
        super().__init__()
        self.visual_dim = visual_dim
        self.extra_dim = extra_dim
        
        # Project features to hidden dimension
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.extra_proj = nn.Linear(extra_dim, hidden_dim)

        # Learnable weights for feature combination
        self.alpha = nn.Parameter(torch.tensor(0.8))
        self.beta = nn.Parameter(torch.tensor(0.2))

        # Attention components
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, visual_feat, extra_feat):
        """
        Fuse visual and tabular features using self-attention.
        
        Args:
            visual_feat: Tensor of shape (B, visual_dim)
            extra_feat: Tensor of shape (B, extra_dim)
            
        Returns:
            Fused features of shape (B, hidden_dim)
        """
        # Project features
        visual_proj = self.visual_proj(visual_feat).unsqueeze(1)  # (B, 1, H)
        extra_proj = self.extra_proj(extra_feat).unsqueeze(1)      # (B, 1, H)
        
        # Weighted combination
        combined_features = self.alpha * visual_proj + self.beta * extra_proj

        # Compute attention
        queries = self.query(combined_features)
        keys = self.key(combined_features)
        values = self.value(combined_features)
        
        attention_scores = torch.bmm(
            queries, keys.transpose(1, 2)
        ) / (self.visual_dim ** 0.5)
        attention_weights = self.softmax(attention_scores)
        
        # Apply attention to values
        attended_values = torch.bmm(attention_weights, values)
        fused_features = attended_values.sum(dim=1)
        
        # Output projection
        fused_output = self.output_proj(fused_features)
        
        return fused_output


class SegFormerRegressor(nn.Module):
    """
    SegFormer encoder + self-attention fusion + MLP regressor.
    
    Supports variants: 'b0' and 'b1' with different encoder configurations.
    
    Args:
        encoder_ckpt: Path to pretrained encoder checkpoint
        variant: Model variant ('b0' or 'b1')
        num_extra_feats: Number of tabular features
        hidden_dim: Hidden dimension for regression head
        freeze_encoder: If True, keep encoder frozen during early training
    """

    def __init__(self, encoder_ckpt, variant="b1", num_extra_feats=0,
                 hidden_dim=256, freeze_encoder=False):
        super().__init__()

        assert variant in ['b0', 'b1'], "Only 'b0' and 'b1' variants are supported"

        # ============= Encoder Configuration =============
        if variant == 'b0':
            embed_dims = 32
            encoder_dim = 256
        else:  # 'b1'
            embed_dims = 64
            encoder_dim = 512

        # Build encoder
        self.encoder = MixVisionTransformer(
            in_channels=3,
            embed_dims=embed_dims,
            num_stages=4,
            num_layers=[2, 2, 2, 2],
            num_heads=[1, 2, 5, 8],
            patch_sizes=[7, 3, 3, 3],
            sr_ratios=[8, 4, 2, 1],
            mlp_ratio=4,
            qkv_bias=True,
            out_indices=(3,),
            norm_cfg=dict(type='LN', eps=1e-6)
        )

        # Load pretrained weights
        print(f"[INFO] Using SegFormer-{variant.upper()} encoder with output dim {encoder_dim}")
        print(f"[INFO] Loading encoder weights from: {encoder_ckpt}")
        ckpt = torch.load(encoder_ckpt, map_location='cpu')
        
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']

        # Remove 'backbone.' prefix
        encoder_state = {
            k.replace('backbone.', ''): v for k, v in ckpt.items() 
            if k.startswith('backbone.')
        }
        missing, unexpected = self.encoder.load_state_dict(encoder_state, strict=False)
        print(f"[INFO] Loaded encoder with missing keys: {missing}")
        print(f"[INFO] Unexpected keys: {unexpected}")

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            print("[INFO] Encoder is frozen.")

        # ============= Fusion and Regression Head =============
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        self.fusion_layer = SelfAttentionFusion(
            visual_dim=encoder_dim,
            extra_dim=num_extra_feats,
            hidden_dim=hidden_dim
        ) if num_extra_feats > 0 else None
        
        self.regressor = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, img, extra_feats=None):
        """
        Args:
            img: Image tensor of shape (B, 3, H, W)
            extra_feats: Tabular features tensor of shape (B, F) or None
            
        Returns:
            Predictions of shape (B, 1)
        """
        feat_map = self.encoder(img)[0]          # (B, encoder_dim, H/32, W/32)
        feat_vec = self.pool(feat_map).flatten(1)  # (B, encoder_dim)

        if self.fusion_layer is not None and extra_feats is not None:
            fused_feats = self.fusion_layer(feat_vec, extra_feats)
        else:
            fused_feats = feat_vec
            
        return self.regressor(fused_feats)

    def unfreeze_encoder(self):
        """Unfreeze encoder parameters for fine-tuning."""
        for p in self.encoder.parameters():
            p.requires_grad = True
        print("[INFO] Encoder has been unfrozen.")
