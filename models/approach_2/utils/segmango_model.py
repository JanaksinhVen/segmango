
import torch
import torch.nn as nn
from mmseg.models.backbones import MixVisionTransformer                 
class SelfAttentionFusion(nn.Module):
    def __init__(self, visual_dim, extra_dim, hidden_dim):
        super().__init__()
        self.visual_dim = visual_dim
        self.extra_dim = extra_dim
        
        # Linear layers to project features into the same dimension for attention
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.extra_proj = nn.Linear(extra_dim, hidden_dim)

        self.alpha = nn.Parameter(torch.tensor(0.8))
        self.beta = nn.Parameter(torch.tensor(0.2))

        # Key, Query, Value projections for self-attention
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection layer
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, visual_feat, extra_feat):
        # Project features to the hidden dimension
        visual_proj = self.visual_proj(visual_feat).unsqueeze(1) # (B, 1, H)
        extra_proj = self.extra_proj(extra_feat).unsqueeze(1)    # (B, 1, H)
        
        # Concatenate and create the input for attention
        combined_features = self.alpha * visual_proj + self.beta * extra_proj

        # Calculate Q, K, V
        queries = self.query(combined_features)  # (B, 2, H)
        keys = self.key(combined_features)      # (B, 2, H)
        values = self.value(combined_features)    # (B, 2, H)
        
        # Scaled Dot-Product Attention
        attention_scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.visual_dim ** 0.5)
        attention_weights = self.softmax(attention_scores)
        
        # Apply attention to values
        attended_values = torch.bmm(attention_weights, values) # (B, 2, H)
        
        # Sum the attended features to get a single fused vector
        fused_features = attended_values.sum(dim=1) # (B, H)
        
        # Final output projection
        fused_output = self.output_proj(fused_features)
        
        return fused_output

class SegFormerRegressor(nn.Module):
    """
    SegFormer-B0 or B1 encoder + fully-connected regression head.
    """

    def __init__(self,
                 encoder_ckpt: str,
                 variant: str = "b1",                  # 'b0' or 'b1'
                 num_extra_feats: int = 0,
                 hidden_dim: int = 256,
                 
                 freeze_encoder: bool = False,
                 freeze_regressor: bool = False
                 ):
                 
        super().__init__()

        assert variant in ['b0', 'b1'], "Only 'b0' and 'b1' variants are supported"

        # ------------ Encoder configuration ------------ #
        if variant == 'b0':
            embed_dims = 32
            num_layers = [2, 2, 2, 2]
            num_heads = [1, 2, 5, 8]
            encoder_dim = 256  # Output of stage 4
        else:  # 'b1'
            embed_dims = 64
            num_layers = [2, 2, 2, 2]
            num_heads = [1, 2, 5, 8]
            encoder_dim = 512  # Output of stage 4

        self.encoder = MixVisionTransformer(
            in_channels=3,
            embed_dims=embed_dims,
            num_stages=4,
            num_layers=num_layers,
            num_heads=num_heads,
            patch_sizes=[7, 3, 3, 3],
            sr_ratios=[8, 4, 2, 1],
            mlp_ratio=4,
            qkv_bias=True,
            out_indices=(3,),
            norm_cfg=dict(type='LN', eps=1e-6)
        )

        print(f"[INFO] Using SegFormer-{variant.upper()} encoder with output dim {encoder_dim}")
        print(f"[INFO] Loading encoder weights from: {encoder_ckpt}")
        if encoder_ckpt is not None:
            ckpt = torch.load(encoder_ckpt, map_location='cpu')
            if 'state_dict' in ckpt:
                ckpt = ckpt['state_dict']

            # Remove 'backbone.' prefix from checkpoint keys
            encoder_state = {k.replace('backbone.', ''): v for k, v in ckpt.items() if k.startswith('backbone.')}
            missing, unexpected = self.encoder.load_state_dict(encoder_state, strict=False)
            print(f"[INFO] Loaded encoder with missing keys: {missing}")
            print(f"[INFO] Unexpected keys: {unexpected}")
        else: 
            print("[WARNING] No encoder checkpoint provided. Initializing with random weights.")

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # ------------ Regression head ------------ #
        self.pool = nn.AdaptiveAvgPool2d(1)
        # in_dim = encoder_dim + num_extra_feats
        self.fusion_layer = SelfAttentionFusion(
            visual_dim=encoder_dim,
            extra_dim=num_extra_feats,
            hidden_dim=hidden_dim
        )
        self.regressor = nn.Sequential(
            # nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )
        if freeze_regressor:
            for p in self.regressor.parameters():
                p.requires_grad = False
            for p in self.fusion_layer.parameters():
                p.requires_grad = False


    def forward(self, img, extra_feats=None):
        feat_map = self.encoder(img)[0]                # shape: (B, encoder_dim, H/32, W/32)
        feat_vec = self.pool(feat_map).flatten(1)      # shape: (B, encoder_dim)

        if extra_feats is not None:
            fused_feats = self.fusion_layer(feat_vec, extra_feats)
        else:
            fused_feats = feat_vec
            
        return self.regressor(fused_feats)



    def unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = True
        print("[INFO] Encoder has been unfrozen.")

    def unfreeze_regressor(self):
        for p in self.regressor.parameters():
            p.requires_grad = True
        for p in self.fusion_layer.parameters():
            p.requires_grad = True
        print('[INFO] Regressor has been unfrozen')



class MultiImageSegFormerRegressor(nn.Module):
    def __init__(self, base_model: nn.Module, num_inputs: int = 8, hidden_dim: int = 32):
        """
        base_model: The SegFormerRegressor model.
        num_inputs: Number of images per sample (default 8).
        hidden_dim: Hidden layer dimension in the second-stage MLP.
        """
        super().__init__()
        self.base_model = base_model
        self.num_inputs = num_inputs

        # Final regressor over 8 outputs
        self.final_regressor = nn.Sequential(
            nn.Linear(num_inputs, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, imgs, extra_feats=None):
        """
        imgs        : Tensor[B, 8, 3, H, W]
        extra_feats : Tensor[B, 8, F] or None
        """
        B, N, C, H, W = imgs.shape
        assert N == self.num_inputs, f"Expected {self.num_inputs} images, got {N}"

        imgs = imgs.view(B * N, C, H, W)  # (B*N, 3, H, W)
        if extra_feats is not None:
            extra_feats = extra_feats.view(B * N, -1).float()   # (B*N, F)
        # print('extra_feats:',extra_feats.shape)
        preds = self.base_model(imgs, extra_feats)     # (B*N, 1)
        preds = preds.view(B, N)                       # (B, 8)

        return self.final_regressor(preds)             # (B, 1)
