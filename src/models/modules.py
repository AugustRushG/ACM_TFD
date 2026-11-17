import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=True, norm_last_layer=True, 
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        """
        in_dim: Input feature dimension (from backbone)
        out_dim: Number of prototypes/clusters (e.g., 65536 in DINO)
        bottleneck_dim: Dimension of the L2 normalized bottleneck (e.g., 256)
        """
        super().__init__()
        
        # --- 1. Build the MLP ---
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
                
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
            
        # --- 2. Build the Weight-Normalized Last Layer (Prototypes) ---
        self.apply_norm_last_layer = norm_last_layer
        self.last_layer = weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        
        # Initialize magnitude to 1.0 (so it acts as pure Cosine Similarity)
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def forward(self, x, return_features=True):
        # --- Handle 3D Inputs (Batch, Time, Dim) ---
        is_3d = False
        if x.dim() == 3:
            is_3d = True
            B, T, D = x.shape
            # Flatten to (B*T, D) so BatchNorm1d works correctly
            x = x.reshape(B * T, D)

        # 1. MLP (Projects to bottleneck dim)
        x = self.mlp(x)
        
        # 2. L2 Normalize (The "Bottleneck" on the hypersphere)
        x = F.normalize(x, dim=-1, p=2)

        
        # 3. Prototype Matching (Cosine Similarity)
        x = self.last_layer(x)
        
        # --- Restore Shape ---
        if is_3d:
            # Reshape back to (B, T, out_dim)
            x = x.reshape(B, T, -1)
        if return_features:
            return x, x
        return x


class MLPModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.1, use_bn=True):
        super().__init__()
        layers = []
        
        for i in range(num_layers):
            # Determine input/output dims for this specific layer
            in_d = input_dim if i == 0 else hidden_dim
            out_d = output_dim if i == num_layers - 1 else hidden_dim
            
            # 1. Linear Layer
            layers.append(nn.Linear(in_d, out_d))
            
            # 2. Activation, BN, and Dropout (ONLY for hidden layers)
            # We do not apply these to the final output layer (standard SSL practice)
            if i < num_layers - 1:
                # BatchNorm is crucial for SSL stability
                if use_bn:
                    layers.append(nn.BatchNorm1d(out_d))
                
                layers.append(nn.ReLU())
                
                # Dropout prevents overfitting to specific features
                if dropout > 0.0:
                    layers.append(nn.Dropout(p=dropout))
                    
        self._mlp = nn.Sequential(*layers)
        

    def forward(self, x):
            is_3d = False
            if x.dim() == 3:
                is_3d = True
                B, T, D = x.shape
                # Flatten because BN/Linear usually expect 2D (N, C)
                x = x.reshape(B * T, D)
            
            # 1. Run MLP
            x = self._mlp(x)
            
            # 2. Apply Normalization (Crucial!)
            x = F.normalize(x, p=2, dim=-1)
            
            # 3. Reshape back if needed
            if is_3d:
                x = x.reshape(B, T, -1)
                
            return x


class FCPrediction(nn.Module):

    def __init__(self, feat_dim, num_classes):
        super().__init__()
        self._fc_out = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        batch_size, clip_len, _ = x.shape
        return self._fc_out(x.reshape(batch_size * clip_len, -1)).view(
            batch_size, clip_len, -1)


class GRUModule(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super().__init__()
        self._gru = nn.GRU(
            input_dim, hidden_dim, num_layers=num_layers, batch_first=True,
            bidirectional=True)

    def forward(self, x):
        y, _ = self._gru(x)
        return y

class GRUPrediction(nn.Module):

    def __init__(self, feat_dim, num_classes, hidden_dim, num_layers=1):
        super().__init__()
        self._gru = nn.GRU(
            feat_dim, hidden_dim, num_layers=num_layers, batch_first=True,
            bidirectional=True)
        self._fc_out = FCPrediction(2 * hidden_dim, num_classes)
        self._dropout = nn.Dropout()
        
    def extract_features(self, x):
        y, _ = self._gru(x)
        return y
    
    def forward(self, x, return_features=False):
        y, _ = self._gru(x)
        if return_features:
            return y, self._fc_out(self._dropout(y))
        else:
            return self._fc_out(self._dropout(y))
