"""
Compact Spatio-Temporal Graph Convolutional Network (STGCN).
Optimized for communication-constrained Federated Learning on traffic networks.

Architecture:
  Input (B, T_in, N, C_in)
     │
     ▼
  ST-Conv Block 1 [Temporal GLU Conv -> ChebGraphConv -> Temporal GLU Conv -> Res + LN]
     │
     ▼
  ST-Conv Block 2 [Temporal GLU Conv -> ChebGraphConv -> Temporal GLU Conv -> Res + LN]
     │
     ▼
  Forecasting Output Head [Temporal Pooling/Conv -> Feature Projection]
     │
     ▼
  Output (B, T_out, N) [Predicted Traffic Flow]
"""

from typing import Optional, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.graph_layers import ChebGraphConv, compute_chebyshev_polynomials


class TemporalConvGLU(nn.Module):
    """
    Temporal Gated Linear Unit Convolution along the temporal dimension.
    Uses 2D convolution with kernel size (kernel_size, 1) over (B, C, T, N).
    Preserves time dimension via symmetric padding.
    """
    def __init__(
        self,
        c_in: int,
        c_out: int,
        kernel_size: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.kernel_size = kernel_size
        padding = ((kernel_size - 1) // 2, 0)
        
        # Convolves to 2 * c_out for GLU gate
        self.conv = nn.Conv2d(
            c_in,
            2 * c_out,
            kernel_size=(kernel_size, 1),
            padding=padding
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C_in, T, N)
        Returns: (B, C_out, T, N)
        """
        # (B, 2*C_out, T, N)
        conv_out = self.conv(x)
        p = conv_out[:, :self.c_out, :, :]
        q = conv_out[:, self.c_out:, :, :]
        # GLU gating: P * sigmoid(Q)
        out = p * torch.sigmoid(q)
        return self.dropout(out)


class STConvBlock(nn.Module):
    """
    Spatio-Temporal Convolutional Block:
    Temporal GLU -> Chebyshev Graph Conv -> LeakyReLU -> Temporal GLU -> Residual + LayerNorm
    """
    def __init__(
        self,
        c_in: int,
        c_hidden: int,
        c_out: int,
        k_order: int = 2,
        kt: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        
        # 1. Temporal Conv 1
        self.tconv1 = TemporalConvGLU(c_in, c_hidden, kernel_size=kt, dropout=dropout)
        
        # 2. Spatial Graph Conv
        self.sconv = ChebGraphConv(c_hidden, c_hidden, k_order=k_order)
        self.act = nn.LeakyReLU(0.2)
        
        # 3. Temporal Conv 2
        self.tconv2 = TemporalConvGLU(c_hidden, c_out, kernel_size=kt, dropout=dropout)
        
        # 4. Residual matching projection if c_in != c_out
        if c_in != c_out:
            self.res_conv = nn.Conv2d(c_in, c_out, kernel_size=1)
        else:
            self.res_conv = None
            
        # Normalization
        self.norm = nn.BatchNorm2d(c_out)

    def forward(self, x: torch.Tensor, cheb_poly: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C_in, T, N)
        cheb_poly: (K, N, N)
        Returns: (B, C_out, T, N)
        """
        res = x if self.res_conv is None else self.res_conv(x)
        
        # Temporal 1
        h1 = self.tconv1(x)  # (B, C_hidden, T, N)
        
        # Spatial Graph Conv
        h2 = self.sconv(h1, cheb_poly)  # (B, C_hidden, T, N)
        h2 = self.act(h2)
        
        # Temporal 2
        h3 = self.tconv2(h2)  # (B, C_out, T, N)
        
        # Residual + Norm
        out = self.norm(h3 + res)
        return out


class CompactSTGCN(nn.Module):
    """
    Compact Spatio-Temporal Graph Convolutional Network for Traffic Forecasting.
    
    Parameters:
        num_features: int (default=3, [flow, occupancy, speed])
        in_steps: int (default=12 historical timesteps = 1 hour)
        out_steps: int (default=3 forecast timesteps = 15 minutes)
        hidden_channels: int (default=32)
        k_order: int (Chebyshev order, default=2)
        kt: int (temporal kernel size, default=3)
        dropout: float (default=0.1)
    """
    def __init__(
        self,
        num_features: int = 3,
        in_steps: int = 12,
        out_steps: int = 3,
        hidden_channels: int = 32,
        k_order: int = 2,
        kt: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_features = num_features
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.hidden_channels = hidden_channels
        self.k_order = k_order
        
        # ST-Conv Blocks
        self.block1 = STConvBlock(
            c_in=num_features,
            c_hidden=hidden_channels,
            c_out=hidden_channels,
            k_order=k_order,
            kt=kt,
            dropout=dropout
        )
        
        self.block2 = STConvBlock(
            c_in=hidden_channels,
            c_hidden=hidden_channels,
            c_out=hidden_channels,
            k_order=k_order,
            kt=kt,
            dropout=dropout
        )
        
        # Forecasting Output Head
        # Projects temporal dimension from in_steps -> 1
        self.temporal_head = nn.Conv2d(
            hidden_channels,
            hidden_channels * 2,
            kernel_size=(in_steps, 1)
        )
        self.act_head = nn.LeakyReLU(0.2)
        
        # Projects channel dimension from hidden_channels*2 -> out_steps
        self.output_head = nn.Conv2d(
            hidden_channels * 2,
            out_steps,
            kernel_size=1
        )

    def forward(
        self,
        x: torch.Tensor,
        cheb_poly: torch.Tensor
    ) -> torch.Tensor:
        """
        x: (B, T_in, N, C_in) or (B, C_in, T_in, N)
        cheb_poly: (K, N, N)
        Returns:
            out: (B, T_out, N) predicted traffic flow
        """
        # If input is (B, T_in, N, C_in), permute to (B, C_in, T_in, N)
        if x.dim() == 4 and x.shape[1] == self.in_steps and x.shape[3] == self.num_features:
            x = x.permute(0, 3, 1, 2)  # (B, C_in, T_in, N)
            
        # ST-Conv Block 1
        h = self.block1(x, cheb_poly)
        
        # ST-Conv Block 2
        h = self.block2(h, cheb_poly)
        
        # Temporal pooling conv: (B, hidden, T_in, N) -> (B, hidden*2, 1, N)
        h = self.act_head(self.temporal_head(h))
        
        # Output channel projection: (B, hidden*2, 1, N) -> (B, out_steps, 1, N)
        out = self.output_head(h)
        
        # Squeeze singleton dimension -> (B, out_steps, N)
        out = out.squeeze(2)
        return out

    def get_parameter_count(self) -> Tuple[int, int]:
        """Returns (total_params, trainable_params)."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def get_model_size_kb(self) -> float:
        """Returns approximate uncompressed model size in KB (float32 = 4 bytes)."""
        total, _ = self.get_parameter_count()
        return (total * 4) / 1024.0
