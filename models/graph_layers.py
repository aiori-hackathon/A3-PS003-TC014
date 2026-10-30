"""
Spatial Graph Convolution Layers for STGCN.
Implements:
1. 1st-Order Normalized Graph Convolution (Kipf & Welling)
2. Chebyshev Spectral Graph Convolution (Defferrard et al., Yu et al.)
Weights are strictly node-invariant (parameter shapes depend solely on channel dims and K).
"""

from typing import Optional, List
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_chebyshev_polynomials(laplacian: np.ndarray, k_order: int = 2) -> List[np.ndarray]:
    """
    Computes Chebyshev polynomials T_k(L_tilde) up to order k_order - 1.
    T_0(L) = I
    T_1(L) = L_tilde = 2*L / lambda_max - I (with lambda_max approx 2.0 -> L_tilde = L - I)
    T_k(L) = 2 * L_tilde * T_{k-1}(L) - T_{k-2}(L)
    """
    num_nodes = laplacian.shape[0]
    # Scaled Laplacian: assuming lambda_max ~ 2.0
    l_tilde = laplacian - np.eye(num_nodes, dtype=laplacian.dtype)
    
    t_k = []
    t_0 = np.eye(num_nodes, dtype=laplacian.dtype)
    t_k.append(t_0)
    
    if k_order > 1:
        t_1 = l_tilde.copy()
        t_k.append(t_1)
        
    for k in range(2, k_order):
        t_next = 2.0 * (l_tilde @ t_k[k - 1]) - t_k[k - 2]
        t_k.append(t_next)
        
    return t_k


class ChebGraphConv(nn.Module):
    """
    Chebyshev Spectral Graph Convolutional Layer.
    Formula: y = sum_{k=0}^{K-1} T_k(L_tilde) * x * Theta_k
    
    Input shape:  (B, C_in, T, N)
    Output shape: (B, C_out, T, N)
    Learnable parameters: Theta of shape (K, C_in, C_out), bias (C_out, 1, 1)
    Parameters are completely independent of the number of nodes N.
    """
    def __init__(
        self,
        c_in: int,
        c_out: int,
        k_order: int = 2,
        bias: bool = True
    ):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.k_order = k_order
        
        # Kernel weights: (K, C_in, C_out)
        self.weight = nn.Parameter(torch.empty(k_order, c_in, c_out))
        if bias:
            self.bias = nn.Parameter(torch.empty(c_out))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        # Xavier uniform initialization
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, cheb_polynomials: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C_in, T, N)
        cheb_polynomials: (K, N, N)
        Returns: (B, C_out, T, N)
        """
        b, c_in, t, n = x.shape
        # Permute x to (B, T, N, C_in) for matrix multiplication
        x_perm = x.permute(0, 2, 3, 1)  # (B, T, N, C_in)
        
        # Output accumulator
        out = torch.zeros(b, t, n, self.c_out, device=x.device, dtype=x.dtype)
        
        for k in range(self.k_order):
            t_k = cheb_polynomials[k]  # (N, N)
            theta_k = self.weight[k]    # (C_in, C_out)
            
            # (N, N) x (B, T, N, C_in) -> (B, T, N, C_in)
            rhs = torch.einsum('nm, btmc -> btnc', t_k, x_perm)
            # (B, T, N, C_in) x (C_in, C_out) -> (B, T, N, C_out)
            out = out + torch.matmul(rhs, theta_k)
            
        # Permute back to (B, C_out, T, N)
        out = out.permute(0, 3, 1, 2)
        
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)
            
        return out


class GCNGraphConv(nn.Module):
    """
    1st-Order Normalized Graph Convolution (Kipf & Welling).
    Formula: y = A_norm * x * W
    Input shape:  (B, C_in, T, N)
    Output shape: (B, C_out, T, N)
    Weight shape: (C_in, C_out)
    """
    def __init__(self, c_in: int, c_out: int, bias: bool = True):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.weight = nn.Parameter(torch.empty(c_in, c_out))
        if bias:
            self.bias = nn.Parameter(torch.empty(c_out))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, a_norm: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C_in, T, N)
        a_norm: (N, N)
        """
        # x_perm: (B, T, N, C_in)
        x_perm = x.permute(0, 2, 3, 1)
        # (N, N) x (B, T, N, C_in) -> (B, T, N, C_in)
        ax = torch.einsum('nm, btmc -> btnc', a_norm, x_perm)
        # (B, T, N, C_in) x (C_in, C_out) -> (B, T, N, C_out)
        out = torch.matmul(ax, self.weight).permute(0, 3, 1, 2)
        
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)
        return out
