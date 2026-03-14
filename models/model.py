import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from preprocessing import SkeletonMultiModalViews



def _default_channel_schedule() -> List[int]:
    """Returns [48, 48, 48, 96, 96] — the paper's L=5 backbone configuration."""
    return [48, 48, 48, 96, 96]



class ATTGCN(nn.Module):
    """
    Self-Attention Enhanced Graph Convolution (ATT-GCN).

    Args:
        in_channels  : C  — input feature channels per joint per frame.
        out_channels : K  — output feature channels.
        num_joints   : V  — number of skeleton joints.
    """

    def __init__(self, in_channels: int, out_channels: int, num_joints: int):
        super().__init__()
        self.K = out_channels
        self.V = num_joints

        self.A = nn.Parameter(
            torch.eye(num_joints).unsqueeze(0).repeat(out_channels, 1, 1)
        )
        self.lambda_scale = nn.Parameter(torch.tensor(0.1))

        self.phi   = nn.Linear(in_channels, out_channels)   # embedding φ
        self.theta = nn.Linear(in_channels, out_channels)   # embedding θ
        self.kappa = nn.Linear(out_channels, out_channels)  # projection κ
        self.tau   = nn.Tanh()                              # activation τ

        self.xi = nn.Linear(in_channels, out_channels)

        self.att_q = nn.Linear(out_channels, out_channels)
        self.att_k = nn.Linear(out_channels, out_channels)

        reduced       = max(out_channels // 4, 1)
        self.ca_conv1 = nn.Conv1d(out_channels, reduced, kernel_size=1)
        self.ca_norm  = nn.LayerNorm(reduced)
        self.ca_conv2 = nn.Conv1d(reduced, out_channels, kernel_size=1)

        self.out_norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x   : [B, T, V, C_in]
        Returns:
            Z_G : [B, T, V, K]
        """
        B, T, V, C = x.shape
        K = self.K

        phi_x   = self.phi(x).mean(dim=1)       # [B, V, K]
        theta_x = self.theta(x).mean(dim=1)     # [B, V, K]

        F_ij = self.tau(
            phi_x.unsqueeze(2) - theta_x.unsqueeze(1)
        )

        A_hat = self.kappa(F_ij).permute(0, 3, 1, 2)

        A_shared = self.A.unsqueeze(0).expand(B, -1, -1, -1)   # [B, K, V, V]
        A_tilde  = A_shared + self.lambda_scale * A_hat         # [B, K, V, V]

        H = self.xi(x)                                          # [B, T, V, K]

        A_t = (A_tilde
               .unsqueeze(1)
               .expand(B, T, K, V, V)
               .reshape(B * T * K, V, V))

        H_ck = (H.reshape(B * T, V, K)
                  .permute(0, 2, 1)                             # [B*T, K, V]
                  .reshape(B * T * K, V, 1))

        Z_gcn = torch.bmm(A_t, H_ck).squeeze(-1)               # [B*T*K, V]
        Z_gcn = (Z_gcn
                 .reshape(B, T, K, V)
                 .permute(0, 1, 3, 2))                          # [B, T, V, K]

        XQ = self.att_q(Z_gcn)                                  # [B, T, V, K]
        XK = self.att_k(Z_gcn)                                  # [B, T, V, K]

        XQ_p = XQ.permute(0, 3, 1, 2).mean(dim=2)              # [B, K, V]
        XK_p = XK.permute(0, 3, 1, 2).mean(dim=2)              # [B, K, V]

        scale = math.sqrt(T)
        A_att = torch.softmax(
            torch.bmm(
                XQ_p.reshape(B * K, V, 1),
                XK_p.reshape(B * K, 1, V),
            ) / scale,
            dim=-1,
        )

        H_ck2 = H.permute(0, 3, 1, 2).reshape(B * K, T, V)    # [B*K, T, V]
        Z_A   = torch.bmm(H_ck2, A_att.transpose(1, 2))        # [B*K, T, V]
        Z_A   = (Z_A
                 .reshape(B, K, T, V)
                 .permute(0, 2, 3, 1))                          # [B, T, V, K]

        ca = Z_A.mean(dim=(1, 2)).unsqueeze(-1)
        ca = self.ca_conv1(ca)                                   # [B, K//4, 1]
        ca = F.gelu(self.ca_norm(ca.squeeze(-1)).unsqueeze(-1))
        ca = torch.sigmoid(self.ca_conv2(ca))                    # [B, K, 1]

        A_tilde_c = A_tilde * ca.unsqueeze(-1)                   # [B, K, V, V]

        A_c_t = (A_tilde_c
                 .unsqueeze(1)
                 .expand(B, T, K, V, V)
                 .reshape(B * T * K, V, V))

        H_ck3 = (H.reshape(B * T, V, K)
                   .permute(0, 2, 1)
                   .reshape(B * T * K, V, 1))

        Z_G = torch.bmm(A_c_t, H_ck3).squeeze(-1)               # [B*T*K, V]
        Z_G = (Z_G
               .reshape(B, T, K, V)
               .permute(0, 1, 3, 2))                             # [B, T, V, K]

        return self.out_norm(Z_G)


class MSTCN(nn.Module):
    """
    Multi-Scale Temporal Convolution (MS-TCN).
	
    Args:
        in_channels : input channel count (= ATT-GCN out_channels).
        num_joints  : V — treated as an independent batch dimension.
        branches    : list of (kernel_size, dilation) per branch.
    """

    _DEFAULT_BRANCHES: List[Tuple[int, int]] = [(3, 1), (3, 2), (3, 4)]

    def __init__(
        self,
        in_channels: int,
        num_joints: int,
        branches: Optional[List[Tuple[int, int]]] = None,
    ):
        super().__init__()
        if branches is None:
            branches = self._DEFAULT_BRANCHES
        self.V = num_joints
        n = len(branches)

        base_out = in_channels // n
        branch_outs = [base_out] * (n - 1) + [in_channels - base_out * (n - 1)]

        self.branch_list = nn.ModuleList()
        for out_c, (ks, dil) in zip(branch_outs, branches):
            pad = (ks - 1) * dil // 2
            self.branch_list.append(nn.Sequential(
                nn.Conv1d(in_channels, out_c, kernel_size=ks,
                          dilation=dil, padding=pad, bias=False),
                nn.BatchNorm1d(out_c),
                nn.ReLU(inplace=True),
            ))

        total_out = sum(branch_outs)
        self.residual = nn.Sequential(
            nn.Conv1d(in_channels, total_out, kernel_size=1, bias=False),
            nn.BatchNorm1d(total_out),
        )
        self.out_norm  = nn.LayerNorm(total_out)
        self._out_channels = total_out

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x   : [B, T, V, C_in]
        Returns:
            out : [B, T, V, C_out]
        """
        B, T, V, C = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(B * V, C, T)  # [B*V, C, T]

        out = torch.cat([b(x_flat) for b in self.branch_list], dim=1)
        out = out + self.residual(x_flat)                      # [B*V, C_out, T]

        out = (out
               .reshape(B, V, self._out_channels, T)
               .permute(0, 3, 1, 2))                           # [B, T, V, C_out]
        return self.out_norm(out)



class SpatioTemporalLayer(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, num_joints: int):
        super().__init__()
        self.att_gcn = ATTGCN(in_channels, out_channels, num_joints)
        self.ms_tcn  = MSTCN(out_channels, num_joints)

    @property
    def out_channels(self) -> int:
        return self.ms_tcn.out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.att_gcn(x)
        z = self.ms_tcn(z)
        return z


class GraphEncoder(nn.Module):

    def __init__(
        self,
        in_channels: int,
        num_joints: int,
        channel_schedule: Optional[List[int]] = None,
    ):
        super().__init__()
        if channel_schedule is None:
            channel_schedule = _default_channel_schedule()

        layers, prev_c = [], in_channels
        for out_c in channel_schedule:
            layer  = SpatioTemporalLayer(prev_c, out_c, num_joints)
            layers.append(layer)
            prev_c = layer.out_channels   # actual channels after MS-TCN split

        self.layers   = nn.ModuleList(layers)
        self._out_dim = prev_c

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x   : [B, T, V, C_in]
        Returns:
            E_G : [B, out_dim]
        """
        for layer in self.layers:
            x = layer(x)
        return x.mean(dim=(1, 2))   # global avg-pool over T and V → [B, out_dim]


class MLPViewEncoder(nn.Module):

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.fc1      = nn.Linear(in_dim, hidden_dim)
        self.norm     = nn.LayerNorm(hidden_dim)
        self.fc2      = nn.Linear(hidden_dim, out_dim)
        self._out_dim = out_dim

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, T, D]
        Returns:
            E : [B, out_dim]
        """
        h = F.relu(self.norm(self.fc1(x)))   # [B, T, hidden_dim]
        h = self.fc2(h)                       # [B, T, out_dim]
        return h.mean(dim=1)                  # temporal avg-pool → [B, out_dim]


class SpatioTemporalEncoder(nn.Module):

    def __init__(
        self,
        num_joints: int,
        coord_dim: int = 3,
        channel_schedule: Optional[List[int]] = None,
        mlp_hidden: int = 128,
        mlp_out: int = 64,
    ):
        super().__init__()
        if channel_schedule is None:
            channel_schedule = _default_channel_schedule()

        self.view_builder = SkeletonMultiModalViews()

        self.graph_encoder = GraphEncoder(
            in_channels=coord_dim,
            num_joints=num_joints,
            channel_schedule=channel_schedule,
        )
        graph_out = self.graph_encoder.out_dim

        self.joint_encoder  = MLPViewEncoder(num_joints * coord_dim,     mlp_hidden, mlp_out)
        self.motion_encoder = MLPViewEncoder(num_joints * coord_dim * 2, mlp_hidden, mlp_out)

        self._out_dim = graph_out + mlp_out + mlp_out

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def forward(self, skeleton: torch.Tensor) -> torch.Tensor:
        """
        Args:
            skeleton : [B, T, V, C]
        Returns:
            E        : [B, out_dim]
        """
        node_feats, s_joint, s_motion = self.view_builder(skeleton)

        E_G = self.graph_encoder(node_feats)    # [B, graph_out]
        E_J = self.joint_encoder(s_joint)        # [B, mlp_out]
        E_M = self.motion_encoder(s_motion)      # [B, mlp_out]

        return torch.cat([E_G, E_J, E_M], dim=-1)   # [B, out_dim]


class CrossAttentionFusion(nn.Module):
    """
    Bidirectional cross-attention fusion for TUG and FTSTS embeddings.

    Args:
        embed_dim : C′ — dimension of input embeddings E^T and E^F.
        attn_dim  : d  — projection dimension for Q, K, V (default 64).
    """

    def __init__(self, embed_dim: int, attn_dim: int = 64):
        super().__init__()
        self.WQ_T = nn.Linear(embed_dim, attn_dim)
        self.WK_F = nn.Linear(embed_dim, attn_dim)
        self.WV_F = nn.Linear(embed_dim, attn_dim)

        self.WQ_F = nn.Linear(embed_dim, attn_dim)
        self.WK_T = nn.Linear(embed_dim, attn_dim)
        self.WV_T = nn.Linear(embed_dim, attn_dim)

        self._scale   = math.sqrt(attn_dim)
        self._out_dim = attn_dim * 2

    @property
    def out_dim(self) -> int:
        return self._out_dim

    @staticmethod
    def _cross_attend(
        Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, scale: float
    ) -> torch.Tensor:
        """Single-head cross-attention for 1-D embeddings."""
        Q = Q.unsqueeze(1)   # [B, 1, d]
        K = K.unsqueeze(1)
        V = V.unsqueeze(1)
        attn = torch.softmax(
            torch.bmm(Q, K.transpose(1, 2)) / scale, dim=-1
        )                    # [B, 1, 1]
        return torch.bmm(attn, V).squeeze(1)   # [B, d]

    def forward(self, E_T: torch.Tensor, E_F: torch.Tensor) -> torch.Tensor:
        """
        Args:
            E_T : [B, C′]  TUG embedding.
            E_F : [B, C′]  FTSTS embedding.
        Returns:
            E_fusion : [B, 2*attn_dim]
        """
        E_hat_T = self._cross_attend(
            self.WQ_T(E_T), self.WK_F(E_F), self.WV_F(E_F), self._scale
        )
        E_hat_F = self._cross_attend(
            self.WQ_F(E_F), self.WK_T(E_T), self.WV_T(E_T), self._scale
        )
        return torch.cat([E_hat_T, E_hat_F], dim=-1)   # [B, 2d]


class MLPClassifier(nn.Module):

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)



class FallRiskAssessmentModel(nn.Module):

    def __init__(
        self,
        num_joints: int = 49,
        coord_dim: int = 3,
        channel_schedule: Optional[List[int]] = None,
        mlp_hidden: int = 128,
        mlp_out: int = 64,
        attn_dim: int = 64,
        cls_hidden: int = 256,
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        if channel_schedule is None:
            channel_schedule = _default_channel_schedule()

        self.encoder = SpatioTemporalEncoder(
            num_joints=num_joints,
            coord_dim=coord_dim,
            channel_schedule=channel_schedule,
            mlp_hidden=mlp_hidden,
            mlp_out=mlp_out,
        )

        self.fusion = CrossAttentionFusion(
            embed_dim=self.encoder.out_dim,
            attn_dim=attn_dim,
        )

        self.classifier = MLPClassifier(
            in_dim=self.fusion.out_dim,
            hidden_dim=cls_hidden,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(
        self,
        tug_skeleton: torch.Tensor,
        ftsts_skeleton: torch.Tensor,
    ) -> torch.Tensor:
        
        E_T      = self.encoder(tug_skeleton)     # [B, C′]
        E_F      = self.encoder(ftsts_skeleton)   # [B, C′]
        E_fusion = self.fusion(E_T, E_F)          # [B, 2d]
        return self.classifier(E_fusion)          # [B, num_classes]

    def loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        
        return F.cross_entropy(logits, labels)
