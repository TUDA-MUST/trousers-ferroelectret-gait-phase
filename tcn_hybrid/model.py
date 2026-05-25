"""
Multi-head temporal convolutional network for residual phase correction.

The network receives a 167-sample, three-channel window of bandpass-filtered
ferroelectret signals (M.~Gastrocnemius, M.~Vastus Medialis, M.~Rectus Femoris)
at 110 Hz and produces three outputs:

    head_abs  : (sin, cos) of the predicted absolute gait phase
    head_add  : (sin, cos) of the predicted additive correction phi_GT - phi_AO
    head_conf : a scalar in [0, 1] indicating how much to trust the additive
                head versus the absolute head at the current sample.

The hybrid estimator in `inference.py` blends the AO's own phase, the
additive correction, and the absolute fallback using the confidence head as
the blend weight. See the paper supplementary Sections S1 and S2 for the
ablation that selected this architecture (7 layers, 48 filters per layer,
dilations [1, 2, 4, 8, 16, 64, 128], kernel 5, GroupNorm at the input,
BatchNorm inside each block, dropout 0.2, residual skip connections from
the second layer onward).

Why the architecture looks like this
------------------------------------
The receptive field at 110 Hz is 893 samples (~8 s, roughly 8 strides),
which lets the network infer the current cadence and stride frequency from
the periodic structure of the bandpass signals without any explicit input
about the AO's state. The GroupNorm at the input normalizes the overall
signal level while preserving the inter-channel amplitude ratios that
encode relative muscle activation; per-channel InstanceNorm destroys this
information and degrades cross-sensor generalization. Skip connections
ensure that gradients survive through the seven dilated layers during
training; at inference time they let lower-frequency features bypass the
deeper layers.
"""

from typing import Sequence

import torch
import torch.nn as nn


class CausalConv1d(nn.Module):
    """Causal 1-D dilated convolution.

    The output at time t depends only on inputs at times t' <= t. This is
    achieved by left-padding the input with (kernel_size - 1) * dilation
    zeros before applying a standard (uncausal) Conv1d.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        # The attribute is named `c` (rather than the longer `conv`) to
        # match the parameter names in the published checkpoint
        # `weights/tcn_final_7L48f.pt`.
        self.c = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c(nn.functional.pad(x, (self.pad, 0)))


class MultiHeadTCN(nn.Module):
    """Seven-layer causal TCN with three output heads.

    The defaults reproduce the paper's selected configuration. The dilation
    sequence [1, 2, 4, 8, 16, 64, 128] doubles up to layer 5, then jumps to
    cover the multi-stride context needed for cadence inference without the
    parameter cost of a fully exponential schedule.
    """

    def __init__(
        self,
        in_channels: int = 3,
        n_filters: int = 48,
        dilations: Sequence[int] = (1, 2, 4, 8, 16, 64, 128),
        kernel_size: int = 5,
        dropout: float = 0.2,
    ):
        super().__init__()

        # Input normalization: GroupNorm with one group is equivalent to
        # LayerNorm over the (channel, time) dimensions. This rescales the
        # overall signal level without flattening the cross-channel
        # amplitude differences.
        self.norm = nn.GroupNorm(1, in_channels, affine=True)

        # Dilated causal convolutions with per-layer BatchNorm. The first
        # layer maps in_channels -> n_filters; subsequent layers keep the
        # channel count constant so residual connections can be added in
        # the forward pass.
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(CausalConv1d(in_channels, n_filters, kernel_size, dilations[0]))
        self.bns.append(nn.BatchNorm1d(n_filters))
        for d in dilations[1:]:
            self.convs.append(CausalConv1d(n_filters, n_filters, kernel_size, d))
            self.bns.append(nn.BatchNorm1d(n_filters))

        self.drop = nn.Dropout(dropout)

        # Three task-specific output heads. All three share the same
        # convolutional backbone, which forces the shared representation
        # to be useful for all three tasks at once. Supplementary Section
        # S2 documents the gain from this multi-task setup.
        self.head_abs = nn.Linear(n_filters, 2)    # sin/cos of absolute phase
        self.head_add = nn.Linear(n_filters, 2)    # sin/cos of additive correction
        self.head_conf = nn.Linear(n_filters, 1)   # scalar confidence weight

    def forward(self, x: torch.Tensor):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor of shape (batch, 3, 167)
            Bandpass-filtered, 110 Hz-sampled multi-sensor window. The
            most recent sample is at index -1.

        Returns
        -------
        (abs_out, add_out, conf_out)
            abs_out : (batch, 2) — sin(phi_pred), cos(phi_pred)
            add_out : (batch, 2) — sin(delta_phi), cos(delta_phi)
            conf_out : (batch, 1) — sigmoid confidence in the additive head
        """
        x = self.norm(x)

        # First convolution after normalization, no residual yet.
        h = self.drop(torch.relu(self.bns[0](self.convs[0](x))))

        # Remaining layers: each adds its activation to the running sum to
        # form a dense skip connection across the depth of the network.
        for i in range(1, len(self.convs)):
            h = self.drop(torch.relu(self.bns[i](self.convs[i](h)))) + h

        # Take only the most recent time step. The TCN's causal padding
        # guarantees that this scalar feature vector summarizes the entire
        # past window up to the present sample.
        features = h[:, :, -1]

        abs_out = self.head_abs(features)
        add_out = self.head_add(features)
        conf_out = torch.sigmoid(self.head_conf(features))
        return abs_out, add_out, conf_out
