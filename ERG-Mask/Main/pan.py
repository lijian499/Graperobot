from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
    ) -> None:
        super().__init__()

        padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PAN(nn.Module):
    """
    Bottom-up path aggregation for F3, F4 and F5.

    F3 is at 1/4 scale.
    F4 and F5 are at 1/8 scale in the modified ERG-Mask backbone.
    """

    def __init__(
        self,
        channels: int = 256,
    ) -> None:
        super().__init__()

        self.p3_refine = ConvBNReLU(
            channels,
            channels,
            kernel_size=3,
            stride=1,
        )

        self.p3_down = ConvBNReLU(
            channels,
            channels,
            kernel_size=3,
            stride=2,
        )

        self.p4_refine = ConvBNReLU(
            channels,
            channels,
            kernel_size=3,
            stride=1,
        )

        self.p4_to_p5 = ConvBNReLU(
            channels,
            channels,
            kernel_size=3,
            stride=1,
        )

        self.p5_refine = ConvBNReLU(
            channels,
            channels,
            kernel_size=3,
            stride=1,
        )

        self.out_channels = channels

    def forward(
        self,
        inputs: Sequence[torch.Tensor],
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if len(inputs) != 3:
            raise ValueError("inputs must be [F3, F4, F5].")

        f3, f4, f5 = inputs

        p3 = self.p3_refine(f3)

        p3_down = self.p3_down(p3)
        if p3_down.shape[-2:] != f4.shape[-2:]:
            p3_down = F.interpolate(
                p3_down,
                size=f4.shape[-2:],
                mode="nearest",
            )

        p4 = self.p4_refine(
            p3_down + f4
        )

        p4_for_p5 = self.p4_to_p5(p4)
        if p4_for_p5.shape[-2:] != f5.shape[-2:]:
            p4_for_p5 = F.interpolate(
                p4_for_p5,
                size=f5.shape[-2:],
                mode="nearest",
            )

        p5 = self.p5_refine(
            p4_for_p5 + f5
        )

        return p3, p4, p5
