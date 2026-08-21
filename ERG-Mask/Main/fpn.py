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


class PyramidFeatures(nn.Module):
    """
    Top-down feature fusion for C3, C4 and C5.
        C3: 1/4 scale, 512 channels
        C4: 1/8 scale, 1024 channels
        C5: 1/8 scale, 2048 channels
    """

    def __init__(
        self,
        in_channels: Sequence[int] = (512, 1024, 2048),
        out_channels: int = 256,
    ) -> None:
        super().__init__()

        if len(in_channels) != 3:
            raise ValueError(
                "in_channels must be [C3_channels, C4_channels, C5_channels]."
            )

        c3_channels, c4_channels, c5_channels = in_channels

        self.c3_lateral = nn.Conv2d(
            c3_channels,
            out_channels,
            kernel_size=1,
        )
        self.c4_lateral = nn.Conv2d(
            c4_channels,
            out_channels,
            kernel_size=1,
        )
        self.c5_lateral = nn.Conv2d(
            c5_channels,
            out_channels,
            kernel_size=1,
        )

        self.f3_smooth = ConvBNReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
        )
        self.f4_smooth = ConvBNReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
        )
        self.f5_smooth = ConvBNReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
        )

        self.out_channels = out_channels

    def forward(
        self,
        inputs: Sequence[torch.Tensor],
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if len(inputs) != 3:
            raise ValueError("inputs must be [C3, C4, C5].")

        c3, c4, c5 = inputs

        f5 = self.c5_lateral(c5)

        c4_lat = self.c4_lateral(c4)
        if c4_lat.shape[-2:] != f5.shape[-2:]:
            f5_for_c4 = F.interpolate(
                f5,
                size=c4_lat.shape[-2:],
                mode="nearest",
            )
        else:
            f5_for_c4 = f5

        f4 = c4_lat + f5_for_c4

        c3_lat = self.c3_lateral(c3)
        f4_up = F.interpolate(
            f4,
            size=c3_lat.shape[-2:],
            mode="nearest",
        )
        f3 = c3_lat + f4_up

        f3 = self.f3_smooth(f3)
        f4 = self.f4_smooth(f4)
        f5 = self.f5_smooth(f5)

        return f3, f4, f5
