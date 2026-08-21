from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeSubnetwork(nn.Module):
    """
    Edge-prediction branch of ERG-Mask.

    C1-C5
      -> 1x1 Conv + ReLU
      -> additional 1x1 side-output Conv
      -> bilinear upsampling to the input resolution
      -> concatenation
      -> final 1x1 Conv
      -> sigmoid boundary probabilities
    """

    def __init__(
        self,
        in_channels: Sequence[int] = (
            64,
            256,
            512,
            1024,
            2048,
        ),
        hidden_channels: int = 64,
    ) -> None:
        super().__init__()

        if len(in_channels) != 5:
            raise ValueError(
                "in_channels must contain five values for C1-C5."
            )

        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        hidden_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True,
                    ),
                    nn.ReLU(inplace=True),
                )
                for channels in in_channels
            ]
        )

        self.side_heads = nn.ModuleList(
            [
                nn.Conv2d(
                    hidden_channels,
                    1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                )
                for _ in range(5)
            ]
        )

        self.fuse_head = nn.Conv2d(
            5,
            1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

    def forward(
        self,
        features: Sequence[torch.Tensor],
        output_size: Tuple[int, int],
        return_logits: bool = False,
    ) -> List[torch.Tensor]:
        if len(features) != 5:
            raise ValueError(
                "features must be [C1, C2, C3, C4, C5]."
            )

        side_logits: List[torch.Tensor] = []

        for feature, projection, side_head in zip(
            features,
            self.projections,
            self.side_heads,
        ):
            x = projection(feature)
            x = side_head(x)

            x = F.interpolate(
                x,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )

            side_logits.append(x)

        fused_logits = self.fuse_head(
            torch.cat(
                side_logits,
                dim=1,
            )
        )

        outputs = side_logits + [fused_logits]

        if return_logits:
            return outputs

        return [
            torch.sigmoid(output)
            for output in outputs
        ]


def class_balanced_bce_with_ignore(
    logits: torch.Tensor,
    target: torch.Tensor,
    mu: float = 1.0,
    ignore_index: int = 2,
) -> torch.Tensor:
    """
    Class-balanced weighted BCE with an ignore label.

    target:
        0 = non-boundary
        1 = boundary
        2 = ignore
    """
    if logits.ndim == 4 and logits.shape[1] == 1:
        logits = logits[:, 0]

    if target.ndim == 4 and target.shape[1] == 1:
        target = target[:, 0]

    if logits.shape != target.shape:
        raise ValueError(
            f"logits shape {logits.shape} "
            f"does not match target shape {target.shape}."
        )

    valid = target != ignore_index
    positive = (target == 1) & valid
    negative = (target == 0) & valid

    n_positive = positive.sum().to(logits.dtype)
    n_negative = negative.sum().to(logits.dtype)

    denominator = (
        n_positive + n_negative
    ).clamp_min(1.0)

    negative_weight = (
        n_positive / denominator
    )

    positive_weight = (
        mu * n_negative / denominator
    )

    loss = torch.zeros_like(logits)

    if positive.any():
        loss[positive] = (
            positive_weight
            * F.softplus(-logits[positive])
        )

    if negative.any():
        loss[negative] = (
            negative_weight
            * F.softplus(logits[negative])
        )

    if not valid.any():
        return logits.sum() * 0.0

    return loss[valid].mean()


def deep_supervision_edge_loss(
    logits_list: Sequence[torch.Tensor],
    target: torch.Tensor,
    mu: float = 1.0,
    ignore_index: int = 2,
) -> torch.Tensor:

    if len(logits_list) != 6:
        raise ValueError(
            "Expected five side outputs "
            "and one final fused output."
        )

    total_loss = logits_list[0].sum() * 0.0

    for logits in logits_list:
        total_loss = total_loss + class_balanced_bce_with_ignore(
            logits,
            target,
            mu=mu,
            ignore_index=ignore_index,
        )

    return total_loss
