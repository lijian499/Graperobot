from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


def conv3x3(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    dilation: int = 1,
) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        bias=False,
    )


def conv1x1(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        dilation: int = 1,
    ) -> None:
        super().__init__()

        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = conv3x3(
            planes,
            planes,
            stride=stride,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = conv1x1(
            planes,
            planes * self.expansion,
        )
        self.bn3 = nn.BatchNorm2d(
            planes * self.expansion,
        )

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.relu(out + identity)
        return out


class ModifiedResNet101(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.inplanes = 64

        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=7,
            stride=1,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.layer1 = self._make_layer(
            planes=64,
            blocks=3,
            stride=1,
            dilation=1,
        )
        self.layer2 = self._make_layer(
            planes=128,
            blocks=4,
            stride=2,
            dilation=1,
        )
        self.layer3 = self._make_layer(
            planes=256,
            blocks=23,
            stride=2,
            dilation=2,
        )
        self.layer4 = self._make_layer(
            planes=512,
            blocks=3,
            stride=1,
            dilation=4,
        )

        self.out_channels = (64, 256, 512, 1024, 2048)

        self._init_weights()

    def _make_layer(
        self,
        planes: int,
        blocks: int,
        stride: int,
        dilation: int,
    ) -> nn.Sequential:
        outplanes = planes * Bottleneck.expansion

        downsample = None
        if stride != 1 or self.inplanes != outplanes:
            downsample = nn.Sequential(
                conv1x1(
                    self.inplanes,
                    outplanes,
                    stride=stride,
                ),
                nn.BatchNorm2d(outplanes),
            )

        layers = [
            Bottleneck(
                self.inplanes,
                planes,
                stride=stride,
                downsample=downsample,
                dilation=dilation,
            )
        ]

        self.inplanes = outplanes

        for _ in range(1, blocks):
            layers.append(
                Bottleneck(
                    self.inplanes,
                    planes,
                    stride=1,
                    downsample=None,
                    dilation=dilation,
                )
            )

        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        x = self.relu(self.bn1(self.conv1(x)))
        c1 = x

        x = self.maxpool(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        return c1, c2, c3, c4, c5


def _load_torchvision_imagenet1k_weights(
    model: ModifiedResNet101,
) -> None:

    try:
        from torchvision.models import (
            ResNet101_Weights,
            resnet101 as torchvision_resnet101,
        )
    except ImportError as exc:
        raise ImportError(
            "torchvision is required when pretrained=True."
        ) from exc

    source = torchvision_resnet101(
        weights=ResNet101_Weights.IMAGENET1K_V1
    ).state_dict()

    target = model.state_dict()

    compatible = {
        key: value
        for key, value in source.items()
        if key in target and target[key].shape == value.shape
    }

    model.load_state_dict(
        compatible,
        strict=False,
    )


def resnet101(
    pretrained: bool = False,
) -> ModifiedResNet101:
    model = ModifiedResNet101()

    if pretrained:
        _load_torchvision_imagenet1k_weights(model)

    return model
