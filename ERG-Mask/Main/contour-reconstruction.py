"""
ERG-Mask

This file demonstrates:
1. modified ResNet-101 backbone;
2. FPN top-down feature fusion;
3. PAN bottom-up feature aggregation;
4. edge-prediction branch;
5. polar-coordinate contour reconstruction.

"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch

from edges import EdgeSubnetwork
from fpn import PyramidFeatures
from pan import PAN
from resnet import resnet101


class PolarContourReconstructor:
    """
    Polar-coordinate contour reconstruction used by ERG-Mask.
    """

    def __init__(
        self,
        n_rays: int = 16,
        edge_threshold: float = 0.5,
        max_consecutive_missing: int = 2,
        closing_kernel: int = 3,
        closing_iterations: int = 1,
    ) -> None:
        self.n_rays = int(n_rays)
        self.edge_threshold = float(edge_threshold)
        self.max_consecutive_missing = int(
            max_consecutive_missing
        )
        self.closing_kernel = int(closing_kernel)
        self.closing_iterations = int(
            closing_iterations
        )

    def reconstruct(
        self,
        edge_probability,
        boxes_xyxy: Sequence[
            Sequence[float]
        ],
    ) -> List[Dict]:
        edge = self._to_2d_numpy(
            edge_probability
        )

        height, width = edge.shape

        binary = (
            edge >= self.edge_threshold
        ).astype(np.uint8)

        _, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                binary,
                connectivity=8,
            )
        )

        component_sizes = stats[
            :,
            cv2.CC_STAT_AREA,
        ]

        results: List[Dict] = []

        for box in boxes_xyxy:
            results.append(
                self._reconstruct_one(
                    binary,
                    labels,
                    component_sizes,
                    box,
                    width,
                    height,
                )
            )

        return results

    @staticmethod
    def _to_2d_numpy(
        edge_probability,
    ) -> np.ndarray:
        if isinstance(
            edge_probability,
            torch.Tensor,
        ):
            array = (
                edge_probability
                .detach()
                .float()
                .cpu()
                .numpy()
            )
        else:
            array = np.asarray(
                edge_probability
            )

        while (
            array.ndim > 2
            and array.shape[0] == 1
        ):
            array = array[0]

        if array.ndim != 2:
            raise ValueError(
                "edge_probability must reduce "
                f"to HxW, got {array.shape}."
            )

        return np.clip(
            array.astype(np.float32),
            0.0,
            1.0,
        )

    def _reconstruct_one(
        self,
        binary: np.ndarray,
        labels: np.ndarray,
        component_sizes: np.ndarray,
        box: Sequence[float],
        width: int,
        height: int,
    ) -> Dict:
        x1, y1, x2, y2 = map(
            float,
            box,
        )

        cx = float(
            np.clip(
                (x1 + x2) / 2.0,
                0,
                width - 1,
            )
        )
        cy = float(
            np.clip(
                (y1 + y2) / 2.0,
                0,
                height - 1,
            )
        )

        radii: List[
            float | None
        ] = [
            None
        ] * self.n_rays

        for ray_index in range(
            self.n_rays
        ):
            theta = (
                2.0
                * np.pi
                * ray_index
                / self.n_rays
            )

            (
                xs,
                ys,
                distances,
            ) = self._sample_ray(
                cx,
                cy,
                theta,
                width,
                height,
            )

            foreground = (
                binary[
                    ys,
                    xs,
                ]
                > 0
            )

            segments = (
                self._foreground_segments(
                    foreground
                )
            )

            if len(segments) == 0:
                continue

            if len(segments) == 1:
                start, _ = segments[0]
                radii[
                    ray_index
                ] = float(
                    distances[start]
                )
                continue

            encountered_labels = labels[
                ys[foreground],
                xs[foreground],
            ]

            encountered_labels = (
                encountered_labels[
                    encountered_labels > 0
                ]
            )

            if (
                encountered_labels.size
                == 0
            ):
                continue

            unique_labels = np.unique(
                encountered_labels
            )

            selected_label = int(
                max(
                    unique_labels,
                    key=lambda label: int(
                        component_sizes[
                            int(label)
                        ]
                    ),
                )
            )

            positions = np.where(
                labels[
                    ys,
                    xs,
                ]
                == selected_label
            )[0]

            if positions.size > 0:
                radii[
                    ray_index
                ] = float(
                    distances[
                        int(
                            positions[0]
                        )
                    ]
                )

        if (
            self._max_circular_missing_run(
                radii
            )
            > self.max_consecutive_missing
        ):
            return self._invalid_result(
                width,
                height,
                "More than two consecutive "
                "rays had no intersection.",
            )

        if all(
            radius is None
            for radius in radii
        ):
            return self._invalid_result(
                width,
                height,
                "No valid ray-boundary "
                "intersections.",
            )

        interpolated_radii = (
            self._interpolate_missing_radii(
                radii
            )
        )

        points = []

        for (
            ray_index,
            radius,
        ) in enumerate(
            interpolated_radii
        ):
            theta = (
                2.0
                * np.pi
                * ray_index
                / self.n_rays
            )

            x = (
                cx
                + radius
                * np.cos(theta)
            )
            y = (
                cy
                + radius
                * np.sin(theta)
            )

            x = int(
                round(
                    np.clip(
                        x,
                        0,
                        width - 1,
                    )
                )
            )
            y = int(
                round(
                    np.clip(
                        y,
                        0,
                        height - 1,
                    )
                )
            )

            points.append(
                (x, y)
            )

        contour = np.asarray(
            points,
            dtype=np.int32,
        )

        mask = np.zeros(
            (
                height,
                width,
            ),
            dtype=np.uint8,
        )

        cv2.fillPoly(
            mask,
            [
                contour.reshape(
                    -1,
                    1,
                    2,
                )
            ],
            1,
        )

        kernel = np.ones(
            (
                self.closing_kernel,
                self.closing_kernel,
            ),
            dtype=np.uint8,
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=(
                self.closing_iterations
            ),
        )

        (
            y_nonzero,
            x_nonzero,
        ) = np.where(
            mask > 0
        )

        if x_nonzero.size == 0:
            return self._invalid_result(
                width,
                height,
                "The reconstructed mask "
                "is empty.",
            )

        bbox = (
            int(x_nonzero.min()),
            int(y_nonzero.min()),
            int(x_nonzero.max()),
            int(y_nonzero.max()),
        )

        return {
            "valid": True,
            "reason": "",
            "contour": contour,
            "mask": mask,
            "bbox": bbox,
        }

    @staticmethod
    def _foreground_segments(
        foreground: np.ndarray,
    ) -> List[
        Tuple[int, int]
    ]:
        indices = np.flatnonzero(
            foreground
        )

        if indices.size == 0:
            return []

        split_points = np.where(
            np.diff(indices) > 1
        )[0]

        starts = np.r_[
            indices[0],
            indices[
                split_points + 1
            ],
        ]

        ends = np.r_[
            indices[
                split_points
            ],
            indices[-1],
        ]

        return [
            (
                int(start),
                int(end),
            )
            for start, end
            in zip(
                starts,
                ends,
            )
        ]

    @staticmethod
    def _sample_ray(
        cx: float,
        cy: float,
        theta: float,
        width: int,
        height: int,
    ):
        dx = float(
            np.cos(theta)
        )
        dy = float(
            np.sin(theta)
        )

        epsilon = 1e-12
        candidates = []

        if dx > epsilon:
            candidates.append(
                (
                    width
                    - 1
                    - cx
                )
                / dx
            )
        elif dx < -epsilon:
            candidates.append(
                (0 - cx) / dx
            )

        if dy > epsilon:
            candidates.append(
                (
                    height
                    - 1
                    - cy
                )
                / dy
            )
        elif dy < -epsilon:
            candidates.append(
                (0 - cy) / dy
            )

        positive = [
            value
            for value
            in candidates
            if value >= 0
        ]

        t_max = (
            min(positive)
            if positive
            else 0.0
        )

        n_steps = max(
            int(
                np.ceil(t_max)
            )
            + 1,
            2,
        )

        t = np.linspace(
            0.0,
            t_max,
            n_steps,
            dtype=np.float32,
        )

        xs = np.rint(
            cx + t * dx
        ).astype(np.int32)

        ys = np.rint(
            cy + t * dy
        ).astype(np.int32)

        xs = np.clip(
            xs,
            0,
            width - 1,
        )

        ys = np.clip(
            ys,
            0,
            height - 1,
        )

        keep = np.ones(
            len(xs),
            dtype=bool,
        )

        if len(xs) > 1:
            keep[1:] = (
                (
                    xs[1:]
                    != xs[:-1]
                )
                |
                (
                    ys[1:]
                    != ys[:-1]
                )
            )

        xs = xs[keep]
        ys = ys[keep]

        distances = np.sqrt(
            (
                xs.astype(
                    np.float32
                )
                - cx
            ) ** 2
            +
            (
                ys.astype(
                    np.float32
                )
                - cy
            ) ** 2
        )

        return (
            xs,
            ys,
            distances,
        )

    @staticmethod
    def _max_circular_missing_run(
        radii,
    ) -> int:
        missing = np.asarray(
            [
                radius is None
                for radius
                in radii
            ],
            dtype=np.uint8,
        )

        n = len(missing)

        if missing.sum() == 0:
            return 0

        if missing.sum() == n:
            return n

        doubled = np.concatenate(
            [
                missing,
                missing,
            ]
        )

        best = 0
        current = 0

        for value in doubled:
            if value:
                current += 1
                best = max(
                    best,
                    current,
                )
            else:
                current = 0

        return min(
            best,
            n,
        )

    @staticmethod
    def _interpolate_missing_radii(
        radii,
    ) -> List[float]:
        output = list(
            radii
        )
        n = len(output)

        valid_count = sum(
            radius is not None
            for radius in output
        )

        if valid_count == 0:
            raise ValueError(
                "No valid radii "
                "are available."
            )

        for i in range(n):
            if output[i] is not None:
                continue

            previous_distance = 1
            while (
                output[
                    (
                        i
                        - previous_distance
                    )
                    % n
                ]
                is None
            ):
                previous_distance += 1

            next_distance = 1
            while (
                output[
                    (
                        i
                        + next_distance
                    )
                    % n
                ]
                is None
            ):
                next_distance += 1

            previous_index = (
                i
                - previous_distance
            ) % n

            next_index = (
                i
                + next_distance
            ) % n

            previous_radius = float(
                output[
                    previous_index
                ]
            )
            next_radius = float(
                output[
                    next_index
                ]
            )

            alpha = (
                previous_distance
                /
                float(
                    previous_distance
                    + next_distance
                )
            )

            output[i] = (
                (
                    1.0
                    - alpha
                )
                * previous_radius
                +
                alpha
                * next_radius
            )

        return [
            float(radius)
            for radius in output
        ]

    @staticmethod
    def _invalid_result(
        width: int,
        height: int,
        reason: str,
    ) -> Dict:
        return {
            "valid": False,
            "reason": reason,
            "contour": np.empty(
                (0, 2),
                dtype=np.int32,
            ),
            "mask": np.zeros(
                (
                    height,
                    width,
                ),
                dtype=np.uint8,
            ),
            "bbox": None,
        }


def network_smoke_test() -> None:

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    backbone = resnet101(
        pretrained=False
    ).to(device)

    fpn = PyramidFeatures().to(
        device
    )

    pan = PAN().to(
        device
    )

    edge_branch = EdgeSubnetwork().to(
        device
    )

    backbone.eval()
    fpn.eval()
    pan.eval()
    edge_branch.eval()

    x = torch.randn(
        1,
        3,
        64,
        64,
        device=device,
    )

    with torch.no_grad():
        (
            c1,
            c2,
            c3,
            c4,
            c5,
        ) = backbone(x)

        (
            f3,
            f4,
            f5,
        ) = fpn(
            (
                c3,
                c4,
                c5,
            )
        )

        (
            p3,
            p4,
            p5,
        ) = pan(
            (
                f3,
                f4,
                f5,
            )
        )

        edge_outputs = edge_branch(
            (
                c1,
                c2,
                c3,
                c4,
                c5,
            ),
            output_size=(
                64,
                64,
            ),
        )

    print(
        "C shapes:",
        [
            tuple(t.shape)
            for t in (
                c1,
                c2,
                c3,
                c4,
                c5,
            )
        ],
    )

    print(
        "P shapes:",
        [
            tuple(t.shape)
            for t in (
                p3,
                p4,
                p5,
            )
        ],
    )

    print(
        "Edge map:",
        tuple(
            edge_outputs[-1].shape
        ),
    )


def contour_smoke_test() -> None:

    edge_map = np.zeros(
        (
            512,
            512,
        ),
        dtype=np.float32,
    )

    cv2.circle(
        edge_map,
        center=(
            256,
            256,
        ),
        radius=100,
        color=1.0,
        thickness=2,
    )

    predicted_boxes = [
        [
            150,
            150,
            360,
            360,
        ]
    ]

    reconstructor = (
        PolarContourReconstructor(
            n_rays=16,
            edge_threshold=0.5,
        )
    )

    result = (
        reconstructor.reconstruct(
            edge_map,
            predicted_boxes,
        )[0]
    )

    print(
        "Contour valid:",
        result["valid"],
    )

    print(
        "Reconstructed bbox:",
        result["bbox"],
    )

    print(
        "Contour points:",
        result[
            "contour"
        ].shape[0],
    )


if __name__ == "__main__":
    network_smoke_test()
    contour_smoke_test()
